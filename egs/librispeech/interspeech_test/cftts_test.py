import argparse
import json
import os
from pathlib import Path
import re
import shutil
from tqdm import tqdm
import yaml
import numpy as np
import pandas as pd
import torch
import torchaudio
from pyannote.audio import Pipeline

from linearvc import linearvc
from linearvc.cf_tts.models.tts import ZipVoice

from linearvc.cf_tts.utils.common import normalize_input, invert_normalized_input, load_config, get_speaker_feats
from linearvc.cf_tts.utils.checkpoints import load_checkpoint


# -------------------------
# main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech_root",
        type=Path,
    )
    parser.add_argument(
        "--librispeech_transcript_root",
        type=Path,
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="output speech directory",
    )
    parser.add_argument(
        "--num_utt_per_speaker",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--feat_path",
        type=Path,
    )
    parser.add_argument(
        "--frame_limit",
        type=int,
        default=500
    )
    parser.add_argument(
        "--config", 
        required=True
    )
    parser.add_argument(
        "--set",
        type=str,
        help="librispeech, cv",
        default="cv"
    )
    parser.add_argument(
        "--inference_file_dir",
        type=Path,
        default="linearvc/exp/tts/inference/inference_0.tsv",
        help="For CV only"
    )
    parser.add_argument(
        "--phone_duration_file",
        type=Path,
        default="linearvc/exp/asr/LibriSpeech/spm/phone_durations.json"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sampling_steps", default=16, type=int)
    parser.add_argument("--vad", default=True, type=bool)
    args = parser.parse_args()

    librispeech_root = Path(args.librispeech_root)
    librispeech_transcript_root = Path(args.librispeech_transcript_root)
    feat_path = Path(args.feat_path)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)

    with open('linearvc/egs/librispeech/libri_test/speakers.json', 'r') as file:
        speakers = json.load(file)
    with open(args.phone_duration_file, 'r') as file:
        phone_durations = json.load(file)
    
    in_speakers = speakers['lists']['test-clean_source']
    out_speakers = speakers['lists']['test-other_target']
    maps = speakers['maps']

    # -------------------------
    # load models
    # -------------------------

    print("Loading ZipVoice...")
    model = ZipVoice(**cfg["model"]["tts"]["zipvoice"]).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    print("Loading WavLM + HiFiGAN...")
    wavlm = torch.hub.load(
        "bshall/knn-vc",
        "wavlm_large",
        trust_repo=True,
        device=device
    )
    if cfg["model"]["hifigan_path"] == 'none':
        hifigan, _ = torch.hub.load(
            "bshall/knn-vc",
            "hifigan_wavlm",
            trust_repo=True,
            prematched=True,
            device=device
        )
    else:
        from linearvc.hifigan.models import Generator
        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self
        def load_hifigan_checkpoint(filepath, device):
            assert os.path.isfile(filepath)
            print("Loading '{}'".format(filepath))
            checkpoint_dict = torch.load(filepath, map_location=device)
            print("Complete.")
            return checkpoint_dict
        config_file = os.path.join(os.path.split(cfg["model"]["hifigan_path"])[0], 'config.json')
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        hifigan = Generator(h).to(device)
        state_dict_g = load_hifigan_checkpoint(cfg["model"]["hifigan_path"], device)
        hifigan.load_state_dict(state_dict_g['generator'])
    if cfg['training']['content_factorization']['type'] != 'content':
        knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
        matching_sets = {}
        for out_speaker in tqdm(out_speakers):
            spk_wavs = (librispeech_root / 'test-other' / out_speaker).rglob('*.flac')
            matching_set = knn_vc.get_matching_set(spk_wavs, vad_trigger_level=0)
            matching_sets[out_speaker] = matching_set
    linearvc_model = linearvc.LinearVC(wavlm, hifigan, device)
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
    pipeline.to(device)

    # -------------------------
    # load transform
    # -------------------------

    resamplers = {}
    if cfg['training']['content_factorization']['type'] == 'content':
        if cfg["training"]['content_factorization']["content_factorization_file"] is not None:
            transforms = np.load(cfg["training"]['content_factorization']["content_factorization_file"], allow_pickle=True).item()
            transform = torch.tensor(np.linalg.pinv(transforms[list(transforms.keys())[0]])).to(device)
        else:
            transform = None
    elif cfg['training']['content_factorization']['type'] == 'speaker':
        from cuvs.neighbors import brute_force
        feats = get_speaker_feats(
            tgt_speaker_root=cfg['training']['content_factorization']['factorization_speaker'],
            linearvc_model=linearvc_model,
            device=device
        )
        index = brute_force.build(feats)
        transform = {
            'feats': feats,
            'index': index,
            'brute_force': brute_force
        }

    # -------------------------
    # load tokenizer
    # -------------------------

    if cfg['data']['text_tokenizer']['type'] == 'spm':
        import sentencepiece as spm
        print("Loading SentencePiece...")
        sp = spm.SentencePieceProcessor()
        sp.load(cfg['data']['text_tokenizer']['tokenizer_file'])
    elif cfg['data']['text_tokenizer']['type'] == 'phone':
        text_tokenizer = {}
        punctuation_marks = ''
        with open(cfg['data']['text_tokenizer']['tokenizer_file'], 'r') as phone_json_file:
            phones = json.load(phone_json_file)
            for category in phones.keys():
                category_tokens = phones[category]
                for token_id, token in enumerate(category_tokens):
                    text_tokenizer[token] = token_id
                    if category == 'punctuations':
                        punctuation_marks += token
        if cfg['data']['text_tokenizer']['pre_phonemized']:
            from speechbrain.inference.text import GraphemeToPhoneme
            g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", savedir="pretrained_models/soundchoice-g2p")
        else:
            from phonemizer.backend import EspeakBackend
            from phonemizer.separator import Separator
            separator = Separator(phone='-', word=' ')
            backend = EspeakBackend(
                language='en-us', 
                preserve_punctuation=True, 
                punctuation_marks=punctuation_marks,
                words_mismatch='ignore'
            )
            phonemize = backend.phonemize


    # -------------------------
    # Inference
    # -------------------------

    if cfg['training']['content_factorization']['type'] == 'content':
        transforms_target = {}
        for out_speaker in out_speakers:
            out_feats = np.load(feat_path / 'test-other' / (out_speaker + '.npy'))
            out_feats = torch.tensor(out_feats[:args.frame_limit]).to(device).float()
            transform_content = torch.matmul(out_feats, transform).squeeze(0)
            transform_tgt = torch.matmul(torch.linalg.pinv(transform_content), out_feats.squeeze(1)) # [r, 1024]
            transform_tgt = torch.tensor(transform_tgt).to(device)
            transforms_target[out_speaker] = transform_tgt

    if args.set == 'librispeech':
        for in_speaker in tqdm(in_speakers):
            (out_dir / in_speaker).mkdir(parents=True, exist_ok=True)
            spk_wavs = list((librispeech_root / 'test-clean' / in_speaker).rglob('*.flac'))[:args.num_utt_per_speaker]
            transcripts = {}
            with open(librispeech_transcript_root / 'test-clean' / (in_speaker + '.txt'), 'r') as transcript_file:
                for line in transcript_file:
                    line_elements = line.strip().split('|')
                    transcripts[line_elements[0]] = line_elements[1]
            for spk_wav in spk_wavs:
                transcript = transcripts[spk_wav.stem]
                prompt_features = torch.tensor([[[]]], device=device)
                prompt_tokens = [[]]

                # tokenize text
                if cfg['data']['text_tokenizer']['type'] == 'spm':
                    tokens = [sp.encode_as_ids(inference_item['text'])]
                elif cfg['data']['text_tokenizer']['type'] == 'phone':
                    if cfg['data']['text_tokenizer']['pre_phonemized']:
                        text = g2p(transcript)
                    else:
                        text = phonemize([transcript], separator=separator, strip=True)
                        text = [re.sub(r"""([;:,.!?¡¿—…"«»“”\(\)\{\}\[\]])""", r"-\1", text[0])]
                        text = [phone.replace('- ', '-').replace(' ', '-').split('-') for phone in text][0]
                    total_duration = 0
                    for text_token in text:
                        if text_token in phone_durations:
                            total_duration += phone_durations[text_token]
                    tokens = [[text_tokenizer[phone] for phone in text if phone in text_tokenizer]]
                # flow matching sampling
                if cfg['training']['content_factorization']['type'] == 'fbank':
                    target_duration = int(total_duration * 100 * 1.2)
                else:
                    target_duration = int(total_duration * 50 * 1.2)
                with torch.no_grad():
                    out_feats = model.sample(
                        tokens=tokens,
                        prompt_tokens=prompt_tokens,
                        prompt_features=prompt_features,
                        prompt_features_lens=torch.tensor([prompt_features.shape[1]], device=device),
                        duration='real',
                        features_lens=torch.tensor([target_duration], device=device),
                        num_step=int(args.sampling_steps)
                    )[0]
                # vocode
                with torch.no_grad():
                    if cfg['training']['normalize_input']:
                        out_feats = invert_normalized_input(out_feats)
                    out_feats = out_feats / cfg['training']['feature_scale']
                    if cfg['training']['content_factorization']['type'] == 'content':
                        for out_speaker in speakers['maps'][in_speaker]:
                            transform_tgt = transforms_target[out_speaker]
                            out_feats_spk = torch.matmul(out_feats, transform_tgt)
                            audio = linearvc_model.hifigan(out_feats_spk).detach().cpu().squeeze(0)
                            torchaudio.save(str(out_dir / in_speaker / (spk_wav.stem + '_' + out_speaker + '.wav')), audio, 16000)
                    else:
                        # vocode and voice-convert
                        audio_root = linearvc_model.hifigan(out_feats).squeeze(0)
                        input_features = knn_vc.get_features(audio_root)
                        for out_speaker in speakers['maps'][in_speaker]:
                            wav_hat = knn_vc.match(input_features, matching_sets[out_speaker], topk=4).unsqueeze(0)
                            torchaudio.save(str(out_dir / in_speaker / (spk_wav.stem + '_' + out_speaker + '.wav')), wav_hat, 16000)
    elif args.set == 'cv':
        inference_file_df = pd.read_csv(args.inference_file_dir, sep='|', header=0, index_col=None, quoting=3)
        for index, inference_item in tqdm(inference_file_df.iterrows(), total=inference_file_df.shape[0]):
            # process style prompt
            if 'prompt_audio' in inference_item:
                assert len(inference_item['prompt_transcript']) > 0
                wav, sr = torchaudio.load(str(inference_item['prompt_audio']), backend='sox')
                # wav, sr = torchaudio.load(str(inference_item['prompt_audio']), backend='soundfile')
                if sr != 16000:
                    if sr not in resamplers:
                        resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                    wav = resamplers[sr](wav)
                    sr = 16000
                if args.vad:
                    diarization = pipeline(
                        {
                            'waveform': wav,
                            'sample_rate': sr,
                        },
                        num_speakers = 1
                    )
                    new_wav = torch.zeros(wav.shape)
                    wav_start = -1
                    wav_end = -1
                    for segment, _, _ in diarization.itertracks(yield_label=True):
                        start_frame = int(segment.start * sr)
                        if wav_start == -1:
                            wav_start = start_frame
                        end_frame = int(segment.end * sr)
                        new_wav[:, start_frame:end_frame] = wav[:, start_frame:end_frame]
                    new_wav = new_wav[:, wav_start:wav_end]
                    wav = new_wav
                wavs = wav.to(device)
                prompt_features, _ = linearvc_model.wavlm.extract_features(wavs, output_layer=6)
                if transform is not None:
                    if cfg['training']['content_factorization']['type'] == 'content':
                        prompt_features = torch.matmul(prompt_features, transform)
                    elif cfg['training']['content_factorization']['type'] == 'speaker':
                        prompt_features = match_knn(prompt_features, transform)
                prompt_features = prompt_features * cfg['training']['feature_scale']
                if cfg['training']['normalize_input']:
                    prompt_features = normalize_input(prompt_features)
                prompt_features = prompt_features.detach()
            else:
                prompt_features = torch.tensor([[[]]], device=device)
                prompt_tokens = [[]]

            # tokenize text
            if cfg['data']['text_tokenizer']['type'] == 'spm':
                tokens = [sp.encode_as_ids(inference_item['text'])]
                if inference_item['prompt_audio'] is not None:
                    prompt_tokens = [sp.encode_as_ids(inference_item['prompt_transcript'])]
            elif cfg['data']['text_tokenizer']['type'] == 'phone':
                if cfg['data']['text_tokenizer']['pre_phonemized']:
                    text = g2p(inference_item['text'])
                    if inference_item['prompt_audio'] is not None:
                        prompt_text = g2p(inference_item['prompt_transcript'])
                else:
                    text = phonemize([inference_item['text']], separator=separator, strip=True)
                    text = [re.sub(r"""([;:,.!?¡¿—…"«»“”\(\)\{\}\[\]])""", r"-\1", text[0])]
                    text = [phone.replace('- ', '-').replace(' ', '-').split('-') for phone in text][0]
                    if inference_item['prompt_audio'] is not None:
                        prompt_text = phonemize([inference_item['prompt_transcript']], separator=separator, strip=True)
                        prompt_text = [re.sub(r"""([;:,.!?¡¿—…"«»“”\(\)\{\}\[\]])""", r"-\1", prompt_text[0])]
                        prompt_text = [phone.replace('- ', '-').replace(' ', '-').split('-') for phone in prompt_text][0]
                tokens = [[text_tokenizer[phone] for phone in text if phone in text_tokenizer]]
                if inference_item['prompt_audio'] is not None:
                    prompt_tokens = [[text_tokenizer[phone] for phone in prompt_text if phone in text_tokenizer]]

            # flow matching sampling
            if 'feature_lengths' in inference_item and inference_item['feature_lengths'] > 0:
                duration = 'real'
                feature_lengths = inference_item['feature_lengths']
            else:
                duration = 'predict'
                feature_lengths = -1
            with torch.no_grad():
                out_feats = model.sample(
                    tokens=tokens,
                    prompt_tokens=prompt_tokens,
                    prompt_features=prompt_features,
                    prompt_features_lens=torch.tensor([prompt_features.shape[1]], device=device),
                    duration=duration,
                    features_lens=torch.tensor([feature_lengths], device=device),
                    num_step=int(args.sampling_steps)
                )[0]

            # vocode
            with torch.no_grad():
                if cfg['training']['normalize_input']:
                    out_feats = invert_normalized_input(out_feats)
                out_feats = out_feats / cfg['training']['feature_scale']
                if cfg['training']['content_factorization']['type'] == 'content':
                    if 'target_speaker' in inference_item and inference_item['target_speaker'] in transforms:
                        transform_tgt = transforms[inference_item['target_speaker']]
                        out_feats = torch.matmul(out_feats, transform_tgt)
                    elif 'target_speaker_audio' in inference_item:
                        with torch.no_grad():
                            transform_wav, sr = torchaudio.load(inference_item['target_speaker_audio'])
                            if sr != 16000:
                                if sr not in resamplers:
                                    resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                                transform_wav = resamplers[sr](transform_wav)
                                sr = 16000
                            transform_features, _ = linearvc_model.wavlm.extract_features(transform_wav.to(device), output_layer=6) # [1, t, 1024]
                            if 'target_speaker_num_frames' in inference_item and inference_item['target_speaker_num_frames'] > 0:
                                frames = torch.randperm(transform_features.shape[1])[:args.target_speaker_num_frames]
                                transform_features = transform_features[:, frames, :]
                            transform_content = torch.matmul(transform_features, transform).squeeze(0).cpu().detach().numpy() # [t, r]
                            transform_tgt = np.matmul(np.linalg.pinv(transform_content), transform_features.squeeze(1).cpu().detach().numpy()) # [r, 1024]
                            transform_tgt = torch.tensor(transform_tgt).to(device)
                            out_feats = torch.matmul(out_feats, transform_tgt)
                    elif cfg['training']['content_factorization']['type'] == 'speaker':
                        out_feats = match_knn(out_feats, transform_tgt)
                audio = linearvc_model.hifigan(out_feats)

            audio = audio.squeeze().cpu()

            # save
            torchaudio.save(str(out_dir / 'out_wavs' / inference_item['out_wav_name']), audio.unsqueeze(0), 16000)
            shutil.copy2(inference_item['prompt_audio'], out_dir / 'prompt_wavs' / inference_item['out_wav_name'])



if __name__ == "__main__":
    main()

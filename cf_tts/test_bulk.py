import argparse
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inference_file_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sampling_steps", default=16, type=int)
    parser.add_argument("--vad", default=True, type=bool)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    (out_dir / 'out_wavs').mkdir(parents=True, exist_ok=True)
    (out_dir / 'prompt_wavs').mkdir(parents=True, exist_ok=True)

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
    hifigan, _ = torch.hub.load(
        "bshall/knn-vc",
        "hifigan_wavlm",
        trust_repo=True,
        prematched=True,
        device=device
    )
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
        import json
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

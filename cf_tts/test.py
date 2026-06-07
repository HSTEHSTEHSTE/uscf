import argparse
from pathlib import Path
import re
import yaml
import numpy as np
import torch
import torchaudio

from linearvc import linearvc
from linearvc.cf_tts.models.tts import ZipVoice

from linearvc.cf_tts.utils.common import normalize_input, invert_normalized_input, load_config, get_speaker_feats, match_knn
from linearvc.cf_tts.utils.checkpoints import load_checkpoint


# -------------------------
# main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sampling_steps", default=16, type=int)
    parser.add_argument("--target_speaker_audio", default=None)
    parser.add_argument("--target_speaker_audio_root", default=None)
    parser.add_argument("--target_speaker_num_frames", default=-1, type=int)
    parser.add_argument("--target_speaker", default='1272')
    parser.add_argument("--prompt_audio", type=Path, default=None)
    parser.add_argument("--prompt_transcript", type=str, default='')
    parser.add_argument("--feature_lengths", type=int, default=-1)
    parser.add_argument("--vad", type=bool, default=False)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        import os, json
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
    linearvc_model = linearvc.LinearVC(wavlm, hifigan, device)

    # -------------------------
    # load transform
    # -------------------------

    resamplers = {}
    if cfg['training']['content_factorization']['type'] == 'content':
        if cfg["training"]['content_factorization']["content_factorization_file"] is not None:
            transforms = np.load(cfg["training"]['content_factorization']["content_factorization_file"], allow_pickle=True).item()
            transform = torch.tensor(np.linalg.pinv(transforms[list(transforms.keys())[0]])).to(device)
            if args.target_speaker_audio is None:
                transform_tgt = torch.tensor(transforms[args.target_speaker]).to(device)
            else:
                with torch.no_grad():
                    transform_wav, sr = torchaudio.load(args.target_speaker_audio)
                    if sr != 16000:
                        if sr not in resamplers:
                            resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                        transform_wav = resamplers[sr](transform_wav)
                        sr = 16000
                    transform_features, _ = linearvc_model.wavlm.extract_features(transform_wav.to(device), output_layer=6) # [1, t, 1024]
                    if args.target_speaker_num_frames > 0:
                        frames = torch.randperm(transform_features.shape[1])[:args.target_speaker_num_frames]
                        transform_features = transform_features[:, frames, :]
                    transform_content = torch.matmul(transform_features, transform).squeeze(0).cpu().detach().numpy() # [t, r]
                    transform_tgt = np.matmul(np.linalg.pinv(transform_content), transform_features.squeeze(1).cpu().detach().numpy()) # [r, 1024]
                    transform_tgt = torch.tensor(transform_tgt).to(device)
        else:
            transform = None
            transform_tgt = None
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
        if args.target_speaker_audio_root is not None:
            feats_tgt = get_speaker_feats(
                tgt_speaker_root=cfg['training']['content_factorization']['factorization_speaker'],
                linearvc_model=linearvc_model
            )
            index_tgt = brute_force.build(feats_tgt)
            transform_tgt = {
                'feats': feats_tgt,
                'index': index_tgt,
                'brute_force': brute_force
            }
        else:
            transform_tgt = None
    elif cfg['training']['content_factorization']['type'] == 'none':
        transform = None
        transform_tgt = None
    elif cfg['training']['content_factorization']['type'] == 'fbank':
        from speechbrain.lobes.features import Fbank
        transform = Fbank(sample_rate=16000, n_mels=cfg['model']['tts']['zipvoice']['feat_dim'])
        transform_tgt = transform

    # -------------------------
    # load and process prompt
    # -------------------------
    if args.prompt_audio is not None:
        assert len(args.prompt_transcript) > 0
        wav, sr = torchaudio.load(str(args.prompt_audio), backend='sox')
        if sr != 16000:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resamplers[sr](wav)
            sr = 16000
        if args.vad:
            effects = [['reverse']]
            wav = torchaudio.functional.vad(wav, sr)
            wav, sr = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)
            wav = torchaudio.functional.vad(wav, sr)
            wav, sr = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)
        wavs = wav.to(device)
        if cfg['training']['content_factorization']['type'] == 'fbank':
            prompt_features = transform(wavs)
        else:
            prompt_features, _ = linearvc_model.wavlm.extract_features(wavs, output_layer=6)
            if transform is not None:
                if cfg['training']['content_factorization']['type'] == 'content':
                    prompt_features = torch.matmul(prompt_features, transform)
                elif cfg['training']['content_factorization']['type'] == 'speaker':
                    prompt_features = match_knn(prompt_features, transform)
                elif cfg['training']['content_factorization']['type'] == 'none':
                    pass
        prompt_features = prompt_features * cfg['training']['feature_scale']
        if cfg['training']['normalize_input']:
            prompt_features = normalize_input(prompt_features)
        prompt_features = prompt_features.detach()
    else:
        prompt_features = torch.tensor([[[]]], device=device)
        prompt_tokens = [[]]

    # -------------------------
    # tokenize text
    # -------------------------

    if cfg['data']['text_tokenizer']['type'] == 'spm':
        import sentencepiece as spm
        print("Loading SentencePiece...")
        sp = spm.SentencePieceProcessor()
        sp.load(cfg['data']['text_tokenizer']['tokenizer_file'])
        tokens = [sp.encode_as_ids(args.text)]
        if args.prompt_audio is not None:
            prompt_tokens = [sp.encode_as_ids(args.prompt_transcript)]
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
            text = g2p(args.text)
            if args.prompt_audio is not None:
                prompt_text = g2p(args.prompt_transcript)
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
            text = phonemize([args.text], separator=separator, strip=True)
            text = [re.sub(r"""([;:,.!?¡¿—…"«»“”\(\)\{\}\[\]])""", r"-\1", text[0])]
            text = [phone.replace('- ', '-').replace(' ', '-').split('-') for phone in text][0]
            if args.prompt_audio is not None:
                prompt_text = phonemize([args.prompt_transcript], separator=separator, strip=True)
                prompt_text = [phone.replace('- ', '-').replace(' ', '-').split('-') for phone in prompt_text][0]

        tokens = [[text_tokenizer[phone] for phone in text if phone in text_tokenizer]]
        if args.prompt_audio is not None:
            prompt_tokens = [[text_tokenizer[phone] for phone in prompt_text if phone in text_tokenizer]]

    # -------------------------
    # flow matching sampling
    # -------------------------

    print("Running sampling...")
    if args.feature_lengths > 0:
        duration = 'real'
    else:
        duration = 'predict'
    with torch.no_grad():
        out_feats = model.sample(
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            prompt_features_lens=torch.tensor([prompt_features.shape[1]], device=device),
            duration=duration,
            features_lens=torch.tensor([args.feature_lengths], device=device),
            num_step=int(args.sampling_steps)
        )[0]

    # -------------------------
    # vocode
    # -------------------------

    print("Running HiFiGAN...")
    with torch.no_grad():
        if cfg['training']['normalize_input']:
            out_feats = invert_normalized_input(out_feats)
        out_feats = out_feats / cfg['training']['feature_scale']
        if transform_tgt is not None:
            if cfg['training']['content_factorization']['type'] == 'content':
                out_feats = torch.matmul(out_feats, transform_tgt)
            elif cfg['training']['content_factorization']['type'] == 'speaker':
                out_feats = match_knn(out_feats, transform_tgt)
            elif cfg['training']['content_factorization']['type'] == 'none':
                pass
        audio = linearvc_model.hifigan(out_feats)

    audio = audio.squeeze().cpu()

    # -------------------------
    # save
    # -------------------------

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.out, audio.unsqueeze(0), 16000)

    print("Saved:", args.out)


if __name__ == "__main__":
    main()

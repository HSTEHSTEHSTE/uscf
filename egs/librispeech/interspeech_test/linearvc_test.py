import argparse, json, random
import torch, torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from linearvc.linearvc import LinearVC

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech_root",
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
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main(args):
    print("Librispeech root: ", args.librispeech_root)
    print("Out dir: ", args.out_dir)
    print("Num utt per speaker: ", args.num_utt_per_speaker)
    librispeech_root = Path(args.librispeech_root)
    out_dir = Path(args.out_dir)

    with open('linearvc/egs/librispeech/libri_test/speakers.json', 'r') as file:
        speakers = json.load(file)
    
    in_speakers = speakers['lists']['test-clean_source']
    out_speakers = speakers['lists']['test-other_target']

    # Load all the required models
    wavlm = torch.hub.load(
        "bshall/knn-vc", 
        "wavlm_large", 
        trust_repo=True, 
        progress=True, 
        device=device, 
    )
    hifigan, _ = torch.hub.load(
        "bshall/knn-vc",
        "hifigan_wavlm",
        trust_repo=True,
        prematched=True,
        progress=True,
        device=device,
    )
    linearvc_model = LinearVC(wavlm, hifigan, device)

    for in_speaker in tqdm(in_speakers):
        (out_dir / in_speaker).mkdir(parents=True, exist_ok=True)
        spk_wavs = list((librispeech_root / 'test-clean' / in_speaker).rglob('*.flac'))
        for out_speaker in speakers['maps'][in_speaker]:
            out_spk_wavs = list((librispeech_root / 'test-other' / out_speaker).rglob('*.flac'))
            # Voice conversion projection matrix
            W = linearvc_model.get_projmat(
                spk_wavs,
                out_spk_wavs,
                parallel=False,  # enable if parallel
                vad=False,
            )
            for spk_wav in spk_wavs[:args.num_utt_per_speaker]:
                input_features = linearvc_model.get_features(str(spk_wav))
                wav_hat = linearvc_model.project_and_vocode(input_features, W)
                torchaudio.save(str(out_dir / in_speaker / (spk_wav.stem + '_' + out_speaker + '.wav')), wav_hat[None], 16000)


if __name__ == "__main__":
    args = check_argv()
    main(args)
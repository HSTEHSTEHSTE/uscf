import argparse, json, random, os
import torch, torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech_root",
        type=Path,
        default=""
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
    parser.add_argument(
        "--set",
        type=str,
        help="librispeech, cv",
        default="cv"
    )
    parser.add_argument(
        "--cv_inference_file",
        type=Path,
        default="linearvc/exp/tts/inference/inference_0.tsv"
    )
    return parser.parse_args()


def main(args):
    print("Set: ", args.set)
    print("Librispeech root: ", args.librispeech_root)
    print("Out dir: ", args.out_dir)
    print("Num utt per speaker: ", args.num_utt_per_speaker)
    librispeech_root = Path(args.librispeech_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open('linearvc/egs/librispeech/libri_test/speakers.json', 'r') as file:
        speakers = json.load(file)
    
    in_speakers = speakers['lists']['test-clean_source']
    out_speakers = speakers['lists']['test-other_target']

    if args.set == "librispeech":
        for in_speaker in tqdm(in_speakers):
            (out_dir / in_speaker).mkdir(parents=True, exist_ok=True)
            spk_wavs = list((librispeech_root / 'test-clean' / in_speaker).rglob('*.flac'))[:args.num_utt_per_speaker]
            for spk_wav in spk_wavs:
                input_features = knn_vc.get_features(str(spk_wav))
                for out_speaker in speakers['maps'][in_speaker]:
                    wav_hat = knn_vc.match(input_features, matching_sets[out_speaker], topk=4).unsqueeze(0)
                    torchaudio.save(str(out_dir / in_speaker / (spk_wav.stem + '_' + out_speaker + '.wav')), wav_hat, 16000)
    elif args.set == 'cv':
        cv_inference_file_dir = Path(args.cv_inference_file)
        print("CV inference file: ", cv_inference_file_dir)
        cv_inference_file = pd.read_csv(cv_inference_file_dir, sep='|', header=0, index_col=None, quoting=3)
        for index, entry in tqdm(cv_inference_file.iterrows(), total=cv_inference_file.shape[0]):
            os.symlink(entry["prompt_audio"], str(out_dir / (entry["out_wav_name"][:-4] + '.mp3')))

if __name__ == "__main__":
    args = check_argv()
    main(args)
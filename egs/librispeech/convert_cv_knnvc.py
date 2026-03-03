import argparse, random
import torch, torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv_dir",
        type=Path,
        help="commonvoice directory",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="output speech directory",
    )
    parser.add_argument(
        "--data_file",
        type=Path,
        help="commonvoice data file",
    )
    parser.add_argument(
        "--spk_dir",
        type=Path,
        help="Target speaker root dir"
    )
    return parser.parse_args()


def main(args):
    print("CV dir: ", args.cv_dir)
    print("Out dir: ", args.out_dir)
    print("Data file: ", args.data_file)
    print("Speaker dir: ", args.spk_dir)
    cv_dir = Path(args.cv_dir)
    out_dir = Path(args.out_dir)
    data_file = Path(args.data_file)
    spk_dir = Path(args.spk_dir)

    spk_wavs = spk_dir.rglob('*.flac')
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    matching_set = knn_vc.get_matching_set(spk_wavs)

    input_wavs = pd.read_csv(args.data_file, sep=',', header=0, index_col=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in tqdm(input_wavs.iterrows(), total=input_wavs.shape[0]):
        out_path = out_dir / (entry[1].path[:-4] + '.wav')
        if not out_path.is_file():
            input_features = knn_vc.get_features(str(cv_dir / 'clips' / entry[1].path))
            wav_hat = knn_vc.match(input_features, matching_set, topk=4).unsqueeze(0)
            torchaudio.save(str(out_dir / (entry[1].path[:-4] + '.wav')), wav_hat, 16000)


if __name__ == "__main__":
    args = check_argv()
    main(args)
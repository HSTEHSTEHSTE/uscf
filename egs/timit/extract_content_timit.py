import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timit_wavlm_feat_path",
        type=Path,
        help="TIMIT wavlm feature directory ending in e.g. `dev-clean/`",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="output will be written to a subdirectory",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default='TEST'
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=75
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        help="exclude utterances with filenames in this file",
    )
    parser.add_argument(
        "--transform_path",
        type=Path,
        help="path to transformation matrix (ST, UTXSS, etc.)",
    )
    return parser.parse_args()

def main(args):
    subset = args.subset
    rank = args.rank
    timit_wavlm_feat_path = Path(args.timit_wavlm_feat_path) / subset
    output_dir = Path(args.output_dir) / str(rank) / subset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Subset: ", subset)
    print("Rank: ", rank)
    print("WavLM feat path: ", timit_wavlm_feat_path)
    print("Transform path: ", args.transform_path)
    print("Output dir: ", output_dir)
    transform = torch.tensor(np.load(args.transform_path)).to(device).float()
    speakers = sorted([p.stem for p in (timit_wavlm_feat_path / 'spks').iterdir()])
    timit_wavlm_feats = list((timit_wavlm_feat_path / 'utts').rglob('*.npy'))

    for timit_wavlm_feat in tqdm(timit_wavlm_feats):
        feat = torch.tensor(np.load(timit_wavlm_feat)).to(device).float()
        speaker = str(timit_wavlm_feat).split('/')[-2]
        content = torch.matmul(feat, transform)
        output_fn = (output_dir / (timit_wavlm_feat.parent.relative_to((timit_wavlm_feat_path / 'utts'))) / timit_wavlm_feat.stem).with_suffix(".npy")
        output_fn.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_fn, content.detach().cpu().numpy())

if __name__ == "__main__":
    args = check_argv()
    main(args)
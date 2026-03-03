from numpy import linalg
from pathlib import Path
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
import celer
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import time
import torch
import torchaudio

from linearvc.utils import fast_cosine_dist

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        type=str,
        default='TRAIN'
    )
    parser.add_argument(
        "--wav_dir",
        type=Path,
    )
    parser.add_argument(
        "--feats_dir",
        type=Path,
        help="source speech directory",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=100
    )
    parser.add_argument(
        "--num_index",
        type=int,
        default=1
    )
    parser.add_argument(
        "--out_path_root",
        type=Path,
    )
    return parser.parse_args()

def align(src, refs):
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine")
    neighbors.fit(refs)
    dists, indices = neighbors.kneighbors(src)
    return refs[indices.squeeze(), :]

def main(args):
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)

    subset = args.subset
    rank = 100
    wav_dir = Path(args.wav_dir)
    wav_dir = wav_dir / subset
    n_frames = 8192
    k_top = 1

    feats_dir = Path(args.feats_dir)
    feats_dir = feats_dir / (subset + "_vc") / "spks"
    feats_dict = {}
    print("Reading from:", feats_dir)
    for speaker_feats_fn in tqdm(sorted(feats_dir.glob("*.npy"))):
        speaker = speaker_feats_fn.stem
        feats_dict[speaker] = np.load(speaker_feats_fn, allow_pickle=True)
    print("No. speakers:", len(feats_dict))

    out_path_root = Path(args.out_path_root)

    XS = []
    speakers = sorted(feats_dict)
    for speaker in speakers:
        XS.append(feats_dict[speaker][:, :])



    print("Matching:")
    new_XS = []
    for spk_id in tqdm(list(range(len(XS)))):
        content_spk = [align(XS[0], XS[spk_id])]
        # content_spk = [align(X_other, XS[spk_id]) for X_other in XS]
        content_spk = np.concatenate(content_spk, axis=0)
        new_XS.append(content_spk)

    XS = new_XS
    XS = np.concatenate(XS, axis=-1)
    XS = np.float32(XS)

    start_time = time.time()
    print("SVD")
    U, S, VT = randomized_svd(XS, n_components=rank)
    print("Time expired: ", time.time() - start_time)

    print("Reshaping")
    VT = VT.reshape(-1, len(speakers), 1024).swapaxes(0, 1)
    transforms = {
        f"{speaker}": VT[i, :, :] for i, speaker in enumerate(speakers)
    }

    out_path = out_path_root / ('TIMIT_' + subset + '_vc/spk_0_r' + str(rank))
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / 'XS.npy', XS)
    np.save(out_path / 'U.npy', U)
    np.save(out_path / 'S.npy', S)
    np.save(out_path / 'VT.npy', VT)
    np.save(out_path / 'transforms.npy', transforms)

if __name__ == "__main__":
    args = check_argv()
    main(args)
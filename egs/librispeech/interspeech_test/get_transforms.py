from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
import argparse
import json
import numpy as np
import torch

from linearvc.utils import fast_cosine_dist

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_frames",
        type=int,
        default=8192
    )
    parser.add_argument(
        "--k_top",
        type=int,
        default=4
    )
    parser.add_argument(
        "--feats_dir",
        type=Path,
        help="source speech directory",
    )
    parser.add_argument(
        "--out_root_dir",
        type=Path,
    )
    parser.add_argument(
        "--set_num",
        type=int,
    )
    return parser.parse_args()

def align(src, refs):
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine")
    neighbors.fit(refs)
    dists, indices = neighbors.kneighbors(src)
    return refs[indices.squeeze(), :]

def main(args):
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)

    n_frames = args.n_frames
    k_top = args.k_top
    out_root_dir = Path(args.out_root_dir)

    with open('linearvc/egs/librispeech/libri_test/speakers.json', 'r') as file:
        speakers = json.load(file)

    sources = []
    sources.append(['test-clean_source', 'test-other_target'])
    sources.append(['dev-clean_heldout', 'test-other_target'])
    sources.append(['test-clean_source', 'dev-clean_heldout'])
    sources.append(['dev-clean_heldout', 'test-clean_heldout'])

    # ranks = [10, 20, 30, 50, 75, 100]
    ranks = [256, 512, 1024]

    source_speech = np.load('linearvc/exp/wavlm_feats/librispeech/dev-clean/1272.npy')
    print("Number of frames: ", source_speech.shape[0])

    index = args.set_num
    source = sources[index]

    feats_dir = Path(args.feats_dir)
    feats_dict = {}
    for source_set in source:
        source_set_name = source_set.split('_')[0]
        for spk in speakers['lists'][source_set]:
            feats_dict[spk] = np.load(feats_dir / source_set_name / (spk + '.npy'))

    for rank in tqdm(ranks):
        XS_list = []
        speakers = sorted(feats_dict)
        for speaker in speakers:
            XS_list.append(feats_dict[speaker][:, :])

        XS = [align(source_speech, X) for X in tqdm(XS_list)]
        XS = np.concatenate(XS, axis=-1)
        XS = np.float32(XS)

        U, S, VT = randomized_svd(XS, n_components=rank)

        VT = VT.reshape(-1, len(speakers), 1024).swapaxes(0, 1)
        transforms = {
            f"{speaker}": VT[i, :, :] for i, speaker in enumerate(speakers)
        }

        projmats = {}
        for source in tqdm(feats_dict, leave=False):
            for target in feats_dict:
                if source == target:
                    continue
                W = np.linalg.pinv(transforms[source]) @ transforms[target]
                projmats[f"{source}-{target}"] = (W, None)

        out_path = Path(out_root_dir / 'transforms' / str(index) / ('rank_' + str(rank)))
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(out_path / 'XS.npy', XS)
        np.save(out_path / 'U.npy', U)
        np.save(out_path / 'S.npy', S)
        np.save(out_path / 'VT.npy', VT)
        np.save(out_path / 'transforms.npy', transforms)

if __name__ == "__main__":
    args = check_argv()
    main(args)
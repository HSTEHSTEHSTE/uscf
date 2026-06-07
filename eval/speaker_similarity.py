import argparse
import random
import numpy as np
from scipy.spatial import distance
from pathlib import Path

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_emb_dir",
        type=Path,
        help="reference Resemblyzer embedding directory",
    )
    parser.add_argument(
        "--out_emb_dir",
        type=Path,
        help="output Resemblyzer embedding directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
    )
    return parser.parse_args()

def get_spk_mapping(spks, seed):
    random.seed(seed)
    if len(spks) < 2:
        raise ValueError("Not enough speakers")

    while True:
        shuffled = spks[:]
        random.shuffle(shuffled)

        # check no one maps to themselves
        if all(a != b for a, b in zip(spks, shuffled)):
            return dict(zip(spks, shuffled))

def main(args):
    ref_emb_dir = Path(args.ref_emb_dir)
    out_emb_dir = Path(args.out_emb_dir)
    spks_long = out_emb_dir.rglob('*.npy')
    spks = []
    for spk_long in spks_long:
        spks.append(spk_long.stem)
    spk_map = get_spk_mapping(spks, args.seed)
    sims = []
    for ref_spk in spks:
        out_spk = spk_map[ref_spk]
        ref_emb = np.load(ref_emb_dir / (ref_spk + '.npy'))
        out_emb = np.load(out_emb_dir / (out_spk + '.npy'))
        sims.append(distance.cosine(ref_emb, out_emb))
    sim = sum(sims) / len(sims)
    print("Sim: ", sim)    

if __name__ == "__main__":
    args = check_argv()
    main(args)
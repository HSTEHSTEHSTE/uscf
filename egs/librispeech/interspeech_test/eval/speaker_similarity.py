import argparse
from tqdm import tqdm
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
    parser.add_argument(
        "--anchor_spk",
        type=str,
        default='none',
    )
    return parser.parse_args()

def main(args):
    ref_emb_dir = Path(args.ref_emb_dir)
    out_emb_dir = Path(args.out_emb_dir)
    print('ref emb dir: ', ref_emb_dir)
    print('out emb dir: ', out_emb_dir)
    print('anchor_spk: ', args.anchor_spk)
    ref_embs = list(ref_emb_dir.rglob('*.npy'))
    spk_refs = {}
    for ref_emb in ref_embs:
        spk_refs[ref_emb.stem] = np.load(ref_emb)
    out_embs = list(out_emb_dir.rglob('*.npy'))
    sims = []
    for out_emb_dir in tqdm(out_embs):
        if args.anchor_spk == 'none' or out_emb_dir.parts[-2] == args.anchor_spk:
            out_emb = np.load(out_emb_dir)
            target_spk = out_emb_dir.stem.split('_')[-1]
            ref_emb = spk_refs[target_spk]
            sims.append(1 - distance.cosine(ref_emb, out_emb))
    sim = sum(sims) / len(sims)
    print("Sim: ", sim)    

if __name__ == "__main__":
    args = check_argv()
    main(args)
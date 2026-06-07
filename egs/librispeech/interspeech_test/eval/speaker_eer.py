import argparse
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import distance
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from pathlib import Path

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_emb_dir",
        type=Path,
        help="reference ecapa-tdnn embedding directory",
    )
    parser.add_argument(
        "--ref_utt_emb_dir",
        type=Path,
        help="reference per-utterance ecapa-tdnn embedding directory",
    )
    parser.add_argument(
        "--out_emb_dir",
        type=Path,
        help="output ecapa-tdnn embedding directory",
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
    parser.add_argument(
        "--set",
        type=str,
        help="librispeech, cv",
        default="librispeech"
    )
    return parser.parse_args()

def eer(y, y_score):
    fpr, tpr, _ = roc_curve(y, 1 - y_score, pos_label=1)
    return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

def main(args):
    ref_emb_dir = Path(args.ref_emb_dir)
    ref_utt_emb_dir = Path(args.ref_utt_emb_dir)
    out_emb_dir = Path(args.out_emb_dir)
    print('ref emb dir: ', ref_emb_dir)
    print('ref utt emb dir: ', ref_utt_emb_dir)
    print('out emb dir: ', out_emb_dir)
    print('anchor_spk: ', args.anchor_spk)
    ref_embs = list((ref_emb_dir / 'test-other').rglob('*.npy'))
    spk_refs = {}
    for ref_emb in ref_embs:
        spk_refs[ref_emb.stem] = np.load(ref_emb)
    spk_scores = {}
    for spk_ref in spk_refs:
        ref_emb = spk_refs[spk_ref]
        spk_scores[spk_ref] = {
            'labels': [],
            'scores': [],
            'target_labels': [],
            'target_scores': [],
        }
        ref_utt_embs = [np.load(ref_utt_emb) for ref_utt_emb in (ref_utt_emb_dir / spk_ref).rglob('*.npy')]
        for ref_utt_emb in ref_utt_embs:
            spk_scores[spk_ref]['target_labels'].append(1)
            spk_scores[spk_ref]['target_scores'].append(distance.cosine(ref_emb, ref_utt_emb))
    
    src_ref_embs = list((ref_emb_dir / 'test-clean').rglob('*.npy'))
    src_spk_refs = {}
    for ref_emb in src_ref_embs:
        src_spk_refs[ref_emb.stem] = np.load(ref_emb)
    out_embs = list(out_emb_dir.rglob('*.npy'))
    sims = []
    desims = []
    target_spks = []
    for out_emb_dir in tqdm(out_embs):
        if args.set == 'librispeech':
            if args.anchor_spk == 'none' or out_emb_dir.parts[-2] == args.anchor_spk:
                target_spk = out_emb_dir.stem.split('_')[-1]
                if target_spk not in target_spks:
                    target_spks.append(target_spk)
                out_emb = np.load(out_emb_dir)
                if args.anchor_spk == 'none':
                    src_spk = out_emb_dir.parts[-2]
                else:
                    src_spk = out_emb_dir.parts[-3]
                src_ref_emb = src_spk_refs[src_spk]
                desims.append(1 - distance.cosine(src_ref_emb, out_emb))
                for spk_ref in spk_refs:
                    ref_emb = spk_refs[spk_ref]
                    spk_scores[target_spk]['labels'].append(int(spk_ref == target_spk))
                    sim_score = distance.cosine(ref_emb, out_emb)
                    spk_scores[target_spk]['scores'].append(sim_score)
                    if spk_ref == target_spk:
                        sims.append(1 - sim_score)
                        spk_scores[target_spk]['target_labels'].append(0)
                        spk_scores[target_spk]['target_scores'].append(sim_score)
        elif args.set == 'cv':
            if args.anchor_spk == 'none' or out_emb_dir.parts[-2] == args.anchor_spk:
                if args.anchor_spk == 'none':
                    target_spk = out_emb_dir.parts[-2]
                else:
                    target_spk = out_emb_dir.parts[-3]
                if target_spk not in target_spks:
                    target_spks.append(target_spk)
                out_emb = np.load(out_emb_dir)
                for spk_ref in spk_refs:
                    ref_emb = spk_refs[spk_ref]
                    spk_scores[target_spk]['labels'].append(int(spk_ref == target_spk))
                    sim_score = distance.cosine(ref_emb, out_emb)
                    spk_scores[target_spk]['scores'].append(sim_score)
                    if spk_ref == target_spk:
                        sims.append(1 - sim_score)
                        spk_scores[target_spk]['target_labels'].append(0)
                        spk_scores[target_spk]['target_scores'].append(sim_score)

    eers = []
    for ref_spk in target_spks:
        eers.append(eer(spk_scores[ref_spk]['labels'], np.array(spk_scores[ref_spk]['scores'])))

    target_eers = []
    for ref_spk in target_spks:
        target_eers.append(eer(spk_scores[ref_spk]['target_labels'], np.array(spk_scores[ref_spk]['target_scores'])))

    print("Speaker similarity: ", np.array(sims).mean())
    if len(desims) > 0:
        print("Source speaker similarity: ", np.array(desims).mean())
    print("EER: ", np.array(eers).mean())
    print("EER STD: ", np.std(np.array(eers)))
    print("Target EER: ", np.array(target_eers).mean())
    print("Target EER STD: ", np.std(np.array(target_eers)))

if __name__ == "__main__":
    args = check_argv()
    main(args)
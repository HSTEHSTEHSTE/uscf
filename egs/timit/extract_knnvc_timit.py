import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from cuvs.neighbors import brute_force
from linearvc import linearvc
from linearvc.cf_tts.utils.common import get_speaker_feats, match_knn

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
        "--exclude",
        type=Path,
        help="exclude utterances with filenames in this file",
    )
    parser.add_argument(
        "--tgt_spk_path",
        type=Path,
        help="path to transformation matrix (ST, UTXSS, etc.)",
    )
    return parser.parse_args()

def main(args):
    subset = args.subset
    timit_wavlm_feat_path = Path(args.timit_wavlm_feat_path) / subset
    output_dir = Path(args.output_dir) / subset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Subset: ", subset)
    print("WavLM feat path: ", timit_wavlm_feat_path)
    print("Tgt spk path: ", args.tgt_spk_path)
    print("Output dir: ", output_dir)

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
    linearvc_model = linearvc.LinearVC(wavlm, hifigan, device)

    feats = get_speaker_feats(
        tgt_speaker_root=Path(args.tgt_spk_path),
        linearvc_model=linearvc_model,
        device=device
    )
    index = brute_force.build(feats)
    transform = {
        'feats': feats,
        'index': index,
        'brute_force': brute_force
    }

    speakers = sorted([p.stem for p in (timit_wavlm_feat_path / 'spks').iterdir()])
    timit_wavlm_feats = list((timit_wavlm_feat_path / 'utts').rglob('*.npy'))

    for timit_wavlm_feat in tqdm(timit_wavlm_feats):
        feat = torch.tensor(np.load(timit_wavlm_feat)).to(device).float()
        speaker = str(timit_wavlm_feat).split('/')[-2]
        content = match_knn(feat, transform).squeeze(1)
        output_fn = (output_dir / (timit_wavlm_feat.parent.relative_to((timit_wavlm_feat_path / 'utts'))) / timit_wavlm_feat.stem).with_suffix(".npy")
        output_fn.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_fn, content.detach().cpu().numpy())

if __name__ == "__main__":
    args = check_argv()
    main(args)
#!/usr/bin/env python

"""
Extract WavLM features for a TIMIT set.

Author: Henry Li Xinyuan
Date: 2025
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import sys
import torch
import torchaudio

from linearvc.utils import pca_transform

device = "cuda"


def check_argv():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "--timit_dir",
        type=Path,
        help="TIMIT directory ending in e.g. `TRAIN`, 'TEST'",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="output will be written to a subdirectory",
    )
    parser.add_argument(
        "--pca",
        type=Path,
        help="NumPy archive with PCA parameters (default: no PCA)",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        help="exclude utterances with filenames in this file",
    )
    parser.add_argument(
        "--save_mode",
        type=str,
        help="save mode: 'spks' or 'utts'",
        default='utts',
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        help='wavlm, contentvec',
        default='wavlm'
    )
    return parser.parse_args()


def main(args):
    if args.feature_type == 'wavlm':
        wavlm = torch.hub.load(
            "bshall/knn-vc", "wavlm_large", trust_repo=True, device=device
        )
    elif args.feature_type == 'contentvec':
        from transformers import HubertModel
        import torch.nn as nn
        class HubertModelWithFinalProj(HubertModel):
            def __init__(self, config):
                super().__init__(config)
                self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)
        model = HubertModelWithFinalProj.from_pretrained("lengyue233/content-vec-best").to(device)

    if args.pca is not None:
        print("Reading:", args.pca)
        pca = np.load(args.pca)
        temp = {}
        for key in pca:
            temp[key] = torch.from_numpy(pca[key]).float().to(device)
        pca = temp
    else:
        pca = None

    if args.exclude is not None:
        print("Reading:", args.exclude)
        exclude_utterances = set()
        with open(args.exclude) as f:
            for line in f:
                exclude_utterances.add(line.strip())
    else:
        exclude_utterances = []

    wav_dir = args.timit_dir
    output_dir = args.output_dir / args.save_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Writing to:", output_dir)
    wavs = sorted(wav_dir.rglob('*.WAV'))

    features = {}
    for wav_file in tqdm(wavs):
        if wav_file.stem in exclude_utterances:
            continue
        wav_elements = str(wav_file).split('/')
        speaker = wav_elements[-2]
        wav, sr = torchaudio.load(str(wav_file))
        wav = wav.to(device)
        if args.feature_type == 'wavlm':
            with torch.inference_mode():
                x, _ = wavlm.extract_features(wav, output_layer=6)
        elif args.feature_type == 'contentvec':
            with torch.inference_mode():
                x = model(wav)["last_hidden_state"]
        if pca is not None:
            x = pca_transform(
                x, pca["mean"], pca["components"], pca["explained_variance"]
            )
        x = x.cpu().numpy().squeeze()
        if args.save_mode == 'spks':
            if speaker not in features:
                features[speaker] = [x]
            else:
                features[speaker].append(x)
        elif args.save_mode == 'utts':
            output_fn = (output_dir / (wav_file.parent.relative_to(wav_dir)) / wav_file.stem).with_suffix(".npy")
            output_fn.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_fn, x)

    if args.save_mode == 'spks':
        for speaker in features:
            output_fn = (output_dir / speaker).with_suffix(".npy")
            features[speaker] = np.vstack(features[speaker], dtype=np.float16)
            np.save(output_fn, features[speaker])


if __name__ == "__main__":
    args = check_argv()
    main(args)

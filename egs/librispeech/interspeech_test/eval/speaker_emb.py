import argparse
import numpy as np
import torch, torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from pathlib import Path
from tqdm import tqdm

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--converted_dir",
        type=Path,
        help="converted speech directory",
    )
    parser.add_argument(
        "--out_embed_dir",
        type=Path,
        help="directory to store ECAPA-TDNN output",
    )
    parser.add_argument(
        "--anchor_spk",
        type=str,
        default='none',
    )
    return parser.parse_args()

def main(args):
    print("Converted dir: ", args.converted_dir)
    print("Out embed dir: ", args.out_embed_dir)
    print("Anchor spk: ", args.anchor_spk)
    converted_dir = Path(args.converted_dir)
    if args.anchor_spk == 'none':
        out_embed_dir = Path(args.out_embed_dir)
    else:
        out_embed_dir = Path(args.out_embed_dir) / args.anchor_spk
    out_embed_dir.mkdir(parents=True, exist_ok=True)

    extensions = ['wav', 'flac']
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    spks = list(converted_dir.iterdir())
    resamplers = {}
    for spk in tqdm(spks):
        wavs = []
        embeds = []
        for extension in extensions:
            wavs += (converted_dir / spk).rglob('*.' + extension)

        for wav_name in tqdm(wavs):
            (out_embed_dir / wav_name.parent.relative_to(converted_dir)).mkdir(parents=True, exist_ok=True)
            wav, sr = torchaudio.load(str(wav_name))
            if sr != 16000:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
                wav = resamplers[sr](wav)
                sr = 16000
            if args.anchor_spk == 'none' or wav_name.parts[-2] == args.anchor_spk:
                embed = classifier.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()
                np.save(out_embed_dir / wav_name.parent.relative_to(converted_dir) / (wav_name.stem + '.npy'), embed)
                
if __name__ == "__main__":
    args = check_argv()
    main(args)
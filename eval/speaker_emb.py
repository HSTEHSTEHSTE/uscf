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
    return parser.parse_args()

def main(args):
    converted_dir = Path(args.converted_dir)
    out_embed_dir = Path(args.out_embed_dir)
    out_embed_dir.mkdir(parents=True, exist_ok=True)

    extensions = ['wav', 'flac']
    spks_long = converted_dir.iterdir()
    spks = []
    for spk_long in spks_long:
        spks.append(str(spk_long).split('/')[-1])

    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    resamplers = {}

    for spk in tqdm(spks):
        wavs = []
        embeds = []
        for extension in extensions:
            wavs += (converted_dir / spk).rglob('*.' + extension)

        for wav in tqdm(wavs):
            wav, sr = torchaudio.load(str(wav))
            if sr != 16000:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
                wav = resamplers[sr](wav)
                sr = 16000
            embed = classifier.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()
            embeds.append(embed)

        embeds_avg = np.array(embeds).mean(axis=0)

        np.save(out_embed_dir / (spk + '.npy'), embeds_avg)


if __name__ == "__main__":
    args = check_argv()
    main(args)
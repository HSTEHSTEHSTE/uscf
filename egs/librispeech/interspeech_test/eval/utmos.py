import argparse
import utmosv2
from pathlib import Path
from tqdm import tqdm
import torch, torchaudio

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
        "--length_limit",
        type=int,
        default=-1,
        help="number of wavs to evaluate",
    )
    parser.add_argument(
        "--anchor_spk",
        type=str,
        default='none',
    )
    return parser.parse_args()

def main(args):
    print("Converted dir: ", args.converted_dir)
    out_wav_path = Path(args.converted_dir)

    extensions = ['wav', 'flac', 'mp3']
    out_wavs = []
    for extension in extensions:
        out_wavs += out_wav_path.rglob('*.' + extension)

    model = utmosv2.create_model(pretrained=True)
    model = model.to(device)
    scores = []

    for index, out_wav_path in tqdm(enumerate(out_wavs), total=len(out_wavs)):
        if args.length_limit != -1 and index > args.length_limit:
            break
        if args.anchor_spk == 'none' or out_wav_path.parts[-2] == args.anchor_spk:
            out_wav, sr = torchaudio.load(str(out_wav_path), backend='sox')
            scores.append(model.predict(data=out_wav, sr=sr).item())

    print("UTMOS v2 score: ", sum(scores) / len(scores))

if __name__ == "__main__":
    args = check_argv()
    main(args)
import argparse
import torch
import whisper
from pathlib import Path
from tqdm import tqdm
import pandas as pd

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
        "--out_transcript_dir",
        type=Path,
        help="directory to store ASR output",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default='large',
        help="Whisper model used.",
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

def main(args):
    print("Converted dir: ", args.converted_dir)
    print("Out transcript dir: ", args.out_transcript_dir)
    print("Whisper Model: ", args.whisper_model)
    converted_dir = Path(args.converted_dir)
    
    if args.anchor_spk == 'none':
        out_transcript_dir = Path(args.out_transcript_dir)
    else:
        out_transcript_dir = Path(args.out_transcript_dir) / args.anchor_spk
    out_transcript_dir.mkdir(parents=True, exist_ok=True)

    extensions = ['wav', 'flac', 'mp3']

    model = whisper.load_model(args.whisper_model, device="cuda")

    if args.set == 'librispeech':
        spks = list(converted_dir.iterdir())
        for spk in tqdm(spks):
            wavs = []
            transcripts = {}
            for extension in extensions:
                wavs += (converted_dir / spk).rglob('*.' + extension)

            for wav in tqdm(wavs):
                if args.anchor_spk == 'none' or wav.parts[-2] == args.anchor_spk:
                    transcript = model.transcribe(str(wav), language="english")
                    transcripts[wav.stem] = transcript['text']

            with open(out_transcript_dir / (spk.stem + '.txt'), 'w') as out_transcript_file:
                for transcript in transcripts:
                    out_transcript_file.write(transcript + '|' + transcripts[transcript] + '\n')
    elif args.set == 'cv':
        wavs = []
        transcripts = {}
        for extension in extensions:
            wavs += list(converted_dir.rglob('*.' + extension))
        for wav in tqdm(wavs):
            if args.anchor_spk == 'none' or wav.parts[-2] == args.anchor_spk:
                transcript = model.transcribe(str(wav), language="english")
                transcripts[str(wav.relative_to(converted_dir))] = transcript['text']
        with open(out_transcript_dir / ('out.txt'), 'w') as out_transcript_file:
            for transcript in transcripts:
                out_transcript_file.write(transcript + '|' + transcripts[transcript] + '\n')


if __name__ == "__main__":
    args = check_argv()
    main(args)
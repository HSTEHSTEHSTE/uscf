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
        "--dataset",
        type=str,
        default='librispeech',
        help="dataset structure: librispeech or commonvoice",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default='large',
        help="Whisper model used.",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default='',
        help="CSV file. Only applicable for commonvoice",
    )
    return parser.parse_args()

def main(args):
    print("Converted dir: ", args.converted_dir)
    print("Out transcript dir: ", args.out_transcript_dir)
    print("Dataset: ", args.dataset)
    print("Whisper Model: ", args.whisper_model)
    converted_dir = Path(args.converted_dir)
    out_transcript_dir = Path(args.out_transcript_dir)
    out_transcript_dir.mkdir(parents=True, exist_ok=True)

    extensions = ['wav', 'flac', 'mp3']

    model = whisper.load_model(args.whisper_model, device="cuda")

    if args.dataset == 'librispeech':
        spks_long = converted_dir.iterdir()
        spks = []
        for spk_long in spks_long:
            spks.append(str(spk_long).split('/')[-1])
        for spk in tqdm(spks):
            wavs = []
            transcripts = {}
            for extension in extensions:
                wavs += (converted_dir / spk).rglob('*.' + extension)

            for wav in tqdm(wavs):
                transcript = model.transcribe(str(wav), language="english")
                transcripts[wav.stem] = transcript['text']

            with open(out_transcript_dir / (spk + '.txt'), 'w') as out_transcript_file:
                for transcript in transcripts:
                    out_transcript_file.write(transcript + '|' + transcripts[transcript] + '\n')

    elif args.dataset == 'commonvoice':
        if len(args.csv_file) > 0:
            csv_file = pd.read_csv(args.csv_file, sep='|', header=0, index_col=None, quoting=3)
            csv_file['transcript'] = ''
            for index, entry in tqdm(csv_file.iterrows(), total = csv_file.shape[0]):
                transcript = model.transcribe(str(converted_dir / entry['path']), language='en')
                csv_file.loc[index, 'transcript'] = transcript['text'].replace('|', ' ')
            csv_file.to_csv(out_transcript_dir / Path(args.csv_file).name, sep='|', index=False)
        else:
            wavs = []
            transcripts = {}
            for extension in extensions:
                wavs += converted_dir.rglob('*.' + extension)
            for wav in tqdm(wavs):
                transcript = model.transcribe(str(wav), language="english")
                transcripts[wav.stem] = transcript['text']

            with open(out_transcript_dir / ('out.txt'), 'w') as out_transcript_file:
                for transcript in transcripts:
                    out_transcript_file.write(transcript + '|' + transcripts[transcript] + '\n')


if __name__ == "__main__":
    args = check_argv()
    main(args)
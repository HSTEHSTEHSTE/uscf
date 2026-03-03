import argparse
import jiwer
from pathlib import Path
from whisper_normalizer.english import EnglishTextNormalizer

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_transcript_dir",
        type=Path,
        help="reference transcript directory",
    )
    parser.add_argument(
        "--out_transcript_dir",
        type=Path,
        help="output transcript directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='librispeech',
        help="dataset structure",
    )
    return parser.parse_args()

def main(args):
    english_normalizer = EnglishTextNormalizer()

    ref_transcript_dir = Path(args.ref_transcript_dir)
    out_transcript_dir = Path(args.out_transcript_dir)

    if args.dataset == 'librispeech':
        spks_long = out_transcript_dir.rglob('*.txt')
        spks = []
        for spk_long in spks_long:
            spks.append(spk_long.stem)
        wers = []
        for spk in spks:
            wers_spk = []
            ref_transcripts = {}
            with open(ref_transcript_dir / (spk + '.txt')) as ref_transcript_file:
                for line in ref_transcript_file:
                    line = line.strip()
                    line_elements = line.split('|')
                    ref_transcripts[line_elements[0]] = english_normalizer(line_elements[1].strip())
            out_transcripts = {}
            with open(out_transcript_dir / (spk + '.txt')) as out_transcript_file:
                for line in out_transcript_file:
                    line = line.strip()
                    line_elements = line.split('|')
                    out_transcripts[line_elements[0]] = english_normalizer(line_elements[1].strip())
            for utt in out_transcripts:
                wer = jiwer.wer(ref_transcripts[utt], out_transcripts[utt])
                wers_spk.append(wer)
            wers += wers_spk
    elif args.dataset == 'commonvoice':
        wers = []
        out_transcript_file_dir = out_transcript_dir / 'out.txt'
        ref_transcript_file_dir = ref_transcript_dir / 'out.txt'
        ref_transcripts = {}
        with open(ref_transcript_file_dir, 'r') as ref_transcript_file:
            for line in ref_transcript_file:
                line = line.strip()
                line_elements = line.split('|')
                ref_transcripts[line_elements[0]] = line_elements[1].strip()
        out_transcripts = {}
        with open(out_transcript_file_dir, 'r') as out_transcript_file:
            for line in out_transcript_file:
                line = line.strip()
                line_elements = line.split('|')
                out_transcripts[line_elements[0]] = line_elements[1].strip()
        for utt in out_transcripts:
            wer = jiwer.wer(ref_transcripts[utt], out_transcripts[utt])
            wers.append(wer)
    
    wer = sum(wers) / len(wers)
    print("WER: ", wer)    

if __name__ == "__main__":
    args = check_argv()
    main(args)
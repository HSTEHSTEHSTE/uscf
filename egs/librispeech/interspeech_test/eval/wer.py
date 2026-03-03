import argparse
import jiwer
import json
from pathlib import Path
import pandas as pd
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
        "--set",
        type=str,
        help="librispeech, cv",
        default="librispeech"
    )
    parser.add_argument(
        "--cv_inference_file",
        type=Path,
        default="linearvc/exp/tts/inference/inference_0.tsv"
    )
    return parser.parse_args()

def main(args):
    english_normalizer = EnglishTextNormalizer()

    ref_transcript_dir = Path(args.ref_transcript_dir)
    out_transcript_dir = Path(args.out_transcript_dir)

    wers = []
    if args.set == "librispeech":
        # collect reference transcripts
        with open('linearvc/egs/librispeech/libri_test/speakers.json', 'r') as file:
            speakers = json.load(file)
        ref_transcripts = {}
        for speaker in speakers['lists']['test-clean_source']:
            with open(ref_transcript_dir / 'test-clean' / (str(speaker) + '.txt'), 'r') as file:
                for line in file:
                    line_elements = line.strip().split('|')
                    ref_transcripts[line_elements[0]] = english_normalizer(line_elements[1])
        spks_long = out_transcript_dir.rglob('*.txt')
        spks = []
        for spk_long in spks_long:
            spks.append(spk_long.stem)

        for spk in spks:
            wers_spk = []
            out_transcripts = {}
            with open(out_transcript_dir / (spk + '.txt')) as out_transcript_file:
                for line in out_transcript_file:
                    line = line.strip()
                    line_elements = line.split('|')
                    utt_name = line_elements[0].split('_')[0]
                    out_transcript = english_normalizer(line_elements[1].strip())
                    wer = jiwer.wer(ref_transcripts[utt_name], out_transcript)
                    wers_spk.append(wer)
            wers += wers_spk
    elif args.set == "cv":
        ref_transcripts = {}
        if args.cv_inference_file is None:
            with open(ref_transcript_dir / 'out.txt') as ref_transcript_file:
                for line in ref_transcript_file:
                    line = line.strip()
                    line_elements = line.split('|')
                    ref_transcripts[line_elements[0].split('.')[0]] = english_normalizer(line_elements[1].strip())
        else:
            cv_inference_file_dir = Path(args.cv_inference_file)
            print("CV inference file: ", cv_inference_file_dir)
            cv_inference_file = pd.read_csv(cv_inference_file_dir, sep='|', header=0, index_col=None, quoting=3)
            for index, entry in cv_inference_file.iterrows():
                ref_transcripts[entry["out_wav_name"].split('.')[0]] = english_normalizer(entry["prompt_transcript"])

        with open(out_transcript_dir / 'out.txt') as out_transcript_file:
            for line in out_transcript_file:
                line = line.strip()
                line_elements = line.split('|')
                out_transcript = english_normalizer(line_elements[1].strip())
                utt_name = line_elements[0].split('/')[-1].split('.')[0]
                wer = jiwer.wer(ref_transcripts[utt_name], out_transcript)
                wers.append(wer)

    wer = sum(wers) / len(wers)
    print("WER: ", wer)    

if __name__ == "__main__":
    args = check_argv()
    main(args)
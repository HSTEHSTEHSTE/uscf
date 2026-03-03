import argparse
import json
import random
from pathlib import Path

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech_root",
        type=Path,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    return parser.parse_args()

def main(args):
    speakers = {}
    splits = ['test-clean', 'test-other', 'dev-clean']
    for split in splits:
        speakers[split] = [item.stem for item in (args.librispeech_root / split).iterdir() if item.is_dir()]

    random.seed(args.seed)
    speaker_lists = {}
    speaker_lists['test-clean_source'] = random.sample(speakers['test-clean'], 20)
    speaker_lists['test-other_target'] = random.sample(speakers['test-other'], 20)
    speaker_lists['dev-clean_heldout'] = random.sample(speakers['dev-clean'], 20)
    speaker_lists['test-clean_heldout'] = [speaker for speaker in speakers['test-clean'] if speaker not in speaker_lists['test-clean_source']]

    speaker_maps = {}
    for src_speaker in speaker_lists['test-clean_source']:
        speaker_maps[src_speaker] = random.sample(speaker_lists['test-other_target'], 5)

    speakers = {
        'lists': speaker_lists,
        'maps': speaker_maps
    }
    with open('egs/librispeech/libri_test/speakers.json', 'w') as file:
        json.dump(speakers, file)

if __name__ == "__main__":
    args = check_argv()
    main(args)
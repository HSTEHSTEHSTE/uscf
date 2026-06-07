# cf_tts/dataset.py
import os
import json, yaml
import random
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import torch, torchaudio
from torch.utils.data import Dataset
from torch.utils.data import Sampler


class TTSDatum:
    """
    One training example
    """
    def __init__(
        self,
        wav_path: str,
        text: str,
        speaker_id: str,
        num_frames: int,
        accent: str = None,
    ):
        self.wav_path = wav_path
        self.text = text
        self.speaker_id = speaker_id
        self.num_frames = num_frames
        self.accent = accent


class TTSDataset(Dataset):
    """
    TTS dataset used by CF-TTS.
    """

    def __init__(
        self,
        config_file_path: str,
        split: str = 'train', # train, dev, test
        data: list[TTSDatum] = None,
    ):
        self.resamplers = {}

        # read config
        with open(config_file_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        if data is not None:
            self.data = data

        else:
            self.data: List[TTSDatum] = []

            # load data
            print("Reading Data")
            if self.config['data'][split]['dataset'] == 'librispeech':
                for subset in self.config['data'][split]['librispeech_subsets']:
                    print("Processing ", subset)
                    transcript_files = list((Path(self.config['data']['librispeech_transcript_path']) / subset).glob('*.txt'))
                    spks = []
                    for transcript_file in tqdm(transcript_files):
                        spk = transcript_file.stem
                        spks.append(spk)
                        with open(transcript_file, 'r') as file:
                            for line in file:
                                line = line.strip().split('|')
                                text = line[1].strip()
                                filename = line[0].strip()
                                filename_elements = filename.split('-')
                                wav_path = Path(self.config['data']['librispeech_audio_path']) / subset / spk / filename_elements[1] / (filename + '.flac')
                                num_frames = torchaudio.info(wav_path, backend='soundfile').num_frames
                                self.data.append(TTSDatum(
                                    wav_path=wav_path,
                                    text=text,
                                    speaker_id=spk,
                                    num_frames=num_frames
                                ))
            elif self.config['data'][split]['dataset'] == 'commonvoice':
                audio_root = Path(self.config['data']['commonvoice_root_path']) / 'clips'
                for subset_file in self.config['data'][split]['commonvoice_subsets']:
                    print("Processing ", subset_file)
                    subset_df = pd.read_csv(subset_file, sep='|', header=0, index_col=None, quoting=3)
                    for entry in tqdm(subset_df.iterrows(), total=subset_df.shape[0]):
                        if self.config['data']['accent']['use_accent']:
                            self.data.append(TTSDatum(
                                wav_path=audio_root / entry[1].path,
                                text=entry[1]['sentence'],
                                speaker_id=entry[1]['client_id'],
                                num_frames=int(entry[1]['duration'] * 16000),
                                accent=entry[1]['accents']
                            ))
                        else:
                            self.data.append(TTSDatum(
                                wav_path=audio_root / entry[1].path,
                                text=entry[1]['sentence'],
                                speaker_id=entry[1]['client_id'],
                                num_frames=int(entry[1]['duration'] * 16000)
                            ))
            self.data.sort(key=lambda x: x.num_frames)

    # -------------------------
    # torch dataset
    # -------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if item.wav_path.suffix == '.mp3':
            wav, sr = torchaudio.load(item.wav_path)
        else:
            wav, sr = torchaudio.load(item.wav_path)
        if sr != 16000:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = self.resamplers[sr](wav)

        wav = wav.squeeze(0)

        wav = wav[: self.config['data']['max_audio_len']]

        return {
            "wav": wav,
            "text": item.text,
            "speaker": item.speaker_id,
            "accent": item.accent
        }


# -------------------------
# sampler
# -------------------------
class FrameBatchSampler(Sampler):
    """
    Groups samples so that the total number of audio frames
    per batch does not exceed max_frames.

    This gives:
        short utterances → large batches
        long utterances  → small batches
    """

    def __init__(
        self,
        dataset,
        config_file_path: str,
        split: str = 'train'
    ):
        self.dataset = dataset

        # read config
        with open(config_file_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        self.max_frames = self.config['training']['max_frames_per_batch']

        self.indices = list(range(len(dataset)))

        self.batches = []
        batch = []
        total_frames = 0
        for idx in self.indices:
            frames = min(self.dataset.data[idx].num_frames, self.config['data']['max_audio_len'])

            # if this utterance alone is too big, force it into its own batch
            if frames > self.max_frames:
                if batch:
                    self.batches.append(batch)
                    batch = []
                    total_frames = 0
                self.batches.append([idx])
                continue

            if total_frames + frames > self.max_frames and batch:
                self.batches.append(batch)
                batch = []
                total_frames = 0

            batch.append(idx)
            total_frames += frames

        if batch:
            self.batches.append(batch)
        if self.config['data'][split]['shuffle_batches']:
            random.seed(self.config['training']['random_seed'])
            random.shuffle(self.batches)
        self.shuffle = self.config['data'][split]['shuffle_batches_between_epochs']


    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
        

    def __len__(self):
        # PyTorch allows this to be approximate
        return len(self.batches)


# -------------------------
# batching
# -------------------------

class TTS_Collate:
    """
    Pads variable-length audio and text for batching.
    """
    def __init__(self, cfg):
        self.tokenizer_type = cfg['data']['text_tokenizer']['type']
        self.pre_phonemized = cfg['data']['text_tokenizer']['pre_phonemized']
        if self.tokenizer_type == 'spm':
            import sentencepiece as spm
            self.text_tokenizer = spm.SentencePieceProcessor()
            self.text_tokenizer.load(cfg['data']['text_tokenizer']['tokenizer_file'])
        elif self.tokenizer_type == 'phone':
            import json
            with open(cfg['data']['text_tokenizer']['tokenizer_file'], 'r') as phone_json_file:
                tokens = json.load(phone_json_file)
                self.text_tokenizer = {}
                punctuation_marks = ''
                for category in tokens.keys():
                    category_tokens = tokens[category]
                    for token_id, token in enumerate(category_tokens):
                        self.text_tokenizer[token] = token_id
                        if category == 'punctuations':
                            punctuation_marks += token
            if not self.pre_phonemized:
                from phonemizer.backend import EspeakBackend
                from phonemizer.separator import Separator
                self.separator = Separator(phone='-', word=' ')
                if len(punctuation_marks) > 0:
                    backend = EspeakBackend(
                        language='en-us', 
                        preserve_punctuation=True, 
                        punctuation_marks=punctuation_marks,
                        words_mismatch='ignore'
                    )
                else:
                    backend = EspeakBackend(
                        language='en-us', 
                        preserve_punctuation=True, 
                        words_mismatch='ignore'
                    )
                self.phonemize = backend.phonemize
        
        # accent tokenizer
        self.use_accent = cfg['data']['accent']['use_accent']
        if self.use_accent:
            with open(cfg['data']['accent']['accent_file'], 'r') as file:
                self.accents = json.load(file)['accents']
            self.accent_to_id = {}
            for index, accent in enumerate(self.accents):
                self.accent_to_id[accent] = index


    def __call__(self, batch: List[Dict]):
        wavs = [b["wav"] for b in batch]
        speakers = [b["speaker"] for b in batch]
        texts_raw = [str(b["text"]) for b in batch]
        texts = []
        if self.tokenizer_type == 'spm':
            for text_raw in texts_raw:
                texts.append(self.text_tokenizer.encode_as_ids(text_raw))
        elif self.tokenizer_type == 'phone':
            if self.pre_phonemized:
                for text_raw in texts_raw:
                    text_raw = text_raw.split(' ')
                    texts.append([self.text_tokenizer[phone] for phone in text_raw if phone in self.text_tokenizer])
            else:
                phones = self.phonemize(texts_raw, separator=self.separator, strip=True)
                phones = [re.sub(r"""([;:,.!?¡¿—…"«»“”\(\)\{\}\[\]])""", r"-\1", phone) for phone in phones]
                phones = [phone.replace('- ', '-').replace(' ', '-').split('-') for phone in phones]
                for text in phones:
                    texts.append([self.text_tokenizer[token] for token in text if token in self.text_tokenizer])

        if self.use_accent:
            accents_raw = [b["accent"] for b in batch]
            accents = [self.accent_to_id[accent] for accent in accents_raw]
        else:
            accents = []

        lengths = torch.tensor([w.size(0) for w in wavs])
        max_len = max(lengths)

        padded = torch.zeros(len(wavs), max_len)
        for i, w in enumerate(wavs):
            padded[i, : w.size(0)] = w

        return {
            "wav": padded,
            "wav_lengths": lengths,
            "text": texts,
            "speaker": speakers,
            "accent": accents,
        }
    
    

if __name__ == "__main__":
    # test dataset object
    dataset = TTSDataset(config_file_path='cf_tts/config/config.yaml')
    print(len(dataset))
    print(dataset.__getitem__(0))

    # test sampler
    sampler = FrameBatchSampler(dataset, config_file_path='cf_tts/config/config.yaml')
    print(len(sampler))
    for batch in sampler:
        print(batch)
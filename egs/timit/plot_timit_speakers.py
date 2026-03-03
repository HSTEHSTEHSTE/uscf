import argparse
import hashlib
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
# from umap import UMAP
from scipy.interpolate import interp1d
from scipy.spatial import distance
from sklearn.metrics import roc_curve
from scipy.optimize import brentq

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timit_cf_feat_path",
        type=Path,
        help="TIMIT cf feature directory",
    )
    parser.add_argument(
        "--timit_path",
        type=Path,
        help="TIMIT root path",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        help="cf, wavlm, contentvec",
    )
    return parser.parse_args()

def eer(y, y_score):
    fpr, tpr, _ = roc_curve(y, 1 - y_score, pos_label=1)
    return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

def main(args):
    content_path_root = Path(args.timit_cf_feat_path)
    print("content path: ", content_path_root)
    timit_path_root = Path(args.timit_path)
    print("timit path: ", timit_path_root)

    phoneme_frames = {
        'TRAIN': {},
        'TEST': {}
    }

    for subset in phoneme_frames:
        content_path = content_path_root / subset
        timit_path = timit_path_root / subset
        content_files = list(content_path.rglob('*.npy'))
        for content_file in tqdm(content_files):
            if args.feature_type in ['wavlm', 'contentvec'] and content_file.parts[-4] != 'utts':
                continue
            content = np.load(content_file, allow_pickle=True)
            if args.feature_type in ['wavlm', 'contentvec']:
                phn_file = timit_path / (content_file.relative_to(content_path / 'utts')).parent / (content_file.stem + '.PHN')
            elif args.feature_type == 'cf':
                phn_file = timit_path / (content_file.relative_to(content_path)).parent / (content_file.stem + '.PHN')
            speaker = str(phn_file).split('/')[-2]
            with open(phn_file, 'r') as phns:
                for line in phns:
                    line_elements = line.strip().split(' ')
                    start = int(int(line_elements[0]) / 16 / 20) # /16: frame to millisecond; 20: num of milliseconds in each WavLM frame
                    end = int(int(line_elements[1]) / 16 / 20)
                    phn = line_elements[2]
                    frames = content[start:end]
                    if start < end:
                        if phn != 'h#':
                            if phn not in phoneme_frames[subset]:
                                phoneme_frames[subset][phn] = {
                                    speaker: list(frames)
                                }
                            else:
                                if speaker not in phoneme_frames[subset][phn]:
                                    phoneme_frames[subset][phn][speaker] = list(frames)
                                else:
                                    phoneme_frames[subset][phn][speaker] += list(frames)

        for phn in phoneme_frames[subset]:
            for speaker in phoneme_frames[subset][phn]:
                phoneme_frames[subset][phn][speaker] = np.stack(phoneme_frames[subset][phn][speaker], axis=0)
        # np.save(content_path.parent / 'phonemes_spk.npy', phoneme_frames)

    def persistent_hash_string(data_string):
        """Generates a persistent SHA256 hash for a string."""
        return abs(int(hashlib.sha256(data_string.encode('utf-8')).hexdigest(), 16)) % 2**32

    seed = persistent_hash_string('timit_experiment')
    np.random.seed(seed)

    sample_size = 5
    num_speakers = 10
    samples = []
    chosen_phoneme = 'eh'

    phone_embs = {}
    for phoneme in phoneme_frames['TRAIN'].keys():
        phone_emb_spks = []
        for speaker in phoneme_frames['TRAIN'][phoneme]:
            phone_emb_spks.append(phoneme_frames['TRAIN'][phoneme][speaker])
        phone_embs[phoneme] = np.concatenate(phone_emb_spks, axis=0).mean(axis=0)

    sims = []
    desims = []
    spk_sims = []
    spk_desims = []
    phone_scores = {}
    # phonemes = ['sh', 'iy', 'ae', 'd', 'aa', 'k', 's', 'g', 'w', 'q']
    phonemes = list(phone_embs.keys())

    random.seed(42)

    for phoneme in tqdm(phonemes):
        phone_scores[phoneme] = {
            'labels': [],
            'scores': [],
            'utt_labels': [],
            'utt_scores': [],
        }
        spks = list(phoneme_frames['TEST'][phoneme].keys())
        spk_ref_embs = {}
        for speaker in spks:
            spk_ref_embs[speaker] = phoneme_frames['TEST'][phoneme][speaker].mean(axis=0)
        for speaker in spks:
            for index, emb in enumerate(phoneme_frames['TEST'][phoneme][speaker]):
                for ref_phone in phonemes:
                    ref_emb = phone_embs[ref_phone]
                    dist = distance.cosine(ref_emb, emb)
                    phone_scores[phoneme]['labels'].append(int(ref_phone == phoneme))
                    phone_scores[phoneme]['scores'].append(dist)
                    if ref_phone == phoneme:
                        sims.append(1 - dist)
                    else:
                        desims.append(1 - dist)
        
                for ref_spk in spks:
                    ref_emb = spk_ref_embs[ref_spk]
                    dist = distance.cosine(ref_emb, emb)
                    phone_scores[phoneme]['utt_labels'].append(int(ref_spk == speaker))
                    phone_scores[phoneme]['utt_scores'].append(dist)
                    if ref_spk == speaker:
                        spk_sims.append(1 - dist)
                    else:
                        spk_desims.append(1 - dist)

                # # choose one target trial
                # if phoneme_frames['TEST'][phoneme][speaker].shape[0] > 2:
                #     target_emb_index = random.randint(0, phoneme_frames['TEST'][phoneme][speaker].shape[0] - 2)
                #     if target_emb_index < index:
                #         target_emb = phoneme_frames['TEST'][phoneme][speaker][target_emb_index]
                #     else:
                #         target_emb = phoneme_frames['TEST'][phoneme][speaker][target_emb_index + 1]
                #     dist = distance.cosine(target_emb, emb)
                #     phone_scores[phoneme]['utt_labels'].append(1)
                #     phone_scores[phoneme]['utt_scores'].append(dist)
            
                # # choose 5 non-target trials
                # target_spks = random.sample(spks, k=5)
                # for target_spk in target_spks:
                #     target_emb_index = random.randint(0, phoneme_frames['TEST'][phoneme][target_spk].shape[0] - 1)
                #     target_emb = phoneme_frames['TEST'][phoneme][target_spk][target_emb_index]
                #     dist = distance.cosine(target_emb, emb)
                #     phone_scores[phoneme]['utt_labels'].append(0)
                #     phone_scores[phoneme]['utt_scores'].append(dist)

    eers = []
    for phoneme in phonemes:
        eers.append(eer(phone_scores[phoneme]['labels'], np.array(phone_scores[phoneme]['scores'])))
    utt_eers = []
    for phoneme in phonemes:
        utt_eers.append(eer(phone_scores[phoneme]['utt_labels'], np.array(phone_scores[phoneme]['utt_scores'])))
    print("Phoneme similarity: ", np.array(sims).mean())
    print("Non-target phoneme similarity: ", np.array(desims).mean())
    print("Speaker similarity: ", np.array(spk_sims).mean())
    print("Non-target speaker similarity: ", np.array(spk_desims).mean())
    print("Phoneme EER: ", np.array(eers).mean())
    print("Speaker EER: ", np.array(utt_eers).mean())

    breakpoint()

    # speaker_indices = np.random.choice(np.arange(len(list(phoneme_frames[chosen_phoneme].keys()))), size=num_speakers, replace=False)
    # speakers = []
    # all_speakers = list(phoneme_frames[chosen_phoneme].keys())
    # for speaker_index in speaker_indices:
    #     speakers.append(all_speakers[speaker_index])
    # for speaker in speakers:
    #     if len(phoneme_frames[chosen_phoneme][speaker]) < sample_size:
    #         sample_size = len(phoneme_frames[chosen_phoneme][speaker])
    speakers = ['FAKS0', 'FELC0', 'FJEM0', 'MRES0', 'MPAM0', 'MJLN0', 'MAJC0', 'FJSJ0', 'FCMH1', 'MDBB0']
    for speaker in speakers:
        phn_sample_indices = np.random.choice(np.arange(phoneme_frames[chosen_phoneme][speaker].shape[0]), size=sample_size, replace=False)
        samples.append(phoneme_frames[chosen_phoneme][speaker][phn_sample_indices])

    # # PCA
    # print("Fitting PCA")
    # samples = np.concatenate(samples, axis=0)
    # scaler = StandardScaler()
    # samples_scaled = scaler.fit_transform(samples)
    # pca = PCA(n_components=2)
    # samples_pca = pca.fit_transform(samples)

    # for speaker_index, speaker in enumerate(speakers):
    #     points = samples_pca[speaker_index * sample_size: (speaker_index + 1) * sample_size]
    #     plt.scatter(points[:, 0], points[:, 1], label=speaker)

    # plt.legend()
    # plt.show()
    # plt.savefig(content_path.parent.parent / 'pca.png')
    # plt.clf()

    # # t-SNE
    # print("Fitting t-SNE")
    # tsne = TSNE(n_components=2, random_state=seed)
    # tsne_results = tsne.fit_transform(samples)

    # for speaker_index, speaker in enumerate(speakers):
    #     points = tsne_results[speaker_index * sample_size: (speaker_index + 1) * sample_size]
    #     plt.scatter(points[:, 0], points[:, 1], label=speaker)

    # plt.legend()
    # plt.show()
    # plt.savefig(content_path.parent.parent / 'tsne.png')
    # plt.clf()

    # UMAP
    print("Fitting UMAP")
    umap = UMAP(n_components=2, init='random', random_state=seed)
    proj = umap.fit_transform(samples)

    for speaker_index, speaker in enumerate(speakers):
        points = proj[speaker_index * sample_size: (speaker_index + 1) * sample_size]
        plt.scatter(points[:, 0], points[:, 1], label=speaker)

    plt.legend()
    plt.show()
    plt.savefig(content_path.parent.parent / 'umap.png')
    plt.clf()

if __name__ == "__main__":
    args = check_argv()
    main(args)
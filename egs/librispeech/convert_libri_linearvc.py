import argparse, random
import torch, torchaudio
from pathlib import Path
from tqdm import tqdm
from linearvc.linearvc import LinearVC

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=Path,
        help="source speech directory",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="output speech directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
    )
    return parser.parse_args()

def get_spk_mapping(spks, seed):
    random.seed(seed)
    if len(spks) < 2:
        raise ValueError("Not enough speakers")

    while True:
        shuffled = spks[:]
        random.shuffle(shuffled)

        # check no one maps to themselves
        if all(a != b for a, b in zip(spks, shuffled)):
            return dict(zip(spks, shuffled))

def main(args):
    print("Source dir: ", args.src_dir)
    print("Out dir: ", args.out_dir)
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    extensions = ['wav', 'flac']
    spks_long = src_dir.iterdir()
    spks = []
    for spk_long in spks_long:
        spks.append(str(spk_long).split('/')[-1])
    spk_map = get_spk_mapping(spks, args.seed)

    # Load all the required models
    wavlm = torch.hub.load(
        "bshall/knn-vc", 
        "wavlm_large", 
        trust_repo=True, 
        progress=True, 
        device=device, 
    )
    hifigan, _ = torch.hub.load(
        "bshall/knn-vc",
        "hifigan_wavlm",
        trust_repo=True,
        prematched=True,
        progress=True,
        device=device,
    )
    linearvc_model = LinearVC(wavlm, hifigan, device)

    for spk_src in tqdm(spk_map, total=len(spks)):
        spk_tgt = spk_map[spk_src]

        wavs_src = []
        for extension in extensions:
            wavs_src += (src_dir / spk_src).rglob('*.' + extension)

        wavs_tgt = []
        for extension in extensions:
            wavs_tgt += (src_dir / spk_tgt).rglob('*.' + extension)

        # Voice conversion projection matrix
        W = linearvc_model.get_projmat(
            wavs_src,
            wavs_tgt,
            parallel=False,  # enable if parallel
            vad=False,
        )

        for wav_src in wavs_src:
            input_features = linearvc_model.get_features(wav_src)
            output_wav = linearvc_model.project_and_vocode(input_features, W)
            (out_dir / spk_src).mkdir(parents=True, exist_ok=True)
            torchaudio.save(out_dir / spk_src / (wav_src.stem + '.wav'), output_wav[None], 16000)

if __name__ == "__main__":
    args = check_argv()
    main(args)
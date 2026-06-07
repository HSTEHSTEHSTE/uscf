import argparse, random
import torch, torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from linearvc import linearvc

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv_dir",
        type=Path,
        help="commonvoice directory",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="output speech directory",
    )
    parser.add_argument(
        "--data_file",
        type=Path,
        help="commonvoice data file",
    )
    parser.add_argument(
        "--content_factorization_path",
        type=Path,
        help="root path to content factorization"
    )
    parser.add_argument(
        "--pseudoinverse_type",
        type=str,
        default='lstsq, anchor, true',
        help="lstsq"
    )
    parser.add_argument(
        "--pinv_anchor",
        type=str,
        default='1272',
        help="Pinv anchor ID"
    )
    return parser.parse_args()


def main(args):
    print("CV dir: ", args.cv_dir)
    print("Out dir: ", args.out_dir)
    print("Content factorization path: ", args.content_factorization_path)
    cv_dir = Path(args.cv_dir)
    out_dir = Path(args.out_dir)

    content_path = Path(args.content_factorization_path)
    spk_anchor = args.pinv_anchor
    transforms = np.load(content_path / 'transforms.npy', allow_pickle=True).item()
    if args.pseudoinverse_type == 'lstsq':
        ST = np.load(content_path / 'ST.npy')
    elif args.pseudoinverse_type == 'anchor':
        ST = transforms[spk_anchor]

    extensions = ['wav', 'flac']

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
    linearvc_model = linearvc.LinearVC(wavlm, hifigan, device)

    input_wavs = pd.read_csv(args.data_file, sep=',', header=0, index_col=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in tqdm(input_wavs.iterrows(), total=input_wavs.shape[0]):
        out_path = out_dir / (entry[1].path[:-4] + '.wav')
        if not out_path.is_file():
            spk_tgt = spk_anchor
            input_features = linearvc_model.get_features(str(cv_dir / 'clips' / entry[1].path))
            input_features = input_features.cpu().numpy()
            out_features = torch.tensor(np.dot(np.dot(input_features, np.linalg.pinv(transforms[spk_anchor])), transforms[spk_tgt])).to(device)
            wav_hat = hifigan(out_features.unsqueeze(0)).squeeze(0).detach().cpu()
            torchaudio.save(str(out_dir / (entry[1].path[:-4] + '.wav')), wav_hat, 16000)


if __name__ == "__main__":
    args = check_argv()
    main(args)
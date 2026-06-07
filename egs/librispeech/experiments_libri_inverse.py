import argparse
from pathlib import Path

import numpy as np


def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transform_root_dir",
        type=Path,
        help="transform root directory",
    )
    parser.add_argument(
        "--transform_type",
        type=str,
        help="Type of transform. ST, UTXSS, or USTXS",
    )
    return parser.parse_args()

def main(args):
    transform_root_dir = Path(args.transform_root_dir)

    if args.transform_type == 'ST':
        # compute ST
        VT = np.load(transform_root_dir / "VT.npy") # [k, r, d]
        k, r, d = VT.shape
        VT = VT.reshape([-1, VT.shape[-1]]) # [k * r, d]
        target = np.identity(r) # [r, r]
        target = np.expand_dims(target, 0) # [1, r, r]
        target = target.repeat(k, axis=0) # [k, r, r]
        target = target.reshape([-1, target.shape[-1]]) # [k * r, r]
        lstsq_solution = np.linalg.lstsq(VT, target) 
        ST = lstsq_solution[0] # [1024, 75]
        np.save(transform_root_dir / 'ST.npy', ST)

    else:
        # compute UTXSS
        XS = np.load(transform_root_dir / "XS.npy") # [t, k * d]
        XS = XS.reshape([XS.shape[0], -1, 1024]) # [t, k, d]
        t, k, d = XS.shape
        XS = XS.reshape([-1, d])

        U = np.load(transform_root_dir / "U.npy") # [t, r]
        S = np.load(transform_root_dir / "S.npy") # [r]
        S = np.diag(S) # [r, r]
        US = np.matmul(U, S) # [t, r]
        US = np.expand_dims(US, 1).repeat(k, 1) # [t, k ,r]
        r = US.shape[-1]
        US = US.reshape([-1, r])

        U = np.expand_dims(U, 1).repeat(k, 1) # [t, k ,r]
        U = U.reshape([-1, r])

        if args.transform_type == 'UTXSS':
            lstsq_solution = np.linalg.lstsq(XS, U)
            UTXS = lstsq_solution[0]
            UTXSS = np.matmul(UTXS, S)
            np.save(transform_root_dir / 'UTXSS.npy', UTXSS)
        else:
            US = US / 100
            lstsq_solution = np.linalg.lstsq(XS, US)
            np.save(transform_root_dir / 'USTXS.npy', lstsq_solution[0] * 100)




if __name__ == "__main__":
    args = check_argv()
    main(args)
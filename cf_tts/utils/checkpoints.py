import re

import torch


def manage_checkpoints(ckpt_dir, keep_loss=5, keep_epoch=5, keep_step=5):
    LOSS_RE  = re.compile(r"ckpt_loss_([0-9.]+)_")
    EPOCH_RE = re.compile(r"_epoch_([0-9]+)\.pt$")
    STEP_RE  = re.compile(r"_step_([0-9]+)\.pt$")
    """
    Deletes checkpoints in ckpt_dir, keeping:
      - lowest `keep_loss` losses
      - highest `keep_epoch` epochs
      - highest `keep_step` steps

    Filenames must look like:
      ckpt_loss_1.23_epoch_45.pt
      ckpt_loss_1.23_step_1000.pt
    """
    ckpts = list(ckpt_dir.glob("ckpt_*.pt"))

    by_loss = []
    by_epoch = []
    by_step = []

    for ckpt in ckpts:
        name = ckpt.name

        loss = None
        epoch = None
        step = None

        m = LOSS_RE.search(name)
        if m:
            loss = float(m.group(1))

        m = EPOCH_RE.search(name)
        if m:
            epoch = int(m.group(1))

        m = STEP_RE.search(name)
        if m:
            step = int(m.group(1))

        if loss is not None and epoch is not None:
            by_loss.append((loss, ckpt))
        if epoch is not None:
            by_epoch.append((epoch, ckpt))
        if step is not None:
            by_step.append((step, ckpt))

    # best loss = smallest
    by_loss.sort(key=lambda x: x[0])
    keep_by_loss = {p for _, p in by_loss[:keep_loss]}

    # latest epoch = largest
    by_epoch.sort(key=lambda x: -x[0])
    keep_by_epoch = {p for _, p in by_epoch[:keep_epoch]}

    # latest step = largest
    by_step.sort(key=lambda x: -x[0])
    keep_by_step = {p for _, p in by_step[:keep_step]}

    keep = keep_by_loss | keep_by_epoch | keep_by_step

    deleted = []

    for ckpt in ckpts:
        if ckpt not in keep:
            deleted.append(ckpt)
            ckpt.unlink()

    return {
        "kept": list(keep),
        "deleted": deleted,
    }


def save_checkpoint(model, optimizer, step, path, outdir, cfg):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )
    ckpt_logs = manage_checkpoints(outdir, keep_loss=cfg['training']['num_ckpt_best_loss'], keep_epoch=cfg['training']['num_ckpt_latest_epochs'], keep_step=cfg['training']['num_ckpt_latest_steps'])


def load_checkpoint(model, path, device, optimizer=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]
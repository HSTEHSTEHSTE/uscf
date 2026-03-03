import argparse, math, shutil
from pathlib import Path
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import detect_anomaly

import time
from linearvc import linearvc
from linearvc.cf_tts.dataset import TTSDataset, FrameBatchSampler, TTS_Collate
from linearvc.cf_tts.models.tts import ZipVoice
from linearvc.cf_tts.utils.common import create_grad_scaler, normalize_input, load_config, format_time, get_speaker_feats, match_knn
from linearvc.cf_tts.utils.checkpoints import load_checkpoint, manage_checkpoints, save_checkpoint
from linearvc.cf_tts.utils.optim import get_scheduler

# -------------------------
# helpers
# -------------------------

def validation_pass(dev_loader, linearvc_model, transform, cfg, model, device, step, epoch_batch_length):
    model.eval()

    total_loss = 0.
    total = 0
    for batch_index, batch in enumerate(dev_loader):
        if cfg['data']['dev']['epoch_batch_limit'] > 0 and batch_index >= cfg['data']['dev']['epoch_batch_limit']:
            break
        wavs = batch["wav"].to(device)            # (B, T)
        wav_lengths = batch["wav_lengths"]
        text_ids = batch["text"]                  # list[str]
        speakers = batch["speaker"]               # list[str]
        accents = batch["accent"]

        with torch.no_grad():
            if cfg['training']['content_factorization']['type'] == 'fbank':
                input_features = transform(wavs) # [b, t, d]
            else:
                input_features, _ = linearvc_model.wavlm.extract_features(wavs, output_layer=6)
                if cfg['training']['content_factorization']['type'] == 'content':
                    if transform is not None:
                        input_features = torch.matmul(input_features, transform)
                elif cfg['training']['content_factorization']['type'] == 'speaker':
                    input_features = match_knn(input_features, transform)
                elif cfg['training']['content_factorization']['type'] == 'none':
                    pass
            input_features = input_features * cfg['training']['feature_scale']
            if cfg['training']['normalize_input']:
                input_features = normalize_input(input_features)
            input_features = input_features.detach()

            if cfg['training']['content_factorization']['type'] == 'fbank':
                wav_lengths = torch.full([input_features.shape[0]], input_features.shape[1]).to(device)
            else:
                wav_lengths = (torch.floor((wav_lengths - 400) / 320) + 1).to(device)

            # forward
            loss = model(
                tokens=text_ids,
                features=input_features,
                features_lens=wav_lengths,
                noise=cfg['training']['noise_scale'] * torch.randn_like(input_features).to(device), # Note: noise added to features. Not uniform random noise
                t=torch.rand(input_features.shape[0], 1, 1, device=device),
                accents=accents,
                condition_drop_ratio=0.
            )

        total_loss += loss.item()
        total += 1

    print("Validation")
    print(f"epoch {(step / epoch_batch_length):.3f} | total step {step} | val loss {(total_loss / total):.4f}", flush=True)
    model.train()
    return total_loss / total

# -------------------------
# main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_path", default='')
    parser.add_argument("--ignore_current_step", default=False)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(cfg, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login()
    wandb.init(
        project=cfg['training']['project_name'],
        name=cfg['training']['run_name'],
        id=cfg['training']['run_name'],
    )

    # -------------------------
    # dataset
    # -------------------------

    load_path = None
    if cfg['data']['train']['load_path'] is not None:
        import pickle
        load_path = Path(cfg['data']['train']['load_path'])
        if load_path.is_file():
            print("Reading pre-processed data file", flush=True)
            with open(load_path, 'rb') as file:
                data_list = pickle.load(file)
            train_set = TTSDataset(config_file_path=args.config, split='train', data=data_list)
        else:
            print("Pre-processing data and saving to load_path", flush=True)
            train_set = TTSDataset(config_file_path=args.config, split='train')
            with open(load_path, 'wb') as file:
                pickle.dump(train_set.data, file)
    else:
        print("Pre-processing data", flush=True)
        train_set = TTSDataset(config_file_path=args.config, split='train')
    train_sampler = FrameBatchSampler(train_set, args.config, split='train')

    if cfg['data']['dev']['load_path'] is not None:
        import pickle
        with open(cfg['data']['dev']['load_path'], 'rb') as file:
            data_list = pickle.load(file)['data']
        dev_set = TTSDataset(config_file_path=args.config, split='dev', data=data_list)
    else:
        dev_set = TTSDataset(config_file_path=args.config, split='dev')
    dev_sampler = FrameBatchSampler(dev_set, args.config, split='dev')
    

    tts_collate = TTS_Collate(cfg)
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        collate_fn=tts_collate,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_sampler=dev_sampler,
        collate_fn=tts_collate,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
    )

    # -------------------------
    # model
    # -------------------------

    model = ZipVoice(
        **cfg["model"]["tts"]["zipvoice"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"]
    )

    if cfg['data']['train']['epoch_batch_limit'] > 0:
        epoch_batch_length = min(len(train_loader), cfg['data']['train']['epoch_batch_limit'])
    else:
        epoch_batch_length = len(train_loader)

    if args.checkpoint_path is not None and args.checkpoint_path != 'none':
        current_step = load_checkpoint(model, args.checkpoint_path, device, optimizer=optimizer)
        current_epoch = math.floor((current_step) / epoch_batch_length)
    else:
        current_step = -1
        current_epoch = 0
    if args.ignore_current_step:
        current_step = -1
        current_epoch = 0

    scheduler = get_scheduler(cfg['optim']['scheduler_type'], optimizer, -1, cfg['optim']['scheduler_args'])
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr}")

    scaler = create_grad_scaler()

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

    if cfg['training']['content_factorization']['type'] == 'content':
        if cfg['training']['content_factorization']['content_factorization_file'] is not None:
            transform = np.load(cfg['training']['content_factorization']['content_factorization_file'], allow_pickle=True).item()
            transform = torch.tensor(np.linalg.pinv(transform[list(transform.keys())[0]])).to(device)
        else:
            transform = None
    elif cfg['training']['content_factorization']['type'] == 'speaker':
        from cuvs.neighbors import brute_force
        feats = get_speaker_feats(
            tgt_speaker_root=cfg['training']['content_factorization']['factorization_speaker'],
            linearvc_model=linearvc_model,
            device=device
        )
        index = brute_force.build(feats)
        transform = {
            'feats': feats,
            'index': index,
            'brute_force': brute_force
        }
    elif cfg['training']['content_factorization']['type'] == 'none':
        transform = None
    elif cfg['training']['content_factorization']['type'] == 'fbank':
        from speechbrain.lobes.features import Fbank
        transform = Fbank(sample_rate=16000, n_mels=cfg['model']['tts']['zipvoice']['feat_dim'])


    # -------------------------
    # training loop
    # -------------------------

    outdir = Path(cfg["training"]["out_dir"])
    if not (outdir / 'cfg' / Path(args.config).name).is_file():
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / 'cfg').mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, outdir / 'cfg')

    step = current_epoch * epoch_batch_length
    model.train()
    start_time = time.time()

    for epoch in range(cfg["training"]["epochs"]):
        losses = []
        if epoch < current_epoch:
            continue

        print(f"\nEpoch {epoch}", flush=True)

        for batch_index, batch in enumerate(train_loader):
            if step < current_step:
                step += 1
                continue
            if cfg['data']['train']['epoch_batch_limit'] > 0 and batch_index >= cfg['data']['train']['epoch_batch_limit']:
                break
            wavs = batch["wav"].to(device)            # (B, T)
            wav_lengths = batch["wav_lengths"]
            text_ids = batch["text"]                  # list[str]
            speakers = batch["speaker"]               # list[str]
            accents = batch["accent"]

            with torch.no_grad():
                if cfg['training']['content_factorization']['type'] == 'fbank':
                    input_features = transform(wavs) # [b, t, d]
                else:
                    input_features, _ = linearvc_model.wavlm.extract_features(wavs, output_layer=6)
                    if cfg['training']['content_factorization']['type'] == 'content':
                        if transform is not None:
                            input_features = torch.matmul(input_features, transform)
                    elif cfg['training']['content_factorization']['type'] == 'speaker':
                        # perform knn matching
                        batch_size = input_features.shape[0] # b
                        input_features = input_features.view(-1, input_features.shape[-1]) # [b * t, d]
                        _, neighbors = transform['brute_force'].search(transform['index'], input_features, k=4)
                        neighbors = torch.as_tensor(neighbors, device='cuda') # [b * t, k]
                        neighbors = neighbors.view(-1) # [b * t * k]
                        input_features = transform['feats'].index_select(0, neighbors) # [b * t * k, d]
                        input_features = input_features.view(-1, 4, input_features.shape[-1]) # [b * t, k, d]
                        input_features = torch.mean(input_features, dim=1) # [b * t, d]
                        input_features = input_features.view(batch_size, -1, input_features.shape[-1])
                    elif cfg['training']['content_factorization']['type'] == 'none':
                        pass
                input_features = input_features * cfg['training']['feature_scale']
                if cfg['training']['normalize_input']:
                    input_features = normalize_input(input_features)
                input_features = input_features.detach()

            if cfg['training']['content_factorization']['type'] == 'fbank':
                wav_lengths = torch.full([input_features.shape[0]], input_features.shape[1]).to(device)
            else:
                wav_lengths = (torch.floor((wav_lengths - 400) / 320) + 1).to(device)

            # forward
            torch.manual_seed(step)
            loss = model(
                tokens=text_ids,
                features=input_features,
                features_lens=wav_lengths,
                accents=accents,
                noise=cfg['training']['noise_scale'] * torch.randn_like(input_features).to(device), # Note: noise added to features. Not uniform random noise
                t=torch.rand(input_features.shape[0], 1, 1, device=device),
                condition_drop_ratio=cfg['training']['condition_drop_ratio']
            )

            optimizer.zero_grad()
            if cfg['training']['use_grad_scaler']:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            losses.append(loss.item())

            if not (list(model.parameters())[0].grad.isnan().any()):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_grad_norm'])
                optimizer.step()

            if step % cfg['training']['log_every'] == 0:
                current_time = time.time()
                print(f"epoch {(step / epoch_batch_length):.3f} | time elapsed {format_time(start_time, current_time)} | total step {step} | loss {(loss).item():.4f}", flush=True)
                wandb.log(
                    data = {
                        "train_loss": loss.item(),
                    },
                    step = step
                )

            step += 1

            if step % cfg["training"]["save_every_steps"] == 0 and step > 0:
                validation_loss = validation_pass(
                    dev_loader, 
                    linearvc_model, 
                    transform, 
                    cfg, 
                    model, 
                    device, 
                    step, 
                    epoch_batch_length
                )
                wandb.log(
                    data = {
                        "validation_loss": validation_loss,
                    },
                    step = step
                )
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    outdir / f"ckpt_loss_{validation_loss:.2f}_step_{step}.pt",
                    outdir,
                    cfg
                )

        
        current_time = time.time()
        print(f"epoch {epoch} | time elapsed {format_time(start_time, current_time)} | avg loss {(sum(losses) / len(losses)):.4f}")
        if epoch % cfg["training"]["save_every_epochs"] == 0 and step > 0:
            validation_loss = validation_pass(
                dev_loader, 
                linearvc_model, 
                transform, 
                cfg, 
                model, 
                device, 
                step, 
                epoch_batch_length
            )
            wandb.log(
                data = {
                    "validation_loss": validation_loss,
                },
                step = step
            )
            save_checkpoint(
                model,
                optimizer,
                step,
                outdir / f"ckpt_loss_{validation_loss:.2f}_epoch_{epoch}.pt",
                outdir,
                cfg
            )

        scheduler.step()


if __name__ == "__main__":
    main()

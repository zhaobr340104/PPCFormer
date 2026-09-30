"""Train the final PPCFormer on one paper dataset."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppcformer import PPCFormer
from ppcformer.data import SpectralDataset, TRAINING, WAVELENGTHS, validate_dataset
from ppcformer.utils import (
    capture_rng, evaluate, file_sha256, initialize_model, load_payload, load_weights,
    resolve_device, restore_rng, runtime_info, save_checkpoint, seed_everything,
    seed_worker, should_validate, validation_schedule, write_json,
)
from ppcformer.metrics import charbonnier_loss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(TRAINING), required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Prepared dataset root containing train/validation/test")
    parser.add_argument("--output", type=Path, required=True, help="A new run directory; existing runs require --resume")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="Default: 4000 for Chikusei, 3500 for CAVE")
    parser.add_argument("--batch-size", type=int, help="Default: 8 for Chikusei, 32 for CAVE")
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--resume", type=Path, help="Resume this release's last.pth in the original output directory")
    parser.add_argument("--stop-after-epoch", type=int, help="Stop early without changing the full cosine schedule")
    parser.add_argument("--no-cache", action="store_true", help="Load MAT cubes on demand instead of caching")
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = TRAINING[args.dataset]
    epochs = defaults["epochs"] if args.epochs is None else args.epochs
    batch_size = defaults["batch_size"] if args.batch_size is None else args.batch_size
    if epochs < 1 or batch_size < 1 or args.num_workers < 0 or args.lr <= 0 or args.seed < 0:
        raise ValueError("Invalid training settings")
    if args.resume and args.num_workers != 0:
        raise ValueError("Exact resumed data augmentation requires --num-workers 0")
    if args.resume and args.resume.resolve() != (args.output / "last.pth").resolve():
        raise ValueError("Resume requires last.pth in the original --output directory")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Run directory is not empty; choose a new --output or use --resume")
    device = resolve_device(args.device)
    seed_everything(args.seed)
    identity = validate_dataset(args.data_root, args.dataset, verify_hashes=True)
    model = initialize_model(PPCFormer(WAVELENGTHS[args.dataset]), args.seed)
    dataset = SpectralDataset(args.data_root, args.dataset, "train", cache=not args.no_cache)
    validation = SpectralDataset(args.data_root, args.dataset, "validation", cache=not args.no_cache)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=0,
                            pin_memory=device.type == "cuda")
    source_root = Path(__file__).resolve().parent
    source_files = [source_root / "train.py"] + sorted((source_root / "ppcformer").glob("*.py"))
    config = {
        "dataset": args.dataset, "seed": args.seed, "epochs": epochs, "batch_size": batch_size,
        "lr": args.lr, "num_workers": args.num_workers, "model_config": asdict(model.config),
        "training": defaults, "data_identity": identity,
        "initialization": "module_name_v2", "precision": "float32",
        "source_sha256": {p.relative_to(source_root).as_posix(): file_sha256(p) for p in source_files},
        "validation_schedule": list(validation_schedule(epochs)),
    }
    # JSON canonicalization makes tuple/list metadata portable across serializers.
    config = json.loads(json.dumps(config))
    payload = load_payload(args.resume) if args.resume else None
    if payload is not None:
        if payload.get("run_config") != config:
            raise ValueError("Resume configuration/code/data differ from this run")
        record_path = args.output / "run.json"
        if not record_path.is_file() or json.loads(record_path.read_text(encoding="utf-8"))["config"] != config:
            raise ValueError("Resume into the original --output directory, retaining run.json and best_val.pth")
        if payload.get("best_validation") and not (args.output / "best_val.pth").is_file():
            raise ValueError("Existing best_val.pth is required to preserve model selection")
        incumbent = args.output / "best_val.pth"
        if incumbent.is_file():
            best_payload = load_payload(incumbent)
            if (best_payload.get("run_config") != config
                    or best_payload.get("best_validation") != payload.get("best_validation")):
                raise ValueError("best_val.pth and last.pth are not from the same completed training state")
        load_weights(model, payload, args.dataset)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(loader), eta_min=1e-6)
    start, best = 1, None
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start, best = payload["epoch"] + 1, payload["best_validation"]
        restore_rng(payload["rng"], generator)
    stop = epochs if args.stop_after_epoch is None else min(args.stop_after_epoch, epochs)
    if stop < start:
        raise ValueError("The requested stopping epoch is before the next training epoch")
    args.output.mkdir(parents=True, exist_ok=True)
    if payload is None:
        write_json(args.output / "run.json", {"config": config, "environment": runtime_info(device)})
    print("PPCFormer | %s | seed=%d | params=%d | device=%s" %
          (args.dataset, args.seed, sum(p.numel() for p in model.parameters()), device), flush=True)
    for epoch in range(start, stop + 1):
        model.train()
        total_loss, samples = 0.0, 0
        progress = tqdm(loader, desc="Epoch %d/%d" % (epoch, epochs))
        for raw, sparse, target in progress:
            raw, sparse, target = (x.to(device, non_blocking=True) for x in (raw, sparse, target))
            optimizer.zero_grad(set_to_none=True)
            loss = charbonnier_loss(model(raw, sparse), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach())
            samples += raw.shape[0]
            progress.set_postfix(loss_per_sample=total_loss / samples)
        validation_metrics, improved = None, False
        if should_validate(epoch, epochs):
            validation_metrics, _ = evaluate(model, val_loader, device)
            eligible = epoch >= validation_schedule(epochs)[1]
            if eligible and (best is None or validation_metrics["psnr"] > best["psnr"]):
                best = dict(validation_metrics, epoch=epoch)
                improved = True
            print("Validation:", validation_metrics, "best:", best, flush=True)
        state = {
            "format_version": 1, "model": "PPCFormer", "dataset": args.dataset,
            "epoch": epoch, "run_config": config, "model_config": asdict(model.config),
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": capture_rng(generator),
            "best_validation": best,
        }
        if improved:
            save_checkpoint(args.output / "best_val.pth", dict(state, selected_checkpoint="best_validation"))
        save_checkpoint(args.output / "last.pth", state)
        with (args.output / "train.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"epoch": epoch, "loss_per_sample": total_loss / samples,
                                  "lr": optimizer.param_groups[0]["lr"], "validation": validation_metrics,
                                  "best_validation": best}, allow_nan=False) + "\n")
    print("Training stopped at epoch %d. Best validation: %s" % (stop, best))


if __name__ == "__main__":
    main()

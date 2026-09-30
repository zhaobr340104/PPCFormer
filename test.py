"""Evaluate one final-model checkpoint on full held-out images."""

import argparse
from pathlib import Path

from scipy.io import savemat
from torch.utils.data import DataLoader

from ppcformer import PPCFormer
from ppcformer.data import SpectralDataset, WAVELENGTHS, validate_dataset
from ppcformer.utils import (
    evaluate, file_sha256, load_payload, load_weights, resolve_device,
    runtime_info, seed_everything, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(WAVELENGTHS), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-cubes", action="store_true")
    parser.add_argument("--trusted-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    device = resolve_device(args.device)
    seed_everything(0)
    identity = validate_dataset(args.data_root, args.dataset, verify_hashes=True)
    model = PPCFormer(WAVELENGTHS[args.dataset])
    load_weights(model, load_payload(args.checkpoint, args.trusted_checkpoint), args.dataset)
    model.to(device).eval()
    data = SpectralDataset(args.data_root, args.dataset, args.split)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=True)

    def report(row, prediction):
        print(row, flush=True)
        if args.save_cubes:
            cube = prediction[0].permute(1, 2, 0).cpu().numpy()
            savemat(str(args.output / row["file"]), {"reconstruction": cube, "wavelengths_nm": WAVELENGTHS[args.dataset]}, do_compression=True)

    mean, rows = evaluate(model, loader, device, report)
    write_json(args.output / "metrics.json", {
        "dataset": args.dataset, "split": args.split, "checkpoint_sha256": file_sha256(args.checkpoint),
        "environment": runtime_info(device), "data_identity": identity,
        "per_image": rows, "mean": mean, "aggregation": "equal-weight mean of per-image scores",
    })
    print("Mean:", mean)


if __name__ == "__main__":
    main()

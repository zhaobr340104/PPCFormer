"""CAVE/Chikusei preparation and loading. Run python -m ppcformer.data --help."""

from io import BytesIO
import json
from pathlib import Path
import random
import re
import shutil
import tempfile
from zipfile import ZipFile

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset

from .model import PATTERN
from .utils import file_sha256, write_json


WAVELENGTHS = {
    "Chikusei": (
        651.59, 708.36, 770.29, 858.03, 630.95, 682.56, 749.65, 842.54,
        414.19, 465.80, 532.89, 584.50, 398.71, 434.84, 501.93, 558.70,
    ),
    "CAVE": tuple(float(x) for x in range(400, 701, 20)),
}
TRAINING = {
    "Chikusei": dict(epochs=4000, batch_size=8, patch_size=160, samples_per_epoch=90),
    "CAVE": dict(epochs=3500, batch_size=32, patch_size=128, samples_per_epoch=128),
}


CAVE_TRAIN = (
    "balloons", "beads", "cd", "chart_and_stuffed_toy", "clay", "cloth",
    "egyptian_statue", "face", "fake_and_real_beers", "fake_and_real_food",
    "fake_and_real_lemon_slices", "fake_and_real_lemons", "fake_and_real_peppers",
    "fake_and_real_strawberries", "fake_and_real_sushi", "fake_and_real_tomatoes",
    "feathers", "flowers", "glass_tiles", "hairs", "jelly_beans", "oil_painting",
    "paints", "photo_and_face", "pompoms", "real_and_fake_apples",
)
CAVE_VALIDATION = ("real_and_fake_peppers", "sponges", "stuffed_toys")
CAVE_TEST = ("superballs", "thread_spools", "watercolors")
CHIKUSEI_BANDS = (56, 67, 79, 96, 52, 62, 75, 93, 10, 20, 33, 43, 7, 14, 27, 38)


def expected_records(dataset):
    if dataset == "CAVE":
        return {split: [dict(file=scene + "_16.mat", scene_id=scene, height=512, width=512)
                        for scene in scenes]
                for split, scenes in (("train", CAVE_TRAIN), ("validation", CAVE_VALIDATION), ("test", CAVE_TEST))}
    if dataset != "Chikusei":
        raise ValueError("Unsupported dataset: " + dataset)
    result = {"train": [dict(file="train_scene_001_16.mat", scene_id="chikusei_20140729",
                            top=64, left=60, height=1048, width=2192)]}
    for split, top in (("validation", 1272), ("test", 1932)):
        result[split] = [dict(file="%s_%03d_16.mat" % (split, i + 1), scene_id="chikusei_20140729",
                             top=top, left=left, height=500, width=500)
                         for i, left in enumerate((60, 624, 1188, 1752))]
    return result


def load_cube(path):
    payload = loadmat(str(path))
    if "crop_gt" in payload:
        cube = payload["crop_gt"]["cube"][0, 0]
    elif "cube" in payload:
        cube = payload["cube"]
    else:
        raise ValueError("Expected crop_gt.cube or cube in " + str(path))
    cube = np.array(cube, dtype=np.float32, order="C", copy=True)
    if cube.ndim != 3 or cube.shape[2] != 16 or min(cube.shape[:2]) < 4:
        raise ValueError("Expected an H x W x 16 cube in " + str(path))
    if not np.isfinite(cube).all():
        raise ValueError("Nonfinite cube values in " + str(path))
    return cube


def mosaic_from_cube(cube, pattern=PATTERN):
    sparse = np.zeros_like(cube)
    for phase, band in enumerate(pattern):
        sparse[phase // 4::4, phase % 4::4, band] = cube[phase // 4::4, phase % 4::4, band]
    raw = sparse.sum(axis=2, keepdims=True)
    return tuple(np.ascontiguousarray(x.transpose(2, 0, 1)) for x in (raw, sparse, cube))


def augment_cube(cube, rotations, vertical, horizontal):
    if rotations:
        cube = np.rot90(cube, rotations, axes=(0, 1))
    if horizontal:
        cube = cube[:, ::-1, :]
    if vertical:
        cube = cube[::-1, :, :]
    return np.ascontiguousarray(cube)


class SpectralDataset(Dataset):
    def __init__(self, root, dataset, split, cache=True):
        self.files = sorted((Path(root) / split).glob("*.mat"))
        if not self.files:
            raise FileNotFoundError("No MAT files under " + str(Path(root) / split))
        self.training = split == "train"
        self.settings = TRAINING[dataset]
        self.cubes = [load_cube(path) for path in self.files] if cache else None
        if self.cubes:
            for cube in self.cubes:
                cube.setflags(write=False)

    def __len__(self):
        return self.settings["samples_per_epoch"] if self.training else len(self.files)

    def __getitem__(self, index):
        index %= len(self.files)
        cube = self.cubes[index] if self.cubes is not None else load_cube(self.files[index])
        if self.training:
            size = self.settings["patch_size"]
            if size > min(cube.shape[:2]):
                raise ValueError("Training patch exceeds source image dimensions")
            top = random.randrange(0, cube.shape[0] - size + 1, 4)
            left = random.randrange(0, cube.shape[1] - size + 1, 4)
            cube = cube[top:top + size, left:left + size].copy()
            rotations = random.randint(0, 3)
            vertical, horizontal = random.randint(0, 1), random.randint(0, 1)
            cube = augment_cube(cube, rotations, vertical, horizontal)
        return mosaic_from_cube(cube)


def validate_dataset(root, dataset, verify_hashes=False):
    """Require the fixed paper split, wavelength order, and stored-value scaling."""
    root = Path(root)
    expected = expected_records(dataset)
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_name") != {"CAVE": "CAVE-spatial-v1", "Chikusei": "Chikusei-spatial-v2"}[dataset]:
        raise ValueError("Dataset protocol does not match " + dataset)
    wavelengths = protocol.get("spectral_conversion", {}).get("source_band_centers_nm")
    if wavelengths is None or not np.allclose(wavelengths, WAVELENGTHS[dataset], rtol=0, atol=1e-8):
        raise ValueError("Dataset wavelengths do not match the paper protocol")
    if protocol.get("normalization", {}).get("loader_mode") != "none":
        raise ValueError("Expected stored-value loading without per-crop normalization")
    identity = {"protocol_sha256": file_sha256(protocol_path), "splits": {}}
    artifacts = {x["path"]: x for x in protocol.get("artifacts", [])}
    for split, records in expected.items():
        manifest = root / (split + "_manifest.json")
        if json.loads(manifest.read_text(encoding="utf-8")) != records:
            raise ValueError("Manifest differs from the paper split: " + split)
        files = sorted((root / split).glob("*.mat"))
        if [p.name for p in files] != sorted(x["file"] for x in records):
            raise ValueError("MAT filenames do not match the manifest: " + split)
        file_records = []
        for path in files:
            row = {"name": path.name, "bytes": path.stat().st_size}
            if verify_hashes:
                row["sha256"] = file_sha256(path)
                recorded = artifacts.get(split + "/" + path.name, {})
                if recorded.get("sha256") and row["sha256"] != recorded["sha256"]:
                    raise ValueError("Dataset file hash mismatch: " + str(path))
            file_records.append(row)
        identity["splits"][split] = {"manifest_sha256": file_sha256(manifest), "files": file_records}
    return identity


def save_cube(path, cube, wavelengths, norm_factor=1.0):
    record = np.empty((1, 1), dtype=[("bands", "O"), ("cube", "O"), ("norm_factor", "O")])
    record["bands"][0, 0] = np.asarray(wavelengths, dtype=np.float64).reshape(1, -1)
    record["cube"][0, 0] = np.ascontiguousarray(cube, dtype=np.float32)
    record["norm_factor"][0, 0] = np.asarray([[norm_factor]], dtype=np.float64)
    savemat(str(path), {"crop_gt": record}, do_compression=True, long_field_names=True, oned_as="row")


def collect_cave_members(archive):
    scenes = {}
    for name in archive.namelist():
        parts = name.split("/")
        match = re.search(r"_(\d{2})\.png$", name, re.I)
        if len(parts) < 3 or not parts[0].endswith("_ms") or match is None:
            continue
        scene, band = parts[0][:-3], int(match.group(1)) - 1
        mapping = scenes.setdefault(scene, {})
        if band in mapping:
            raise ValueError("Duplicate CAVE band: " + name)
        mapping[band] = name
    if set(scenes) != set(CAVE_TRAIN + CAVE_VALIDATION + CAVE_TEST):
        raise ValueError("CAVE archive must contain the expected 32 scenes")
    if any(sorted(bands) != list(range(31)) for bands in scenes.values()):
        raise ValueError("Every CAVE scene must contain exactly 31 numbered bands")
    return scenes


def decode_cave_png(content):
    if content[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Expected PNG data")
    bit_depth = content[24]
    with Image.open(BytesIO(content)) as image:
        array = np.asarray(image).copy()
    if array.ndim == 3:
        if array.shape[2] != 4 or bit_depth != 8:
            raise ValueError("Expected monochrome or grayscale RGBA CAVE bands")
        if not (np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 0], array[..., 2]) and np.all(array[..., 3] == 255)):
            raise ValueError("Non-grayscale or non-opaque RGBA band")
        array = array[..., 0]
    if array.ndim != 2 or bit_depth not in (8, 16):
        raise ValueError("Unsupported CAVE PNG representation")
    return np.asarray(array, dtype=np.float32) / np.float32((1 << bit_depth) - 1)


def cave_cube(archive, members):
    bands = [decode_cave_png(archive.read(members[i])) for i in range(0, 31, 2)]
    if any(b.shape != (512, 512) for b in bands):
        raise ValueError("CAVE source bands must be 512 x 512")
    return np.ascontiguousarray(np.stack(bands, axis=-1), dtype=np.float32)


def validate_chikusei_header(path):
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"wavelength\s*=\s*\{(.*?)\}", text, re.I | re.S)
    if match is None:
        raise ValueError("ENVI header is missing wavelength metadata")
    values = np.asarray([float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", match.group(1))])
    if values.size != 128:
        raise ValueError("Expected 128 source wavelengths")
    if values.max() < 10:
        values *= 1000
    if not np.allclose(values[list(CHIKUSEI_BANDS)], WAVELENGTHS["Chikusei"], rtol=0, atol=1e-8):
        raise ValueError("Source wavelength centers do not match the paper band selection")


def chikusei_cube(source, record):
    order = np.argsort(CHIKUSEI_BANDS)
    sorted_bands = np.asarray(CHIKUSEI_BANDS)[order]
    inverse = np.argsort(order)
    h, w = record["height"], record["width"]
    cube = np.empty((h, w, 16), dtype=np.float32)
    for start in range(0, h, 32):
        end = min(h, start + 32)
        raw = np.asarray(source[sorted_bands, record["left"]:record["left"] + w,
                                record["top"] + start:record["top"] + end], dtype=np.float32)
        raw = raw[inverse].transpose(2, 1, 0)
        if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError("Expected finite integer DN values in the Chikusei source")
        if np.any(np.all(raw == 0, axis=2)):
            raise ValueError("Selected Chikusei region contains an all-zero spectrum")
        cube[start:end] = raw / np.float32(4096.0)
    return cube


def prepare_dataset(dataset, source, output, header=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError("Refusing to overwrite " + str(output))
    if dataset == "Chikusei":
        if header is None:
            raise ValueError("Chikusei requires the official ENVI --header")
        validate_chikusei_header(header)
    records = expected_records(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=str(output.parent))).resolve()
    try:
        for split, items in records.items():
            (temporary / split).mkdir()
            write_json(temporary / (split + "_manifest.json"), items)
        source_hash = file_sha256(source)
        artifacts = []

        def emit(split, record, cube):
            path = temporary / split / record["file"]
            print("Preparing " + split + "/" + record["file"], flush=True)
            factor = 3.3598335963930688 if dataset == "Chikusei" else 1.0
            save_cube(path, cube, WAVELENGTHS[dataset], factor)
            artifacts.append(dict(path=split + "/" + path.name, bytes=path.stat().st_size, sha256=file_sha256(path)))

        if dataset == "CAVE":
            with ZipFile(source) as archive:
                scenes = collect_cave_members(archive)
                for split, items in records.items():
                    for record in items:
                        emit(split, record, cave_cube(archive, scenes[record["scene_id"]]))
        else:
            with h5py.File(source, "r") as handle:
                data = handle["chikusei"]
                if data.shape != (128, 2335, 2517) or not np.issubdtype(data.dtype, np.floating):
                    raise ValueError("Unexpected Chikusei HDF5 source layout")
                for split, items in records.items():
                    for record in items:
                        emit(split, record, chikusei_cube(data, record))
        protocol = {
            "protocol_name": "CAVE-spatial-v1" if dataset == "CAVE" else "Chikusei-spatial-v2",
            "protocol_version": 1 if dataset == "CAVE" else 2,
            "source": {"file": source.name, "sha256": source_hash},
            "spectral_conversion": {
                "source_band_centers_nm": list(WAVELENGTHS[dataset]),
                "zero_based_source_band_indices_in_channel_order": list(range(0, 31, 2)) if dataset == "CAVE" else list(CHIKUSEI_BANDS),
            },
            "normalization": {"loader_mode": "none", "metric_data_range": 1.0,
                              "stored_value_scaling": "PNG / (2**bit_depth - 1)" if dataset == "CAVE" else "DN / 4096"},
            "split_records": records,
            "spatial_guard_pixels": 0 if dataset == "CAVE" else 160,
            "artifacts": artifacts,
        }
        if header is not None:
            protocol["source"]["header_sha256"] = file_sha256(header)
        write_json(temporary / "protocol.json", protocol)
        validate_dataset(temporary, dataset, verify_hashes=True)
        temporary.rename(output)
    finally:
        # Only remove the owned staging directory, never source or output data.
        if temporary.exists() and temporary.parent == output.parent and temporary.name.startswith(output.name + ".tmp-"):
            shutil.rmtree(temporary)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(WAVELENGTHS), required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--header", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        validate_dataset(args.output, args.dataset, verify_hashes=True)
        print("Dataset verification passed")
    else:
        if args.source is None:
            parser.error("--source is required when preparing data")
        prepare_dataset(args.dataset, args.source, args.output, args.header)


if __name__ == "__main__":
    main()

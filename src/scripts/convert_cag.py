import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm

INPUT_IMAGE_DIR = Path("/workspaces/NoPoSplat/datasets/cag")
OUTPUT_DIR = Path("/workspaces/NoPoSplat/datasets/cag_chunks")
OUTPUT_DIR.mkdir(exist_ok=True)

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)


def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    keys = [i.stem for i in INPUT_IMAGE_DIR.glob("*")]

    keys.sort()
    return keys


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[str, UInt8[Tensor, "..."]]:
    """Load JPG/PNG images as raw bytes (do not decode)."""
    return {path.stem: load_raw(path) for path in example_path.iterdir() if path.suffix == ".png"}


class Metadata(TypedDict):
    patient_id: str
    study_date: str
    timestamps: Int[Tensor, " frame"]
    cameras: Float[Tensor, "frame entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def load_image_and_metadata(file_path: Path) -> Metadata:
    with open(file_path, 'r') as file:
        data = json.load(file)

    patient_id = data["patient_id"]
    study_date = data["study_date"]

    cameras = []
    timestamps = []
    images = []
    
    for i, frame in enumerate(data["frames"]):
        timestamps.append(i)

        intrinsic = np.array(frame["intrinsic_matrix"], dtype=np.float32)
        extrinsic = np.eye(4, dtype=np.float32)  # Placeholder: Adjust based on your requirements.

        w2c = extrinsic[:3, :].flatten()
        camera = np.concatenate([intrinsic, w2c])
        cameras.append(camera)
        image = load_raw(image_dir / f"frame_{i:0>5}.png")
        images.append(image)

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "patient_id": patient_id,
        "study_date": study_date,
        "timestamps": timestamps,
        "cameras": cameras,
        "images": images,
    }


if __name__ == "__main__":
    # for stage in ("train", "test"):
    for stage in ["train"]:
        keys = get_example_keys(stage)
        print("number of keys:", len(keys))

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
            )
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for key in keys:
            image_dir = INPUT_IMAGE_DIR / key / 'images'
            metadata_file = INPUT_IMAGE_DIR / key / 'transforms.json'
            num_bytes = get_size(image_dir)

            if not image_dir.exists() or not metadata_file.exists():
                print(f"Skipping {key} because it is missing.")
                continue

            # Read images and metadata.
            example = load_image_and_metadata(metadata_file)

            # Add the key to the example.
            example["key"] = key

            print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()

        # generate index
        print("Generate key:torch index...")
        index = {}
        stage_path = OUTPUT_DIR / stage
        stage_path.mkdir(exist_ok=True, parents=True)
        for chunk_path in tqdm(list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage_path))
        with (stage_path / "index.json").open("w") as f:
            json.dump(index, f)

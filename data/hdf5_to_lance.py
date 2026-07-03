#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from stable_worldmodel.data.formats.lance import LanceWriter


META_KEYS = {"ep_len", "ep_offset", "episode_idx", "step_idx"}


def _episode_slices(h5: h5py.File):
    if "ep_len" in h5 and "ep_offset" in h5:
        lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        for start, length in zip(offsets, lengths, strict=True):
            yield int(start), int(length)
        return

    if "episode_idx" in h5:
        episode_idx = np.asarray(h5["episode_idx"][:], dtype=np.int64)
        if episode_idx.size == 0:
            return
        change = np.flatnonzero(np.diff(episode_idx) != 0) + 1
        offsets = np.concatenate(([0], change))
        bounds = np.concatenate((offsets, [episode_idx.size]))
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            yield int(start), int(end - start)
        return

    raise ValueError(
        "HDF5 dataset must contain either ep_len+ep_offset or episode_idx"
    )


def _data_keys(h5: h5py.File) -> list[str]:
    keys = []
    for key, value in h5.items():
        if key in META_KEYS:
            continue
        if not isinstance(value, h5py.Dataset):
            continue
        if value.shape and value.shape[0] > 0:
            keys.append(key)
    if not keys:
        raise ValueError("No episode data arrays found in HDF5 file")
    return keys


def _clean_episode_values(values):
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return list(arr)


def convert_hdf5_to_lance(
    source: str | Path,
    dest: str | Path,
    *,
    mode: str = "overwrite",
    limit_episodes: int | None = None,
) -> None:
    source = Path(source)
    dest = Path(dest)

    with h5py.File(source, "r") as h5, LanceWriter(dest, mode=mode) as writer:
        keys = _data_keys(h5)

        def episodes():
            for ep_idx, (start, length) in enumerate(_episode_slices(h5)):
                if limit_episodes is not None and ep_idx >= limit_episodes:
                    break
                yield {
                    key: _clean_episode_values(h5[key][start : start + length])
                    for key in keys
                }

        writer.write_episodes(episodes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a flat LeWM HDF5 dataset into Lance format."
    )
    parser.add_argument("source", help="Path to the source .h5 file")
    parser.add_argument("dest", help="Path to the destination .lance directory")
    parser.add_argument(
        "--mode",
        default="overwrite",
        choices=("append", "overwrite", "error"),
        help="Writer mode for the destination Lance dataset",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Convert only the first N episodes for validation",
    )
    args = parser.parse_args()
    convert_hdf5_to_lance(
        args.source,
        args.dest,
        mode=args.mode,
        limit_episodes=args.limit_episodes,
    )


if __name__ == "__main__":
    main()

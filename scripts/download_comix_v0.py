#!/usr/bin/env python3
"""Download SHA-pinned Comix v0 shards with the Python standard library."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def is_valid(path, descriptor):
    return path.is_file() and path.stat().st_size == descriptor["bytes"] and digest(path) == descriptor["sha256"]


def download_shards(manifest_path, output, selected=None):
    manifest = json.loads(manifest_path.read_text())
    descriptors = {item["file"]: item for item in manifest["shards"]}
    names = selected or list(descriptors)
    unknown = [name for name in names if name not in descriptors]
    if unknown:
        raise ValueError(f"unlisted shards: {', '.join(unknown)}")

    output.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name in names:
        descriptor = descriptors[name]
        destination = output / name
        if is_valid(destination, descriptor):
            continue
        partial = output / f"{name}.partial"
        with urlopen(f"{manifest['downloadBaseURL']}/{name}") as source, partial.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        if not is_valid(partial, descriptor):
            partial.unlink(missing_ok=True)
            raise ValueError(f"downloaded shard failed verification: {name}")
        os.replace(partial, destination)
        downloaded.append(name)
    return downloaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shard", action="append", dest="shards")
    args = parser.parse_args()
    print(json.dumps({"downloaded": download_shards(args.manifest, args.output, args.shards)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reuse Charm component content hashes while their file metadata is unchanged."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time


CACHE_SCHEMA = 2
TIMESTAMP_SETTLE_NS = 2_000_000_000


class ChangedDuringRead(Exception):
    pass


def signature(info):
    # ctime catches same-size edits even when a tool restores the old mtime.
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def snapshot(root):
    records = []
    pending = [os.fsencode(root)]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                    continue
                info = entry.stat(follow_symlinks=False)
                if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                    continue
                relative = b"./" + os.path.relpath(entry.path, os.fsencode(root))
                target = None
                target_info = None
                if stat.S_ISLNK(info.st_mode):
                    target = os.readlink(entry.path)
                    try:
                        followed = entry.stat(follow_symlinks=True)
                    except (FileNotFoundError, NotADirectoryError):
                        followed = None
                    if followed is not None and stat.S_ISREG(followed.st_mode):
                        target_info = signature(followed)
                records.append((relative, signature(info), target, target_info))
    records.sort(key=lambda item: item[0])  # Match LC_ALL=C sort -z.
    metadata = [(os.fsdecode(path), info, os.fsdecode(target) if target is not None
                 else None, target_info) for path, info, target, target_info in records]
    encoded = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True).encode()
    return records, hashlib.sha256(encoded).hexdigest()


def hash_file(path, expected):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        if signature(os.fstat(stream.fileno())) != expected:
            raise ChangedDuringRead()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if signature(os.fstat(stream.fileno())) != expected:
            raise ChangedDuringRead()
    return digest.hexdigest().encode() + b"\n"


def content_fingerprint(root, records):
    digest = hashlib.sha256()
    for relative, info, target, target_info in records:
        digest.update(relative + b"\0")
        path = os.path.join(os.fsencode(root), relative)
        # sha256sum prefixes its output for filenames requiring escape sequences.
        escaped = b"\\" if b"\\" in relative or b"\n" in relative else b""
        if target is not None:
            # Shell command substitution strips trailing newlines from readlink.
            digest.update(b"symlink:" + target.rstrip(b"\n") + b"\n")
            if target_info is not None:
                digest.update(escaped + hash_file(path, target_info))
        else:
            digest.update(escaped + hash_file(path, info))
    return digest.hexdigest()


def fingerprint(root, cache_root=None, refresh=False):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Charm component is not a directory: " + str(root))
    for attempt in range(3):
        try:
            records, metadata = snapshot(root)
            cache_file = None
            # Fast edits can share a filesystem clock tick, including ctime.
            # Never publish/reuse a digest until all input timestamps have settled.
            cutoff = time.time_ns() - TIMESTAMP_SETTLE_NS
            settled = all(max(info[-2:]) < cutoff
                          and (target_info is None or max(target_info[-2:]) < cutoff)
                          for _, info, _, target_info in records)
            if cache_root is not None and settled:
                identity = hashlib.sha256(os.fsencode(root)).hexdigest()
                cache_file = Path(cache_root) / (identity + "-v2-" + metadata + ".json")
                if not refresh:
                    try:
                        cached = json.loads(cache_file.read_text(encoding="ascii"))
                    except (OSError, ValueError):
                        cached = None
                    if isinstance(cached, dict):
                        result = cached.get("fingerprint")
                        if (cached.get("schema") == CACHE_SCHEMA and cached.get("root") == str(root)
                                and cached.get("metadata") == metadata
                                and isinstance(result, str) and len(result) == 64
                                and all(c in "0123456789abcdef" for c in result)):
                            return result
            result = content_fingerprint(root, records)
            if snapshot(root)[1] != metadata:
                raise ChangedDuringRead()
        except (ChangedDuringRead, FileNotFoundError):
            continue
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Immutable records: concurrent writers never overwrite one another.
                with cache_file.open("x", encoding="ascii") as stream:
                    json.dump({"schema": CACHE_SCHEMA, "root": str(root), "metadata": metadata,
                               "fingerprint": result}, stream, sort_keys=True)
                    stream.write("\n")
            except FileExistsError:
                pass
        return result
    raise RuntimeError("Charm component changed during fingerprinting: " + str(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--cache-root")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(fingerprint(args.root, args.cache_root, args.refresh))


if __name__ == "__main__":
    main()

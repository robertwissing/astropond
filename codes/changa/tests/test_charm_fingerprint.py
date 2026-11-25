#!/usr/bin/env python3
"""Dependency invalidation and cache checks; temporary artifacts are retained."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch


CHANGA = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("charm_fingerprint", CHANGA / "charm_fingerprint.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


LEGACY = r'''
set -euo pipefail
source "$2"
cd "$1"
find . \( -type f -o -type l \) -print0 | LC_ALL=C sort -z |
while IFS= read -r -d '' path; do
    printf '%s\0' "$path"
    if [ -L "$path" ]; then
        printf 'symlink:%s\n' "$(readlink "$path")"
        if [ -f "$path" ]; then build_sha256_file "$path"; fi
    else
        build_sha256_file "$path"
    fi
done | build_sha256_stream
'''


def create(path, data):
    with path.open("xb") as stream:
        stream.write(data)


class CharmFingerprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite_artifacts = Path(tempfile.mkdtemp(prefix="charm-fingerprint-suite."))
        for method in dir(cls):
            if method.startswith("test_"):
                root = cls.suite_artifacts / method / "include"
                root.mkdir(parents=True)
                create(root / "marker.h", b"original\n")
        # Establish settled inputs once for the cache-hit behavior tests.
        time.sleep(helper.TIMESTAMP_SETTLE_NS / 1e9 + 0.1)

    def setUp(self):
        self.artifacts = self.suite_artifacts / self._testMethodName
        self.root = self.artifacts / "include"
        self.cache = self.artifacts / "cache"
        self.header = self.root / "marker.h"

    def digest(self, **kwargs):
        return helper.fingerprint(self.root, self.cache, **kwargs)

    def test_matches_existing_content_fingerprint(self):
        nested = self.root / "nested"
        nested.mkdir()
        create(nested / "space name.h", b"more input\n")
        create(nested / "back\\slash.h", b"escaped filename\n")
        create(nested / "line\nbreak.h", b"another escaped filename\n")
        (self.root / "linked.h").symlink_to("marker.h")
        (self.root / "directory-link").symlink_to("nested", target_is_directory=True)
        legacy = subprocess.check_output(
            ["bash", "-c", LEGACY, "legacy", str(self.root),
             str(CHANGA.parent / "build_common.sh")], text=True).strip()
        self.assertEqual(self.digest(), legacy)

    def test_unchanged_tree_reads_no_file_contents(self):
        before = self.digest()
        records = list(self.cache.iterdir())
        with patch.object(helper, "hash_file", side_effect=AssertionError("content reread")):
            self.assertEqual(self.digest(), before)
        self.assertEqual(list(self.cache.iterdir()), records)

    def test_same_size_edit_with_restored_mtime_changes_digest(self):
        before = self.digest()
        original = self.header.stat()
        with self.header.open("r+b") as stream:
            stream.write(b"modified")
        os.utime(self.header, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.assertEqual(self.header.stat().st_size, original.st_size)
        self.assertEqual(self.header.stat().st_mtime_ns, original.st_mtime_ns)
        self.assertNotEqual(self.digest(), before)

    def test_added_file_changes_digest(self):
        before = self.digest()
        create(self.root / "new.h", b"new header\n")
        self.assertNotEqual(self.digest(), before)

    def test_external_symlink_target_edit_changes_digest(self):
        target = self.artifacts / "external.h"
        create(target, b"outside include\n")
        (self.root / "external-link.h").symlink_to(target)
        before = self.digest()
        with target.open("ab") as stream:
            stream.write(b"changed\n")
        self.assertNotEqual(self.digest(), before)

    def test_previously_dangling_symlink_target_is_detected(self):
        target = self.artifacts / "future.h"
        (self.root / "future-link.h").symlink_to(target)
        before = self.digest()
        create(target, b"now exists\n")
        self.assertNotEqual(self.digest(), before)

    def test_directory_alias_reuses_same_cache(self):
        before = self.digest()
        alias = self.artifacts / "tmp"
        alias.symlink_to(self.root, target_is_directory=True)
        with patch.object(helper, "hash_file", side_effect=AssertionError("duplicate scan")):
            self.assertEqual(helper.fingerprint(alias, self.cache), before)
        self.assertEqual(len(list(self.cache.iterdir())), 1)

    def test_concurrent_cache_population(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: self.digest(), range(8)))
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(len(list(self.cache.iterdir())), 1)
        with patch.object(helper, "hash_file", side_effect=AssertionError("invalid cache")):
            self.assertEqual(self.digest(), results[0])

    def test_refresh_verifies_contents_with_same_result(self):
        before = self.digest()
        with patch.object(helper, "hash_file", wraps=helper.hash_file) as content_read:
            self.assertEqual(self.digest(refresh=True), before)
        self.assertGreater(content_read.call_count, 0)

    def test_change_during_read_is_retried(self):
        actual_hash = helper.hash_file
        reads = []

        def edit_after_first_read(path, expected):
            result = actual_hash(path, expected)
            reads.append(path)
            if len(reads) == 1:
                with self.header.open("ab") as stream:
                    stream.write(b"edited during hash\n")
            return result

        with patch.object(helper, "hash_file", side_effect=edit_after_first_read):
            result = self.digest()
        self.assertGreater(len(reads), 1)
        self.assertEqual(result, self.digest(refresh=True))

    def test_recent_files_are_not_cached(self):
        with self.header.open("ab") as stream:
            stream.write(b"recent change\n")
        before = self.digest()
        self.assertFalse(self.cache.exists())
        with patch.object(helper, "hash_file", wraps=helper.hash_file) as content_read:
            self.assertEqual(self.digest(), before)
        self.assertGreater(content_read.call_count, 0)


if __name__ == "__main__":
    unittest.main()

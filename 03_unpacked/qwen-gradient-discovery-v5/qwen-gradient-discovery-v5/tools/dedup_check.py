#!/usr/bin/env python3
"""Audit textual duplicate content in 03_unpacked; preserve complete version artifacts.

Default policy: duplicates within a version family are allowed, cross-family
identical text is a conflict (exit 2); the program never removes files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def version_family(archive_stem: str) -> str:
    # Versions such as qwen-gradient-discovery-v2, -v3, -v3.1 share a family.
    return re.sub(r'-v\d+(?:\.\d+)*$', '', archive_stem, flags=re.IGNORECASE)


def check_duplicates(unpacked: Path) -> dict:
    unpacked = Path(unpacked)
    if not unpacked.is_dir():
        raise FileNotFoundError(f'No extracted archive directory: {unpacked}')
    groups = defaultdict(list)
    for path in unpacked.rglob('*'):
        if not path.is_file() or path.name == 'package_provenance.json':
            continue
        rel = path.relative_to(unpacked)
        if len(rel.parts) < 2:
            continue
        data = path.read_bytes()
        try:
            data.decode('utf-8')
        except UnicodeDecodeError:
            continue
        if b'\0' in data:
            continue
        digest = hashlib.sha256(data).hexdigest()
        groups[digest].append({'path': str(rel), 'archive': rel.parts[0],
                               'family': version_family(rel.parts[0]), 'bytes': len(data)})
    allowed = []
    cross = []
    for digest, files in sorted(groups.items()):
        if len(files) < 2: continue
        record = {'sha256': digest, 'files': files}
        (allowed if len({x['family'] for x in files}) == 1 else cross).append(record)
    return {'policy': 'preserve-independent-version-artifacts; report cross-family duplicates',
            'within_version_family_allowed': allowed,
            'cross_version_family_conflicts': cross,
            'n_allowed_hash_groups': len(allowed),
            'n_cross_family_conflicts': len(cross)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace-root', type=Path, required=True,
                    help='Workspace directly containing 03_unpacked/, e.g. /repo/qwentest')
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--report-only', action='store_true',
                    help='Exit 0 even when cross-family duplicates exist; no file is removed')
    args = ap.parse_args()
    result = check_duplicates(args.workspace_root / '03_unpacked')
    report = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.output:args.output.write_text(report, encoding='utf-8')
    print(report)
    if result['n_cross_family_conflicts'] and not args.report_only:
        raise SystemExit(2)

if __name__ == '__main__': main()

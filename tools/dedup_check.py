#!/usr/bin/env python3
"""Audit textual duplicate content in vendor; preserve complete version artifacts.

Default policy: duplicates within a version family are allowed, cross-family
identical text is a conflict (exit 2); the program never removes files.

A cross-family group may be allowed ONLY by an explicit entry in
`tools/dedup_allow.json`, keyed by the sha256 of the duplicated bytes and listing
every file in the group. An entry that no longer matches a duplicate group (the
content changed, or one copy was deleted) is reported as STALE and still fails:
an allowlist that rots silently is worse than none.
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


def load_allow(path: Path) -> dict:
    """Read the reviewed cross-family allowlist. Absent file = empty allowlist."""
    if not Path(path).is_file():
        return {'allowed_cross_family_groups': []}
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    groups = data.get('allowed_cross_family_groups', [])
    for g in groups:
        for field in ('sha256', 'reason', 'decided', 'files'):
            if not g.get(field):
                raise ValueError(f'Allowlist entry missing "{field}": {g}')
        if not re.fullmatch(r'[0-9a-f]{64}', g['sha256']):
            raise ValueError(f'Allowlist entry sha256 is not 64 hex chars: {g["sha256"]}')
    if len({g['sha256'] for g in groups}) != len(groups):
        raise ValueError('Allowlist has two entries with the same sha256')
    return data


def check_duplicates(unpacked: Path, allow_path: Path = None) -> dict:
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
    reviewed = []
    for digest, files in sorted(groups.items()):
        if len(files) < 2: continue
        record = {'sha256': digest, 'files': files}
        if len({x['family'] for x in files}) == 1:
            allowed.append(record)
        else:
            cross.append(record)
    # Split the cross-family groups into reviewed (allowlisted) and unexplained.
    conflicts, stale = [], []
    if allow_path is not None:
        entries = {g['sha256']: g for g in load_allow(allow_path)['allowed_cross_family_groups']}
        hit = set()
        for record in cross:
            entry = entries.get(record['sha256'])
            if entry is None:
                conflicts.append(record)
                continue
            hit.add(record['sha256'])
            actual = sorted(x['path'] for x in record['files'])
            claimed = sorted(entry['files'])
            # The entry must name exactly the group that exists; a partial match means
            # a copy was added or removed since the review, so the reason may not hold.
            if actual != claimed:
                conflicts.append({**record, 'allowlist_file_mismatch': {
                    'claimed': claimed, 'actual': actual}})
                continue
            reviewed.append({'sha256': record['sha256'], 'files': record['files'],
                             'reason': entry['reason'], 'decided': entry['decided']})
        for digest, entry in sorted(entries.items()):
            if digest not in hit:
                stale.append({'sha256': digest, 'reason': entry['reason'],
                              'files': entry['files'],
                              'problem': 'no duplicate group with these bytes exists any more'})
    else:
        conflicts = cross
    return {'policy': 'preserve-independent-version-artifacts; report cross-family duplicates',
            'within_version_family_allowed': allowed,
            'cross_family_reviewed': reviewed,
            'cross_version_family_conflicts': conflicts,
            'stale_allowlist_entries': stale,
            'n_allowed_hash_groups': len(allowed),
            'n_cross_family_reviewed': len(reviewed),
            'n_cross_family_conflicts': len(conflicts) + len(stale)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace-root', type=Path, required=True,
                    help='Workspace directly containing vendor/, e.g. /repo/neuron-closed')
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--report-only', action='store_true',
                    help='Exit 0 even when cross-family duplicates exist; no file is removed')
    ap.add_argument('--no-allowlist', action='store_true',
                    help='Ignore tools/dedup_allow.json and report every cross-family duplicate')
    args = ap.parse_args()
    allow = None if args.no_allowlist else args.workspace_root / 'tools' / 'dedup_allow.json'
    result = check_duplicates(args.workspace_root / 'vendor', allow)
    report = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.output:args.output.write_text(report, encoding='utf-8')
    print(report)
    if result['n_cross_family_conflicts'] and not args.report_only:
        raise SystemExit(2)

if __name__ == '__main__': main()

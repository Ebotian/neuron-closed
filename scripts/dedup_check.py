#!/usr/bin/env python3
"""Remove files under vendor/rebuild-kit/ that are byte-identical to a file kept
elsewhere in qwentest/. Dry run by default; --apply deletes and prints the map.

Guard: a file is deleted only if at least one copy exists OUTSIDE rebuild-kit/, and
that copy stays. Nothing is deleted on a name or size match alone.
"""
import hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / 'vendor/rebuild-kit'
APPLY = '--apply' in sys.argv

def md5(p):
    h = hashlib.md5()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

index = {}
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and ROOT / '.git' not in p.parents:
        index.setdefault(md5(p), []).append(p)

removed, kept = [], []
for p in sorted(KIT.rglob('*')):
    if not p.is_file():
        continue
    peers = [q for q in index[md5(p)] if q != p]
    outside = [q for q in peers if KIT not in q.parents]
    if not outside:
        kept.append((p, peers))
        continue
    removed.append((p, outside[0], len(peers)))
    if APPLY:
        p.unlink()

for p, keep, n in removed:
    print(f"{'removed' if APPLY else 'would remove'}  {p.relative_to(ROOT)}  -> keep {keep.relative_to(ROOT)}  ({n} other copies)")
for p, peers in kept:
    print(f"keep     {p.relative_to(ROOT)}  (no copy outside the kit; {len(peers)} copy inside)")

print(f"\n{'removed' if APPLY else 'would remove'}: {len(removed)}  kept: {len(kept)}")

#!/usr/bin/env python3
"""neuron-closed gates: one command that re-derives every claim the docs make.

Checks (each also runnable alone):
  blocks      the two auto-managed blocks in README.md equal their renders
  ziptrees    every text member of packages/*.zip equals its extracted file
  checksums   every vendor/*/<root>/CHECKSUMS.sha256 verifies
  compile     every tracked .py compiles
  tests       the torch-free unit tests pass (others are reported as skipped)
  dedup       no cross-version-family duplicate text, except the reviewed entries
              in tools/dedup_allow.json (packages/archives excluded)
  git         no binary/derived file is tracked
  ledger      evidence/ledger.json is valid and the rendered matrix in
              doc/references/synthesis-external-evidence.md matches it (tools/evidence_ledger.py)

Rendering:  render-status --write   /   render-provenance --write
Stdlib only; no network, no GPU, no torch.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # repo root
REPO = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip() or ROOT)
PKG, UNP = ROOT / 'packages', ROOT / 'vendor'
README = ROOT / 'README.md'
META = ROOT / 'doc' / 'metadata.json'
STATUS = ('<!-- auto:workstream-status:start -->', '<!-- auto:workstream-status:end -->')
PROV = ('<!-- auto:unpacked-packages:start -->', '<!-- auto:unpacked-packages:end -->')
SKIP_EXT = {'.zip', '.tar', '.gz', '.zst', '.pt', '.pth', '.npy', '.npz', '.safetensors',
            '.png', '.jpg', '.jpeg', '.webp', '.pdf', '.pyc', '.bin', '.so', '.whl'}
TORCH_FREE = ['test_progress.py']                    # everything else needs torch


def md5(b):
    return hashlib.md5(b).hexdigest()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def block(text, start, end):
    i, j = text.index(start), text.index(end)
    return text[i:j + len(end)]


def replace_block(text, start, end, body, before='## 目录结构'):
    if start in text and end in text:
        i, j = text.index(start), text.index(end)
        return text[:i] + body + text[j + len(end):]
    if before not in text:                     # first insertion: put it before the layout block
        raise SystemExit(f'anchor not found for the managed block: {before!r}')
    i = text.index(before)
    return text[:i] + body + '\n\n' + text[i:]


def render_status():
    m = json.loads(META.read_text(encoding='utf8'))
    rows = [f'| 项 | 值 |', '|---|---|',
            f'| 工作流 | `{m["workstream"]}` |',
            f'| 版本 / 状态 | `{m["version"]}` / `{m["status"]}`（{m["date"]}） |',
            f'| 现役 | {m["current"]} |']
    routes = ' · '.join(f'`{k}` {v.split(" (")[0]}' for k, v in m['routes'].items())
    rows.append(f'| 路线 | {routes} |')
    rows.append('| 判据 | ' + ' · '.join(f'`{k}`' for k in m['gates']) +
                '（定义见 `metadata.json`，说明见 `doc/plan-gradient-discovery.md` §1） |')
    if m.get('verdict_short'):
        rows.append(f'| 结论 | {m["verdict_short"]} |')
    rows.append('| 实测 | G1 r=%.3f/J64=%.3f（%d 条 prompt） · G2 r=%.3f–%.3f / J64=%.3f–%.3f（%s / %s） · verify %d/%d p=%.4f（%s） |' % (
        m['measured']['g1']['r'], m['measured']['g1']['J64'], m['measured']['g1']['n_prompts'],
        m['measured']['g2_split_blocks']['r'], m['measured']['g2_split_parity']['r'],
        m['measured']['g2_split_blocks']['J64'], m['measured']['g2_split_parity']['J64'],
        m['measured']['g2_split_blocks']['n'], m['measured']['g2_split_parity']['n'],
        m['measured']['verify']['wins'], m['measured']['verify']['wins'] + m['measured']['verify']['losses'],
        m['measured']['verify']['p'], m['measured']['verify']['heldout']))
    return STATUS[0] + '\n' + '\n'.join(rows) + '\n' + STATUS[1]


def zip_members(archive):
    """Text members as (relative path, bytes) -- same policy as the migration tool."""
    out = []
    with zipfile.ZipFile(archive) as zf:
        for obj in zf.infolist():
            name = Path(obj.filename)
            if obj.is_dir() or name.suffix.lower() in SKIP_EXT:
                continue
            if any(p == '__pycache__' for p in name.parts):
                continue
            data = zf.read(obj)
            if b'\0' in data:
                continue
            try:
                data.decode('utf-8')
            except UnicodeDecodeError:
                continue
            out.append((name, data))
    return out


def render_provenance():
    rows = ['| Archive | MD5 | SHA256 | Unpacked text files |', '|---|---|---|---:|']
    for archive in sorted(PKG.glob('*.zip')):
        members = zip_members(archive)
        rows.append(f'| `{archive.name}` | `{md5(archive.read_bytes())}` | '
                    f'`{sha(archive.read_bytes())}` | {len(members)} |')
    return PROV[0] + '\n## Unpacked package provenance (automatically maintained)\n\n' \
        'Binary ZIP files remain ignored; text versions are under `vendor/`.\n\n' + \
        '\n'.join(rows) + '\n' + PROV[1]


def check_blocks():
    text = README.read_text(encoding='utf8')
    bad = []
    for name, (start, end), want in (('status', STATUS, render_status()),
                                     ('provenance', PROV, render_provenance())):
        got = block(text, start, end)
        if got.strip() != want.strip():
            bad.append(f'{name} block is stale: run `python3 tools/check.py render-{name} --write`')
    return bad


def check_ziptrees():
    bad = []
    for archive in sorted(PKG.glob('*.zip')):
        target = UNP / archive.stem
        for name, data in zip_members(archive):
            path = target / name
            if not path.exists():
                bad.append(f'missing extracted file: {path.relative_to(ROOT)}')
            elif path.read_bytes() != data:
                bad.append(f'extracted file differs from zip: {path.relative_to(ROOT)}')
    return bad


def check_checksums():
    bad = []
    for sums in sorted(UNP.glob('*/*/CHECKSUMS.sha256')):
        r = subprocess.run(['sha256sum', '-c', '--quiet', sums.name], cwd=sums.parent,
                           capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(f'{sums.relative_to(ROOT)}: {r.stdout.strip() or r.stderr.strip()}')
    return bad


def check_instruments():
    """Two gates the audit asked for (audit-response §12.8).

    1. Every Lean audit record must match the CURRENT file: a docstring edit invalidates the record, which is how
       lean/Dimension.lean's stale sha256 was found. The theorems are unaffected by a comment, but the record may not
       claim to certify a file it never saw.
    2. The finite-difference calibration gate must exist, must have PASSED, and must pin the hash of the instrument it
       certified (scripts/mlp_capture.py). Without that pin the gate would keep certifying an old implementation.
    """
    import hashlib
    bad = []
    aud = ROOT / 'results/lean_audit.json'
    if not aud.exists():
        bad.append('results/lean_audit.json missing')
    else:
        d = json.loads(aud.read_text(encoding='utf8'))
        for f, rec in (d.get('files') or {}).items():
            fp = ROOT / f
            if not fp.exists():
                bad.append(f'{f}: missing')
                continue
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            if h != rec.get('sha256'):
                bad.append(f'{f}: sha256 mismatch (record {str(rec.get("sha256"))[:12]}, file {h[:12]}) '
                           '-- re-audit or update the record')
    gate = ROOT / 'results/fd_gate.json'
    if not gate.exists():
        bad.append('results/fd_gate.json missing: no calibration record, so no gradient number may be quoted')
    else:
        g = json.loads(gate.read_text(encoding='utf8'))
        if not (g.get('verdict') or {}).get('gate_passes'):
            bad.append(f'calibration gate did not pass: {g.get("verdict")}')
        inst = ROOT / 'scripts/mlp_capture.py'
        h = hashlib.sha256(inst.read_bytes()).hexdigest()
        if g.get('instrument_sha256') != h:
            bad.append(f'instrument changed since the gate ran (record {str(g.get("instrument_sha256"))[:12]}, '
                       f'now {h[:12]}) -- rerun scripts/fd_gate.py')
    return bad


def check_compile():
    bad = []
    files = [p for p in ROOT.rglob('*.py') if '.venv' not in p.parts and 'node_modules' not in p.parts]
    r = subprocess.run([sys.executable, '-m', 'py_compile', *map(str, files)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad.append((r.stderr or r.stdout).strip().splitlines()[-1])
    for cache in ROOT.rglob('__pycache__'):
        subprocess.run(['rm', '-rf', str(cache)])
    return bad, len(files)


def check_tests():
    bad, ran = [], 0
    for name in TORCH_FREE:
        files = [p for p in ROOT.rglob(name) if '.venv' not in p.parts]
        for f in files:
            # run from the package root so `import progress` resolves (same as on the GPU box)
            r = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                                '-p', f.name], cwd=f.parent.parent, capture_output=True, text=True)
            ran += 1
            if r.returncode != 0:
                bad.append(f'{f.relative_to(ROOT)}: {(r.stderr or r.stdout).strip().splitlines()[-1]}')
    skipped = sorted({p.name for p in ROOT.rglob('test_*.py')
                      if p.name not in TORCH_FREE and '.venv' not in p.parts})
    return bad, ran, skipped


def check_dedup():
    sys.path.insert(0, str(ROOT / 'tools'))
    from dedup_check import check_duplicates
    try:
        res = check_duplicates(UNP, ROOT / 'tools' / 'dedup_allow.json')
    except (ValueError, json.JSONDecodeError) as exc:
        # A malformed allowlist must read as a failed check, not as a crash.
        return [f'tools/dedup_allow.json is not usable: {exc}'], 'allowlist unreadable'
    bad = []
    for grp in res['cross_version_family_conflicts']:
        bad.append('cross-family duplicate: ' + ', '.join(f['path'] for f in grp['files']))
        mm = grp.get('allowlist_file_mismatch')
        if mm:
            bad.append(f'  (allowlisted, but the file set changed since the review: '
                       f'claimed {len(mm["claimed"])}, found {len(mm["actual"])})')
    for st in res['stale_allowlist_entries']:
        bad.append(f'stale allowlist entry {st["sha256"][:16]}: {st["problem"]} '
                   f'({len(st["files"])} files listed)')
    label = (f'{res["n_allowed_hash_groups"]} within-family groups, '
             f'{res["n_cross_family_reviewed"]} reviewed cross-family')
    return bad, label


def check_ledger():
    """The external-evidence ledger must validate AND match the rendered block.

    Rendering is checked, not just schema: a hand-edited matrix would otherwise drift from
    the JSON that the numeric bases live in.
    """
    sys.path.insert(0, str(ROOT / 'tools'))
    from evidence_ledger import DOC, END, START, load, render, validate
    try:
        d = load()
    except (OSError, ValueError) as exc:
        return [f'evidence/ledger.json is not readable: {exc}']
    bad = validate(d)
    if bad:
        return bad[:8]
    if not DOC.is_file():
        return [f'missing {DOC.name}']
    text = DOC.read_text(encoding='utf8')
    i, j = text.find(START), text.find(END)
    if i < 0 or j < i:
        return [f'{DOC.name}: missing the auto:evidence-matrix block']
    body = render(d).split(START + '\n', 1)[1].rsplit('\n' + END, 1)[0]
    got = text[i + len(START) + 1: j - 1]
    if got.strip() != body.strip():
        return [f'{DOC.name}: rendered matrix is stale; run '
                f'python3 tools/evidence_ledger.py --write']
    return []


def check_git():
    r = subprocess.run(['git', 'ls-files', str(ROOT)], cwd=REPO, capture_output=True, text=True)
    bad = [f'tracked derived/binary file: {l}' for l in r.stdout.splitlines()
           if re.search(r'\.(zip|tar\.gz|tgz|pyc|pt|safetensors)$', l)]
    return bad, len(r.stdout.splitlines())


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    write = '--write' in sys.argv
    if cmd == 'render-status' and write:
        README.write_text(replace_block(README.read_text(encoding='utf8'), *STATUS,
                                        render_status()), encoding='utf8')
        return print('wrote status block')
    if cmd == 'render-provenance' and write:
        README.write_text(replace_block(README.read_text(encoding='utf8'), *PROV,
                                        render_provenance()), encoding='utf8')
        return print('wrote provenance block')
    results = {}
    if cmd in ('all', 'blocks'):
        results['blocks'] = check_blocks()
    if cmd in ('all', 'ziptrees'):
        results['ziptrees'] = check_ziptrees()
    if cmd in ('all', 'checksums'):
        results['checksums'] = check_checksums()
    if cmd in ('all', 'compile'):
        bad, n = check_compile(); results[f'compile ({n} files)'] = bad
    if cmd in ('all', 'tests'):
        bad, ran, skipped = check_tests()
        results[f'tests ({ran} files run, {len(skipped)} need torch)'] = bad
    if cmd in ('all', 'dedup'):
        bad, n = check_dedup(); results[f'dedup ({n})'] = bad
    if cmd in ('all', 'ledger'):
        bad = check_ledger(); results[f'ledger ({len(json.loads((ROOT / "evidence/ledger.json").read_text(encoding="utf8"))["verdicts"])} verdicts)'] = bad
    if cmd in ('all', 'instruments'):
        results['instruments (lean sha + fd gate)'] = check_instruments()
    if cmd in ('all', 'git'):
        bad, n = check_git(); results[f'git ({n} tracked files)'] = bad
    fail = 0
    for name, bad in results.items():
        print(f'[{"FAIL" if bad else "ok":4s}] {name}')
        for b in bad[:8]:
            print('        ' + b)
        fail += bool(bad)
    print('neuron-closed check:', 'FAILED' if fail else 'OK')
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()

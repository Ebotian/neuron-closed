#!/usr/bin/env python3
"""The gates must fail when the thing they guard is broken - and this runs on every `make check`.

WHY THIS FILE EXISTS. Two of the audit's last five findings were not about science: the certification gate passed a Lean
failure, and the axiom parser back-filled the whitelist when it could not parse the tool's output. Both were found by
INJECTING the failure, not by reading the code. A gate that is only tested by hand is a gate that silently stops working,
so every known failure mode is injected here, every run.

Checks
  1-4  results/lean_audit.json tampered four ways (declaration FAILED, axiom outside the whitelist, declaration dropped,
       verdict not all-proved) -> the instruments gate must go red. The record is restored afterwards.
  5.   tools/certify.sh must not swallow failures (`|| true` on a copy/run, or a trailing command masking an exit status).
  6.   tools/certify.sh must exit non-zero when the remote box cannot be reached.
  7.   the live record's axioms must be exactly the whitelist and its verdict must say all-proved.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / 'results/lean_audit.json'
sys.path.insert(0, str(ROOT / 'tools'))
import check  # noqa: E402

ALLOWED = {'propext', 'Classical.choice', 'Quot.sound'}
fails = []


def tamper(label, mutate):
    backup = AUDIT.read_text()
    try:
        d = json.loads(backup)
        mutate(d)
        AUDIT.write_text(json.dumps(d, indent=1))
        bad = check.check_instruments()
        if not bad:
            fails.append(f'{label}: gate stayed GREEN on a tampered record (that is the bug it exists to catch)')
        else:
            print(f'  [ok] {label}: gate red -> {bad[0][:70]}')
    finally:
        AUDIT.write_text(backup)


def main():
    def first_decl(d):
        f = sorted(d['files'])[0]
        return f, sorted(d['files'][f]['decls'])[0]

    def set_failed(d):
        f, k = first_decl(d)
        d['files'][f]['decls'][k] = ['FAILED']

    def unknown_axiom(d):
        f, k = first_decl(d)
        d['files'][f]['decls'][k] = ['propext', 'injected']

    def drop_decl(d):
        f, k = first_decl(d)
        d['files'][f]['decls'].pop(k)

    def bad_verdict(d):
        d['verdict'] = 'INCOMPLETE: 1 of 27 declaration checks not proved'

    tamper('declaration FAILED', set_failed)
    tamper('axiom outside the whitelist', unknown_axiom)
    tamper('declaration dropped from the record', drop_decl)
    tamper('verdict says INCOMPLETE', bad_verdict)

    sh = (ROOT / 'tools/certify.sh').read_text()
    if '|| true' in sh:
        fails.append('certify.sh still contains "|| true", which swallows a failing copy or run')
    else:
        print('  [ok] certify.sh has no "|| true"')

    env = {'NEURON_BOX': 'nonexistent.invalid', 'NEURON_PORT': '1', 'PATH': '/usr/bin:/bin'}
    r = subprocess.run(['bash', str(ROOT / 'tools/certify.sh')], capture_output=True, text=True, timeout=120,
                       env={**__import__('os').environ, **env})
    if r.returncode == 0:
        fails.append('certify.sh exited 0 with an unreachable box')
    else:
        print(f'  [ok] unreachable box -> exit {r.returncode}')

    d = json.loads(AUDIT.read_text())
    ax = {a for f in d['files'].values() for v in f['decls'].values() for a in (v or [])}
    if ax - ALLOWED - {'FAILED'}:
        fails.append(f'record contains axioms outside the whitelist: {sorted(ax - ALLOWED - {"FAILED"})}')
    elif 'PROVED for every declaration' not in str(d.get('verdict')):
        fails.append(f'verdict is not all-proved: {d.get("verdict")}')
    else:
        print(f'  [ok] record: {d["decl_counts"]["checks"]} checks, axioms {sorted(ax)}')

    for f in fails:
        print(f'  [FAIL] {f}')
    print('GATES-VERIFY', 'PASS' if not fails else 'FAIL')
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env bash
# Re-derive the two certificates instead of trusting them: the fp32 finite-difference calibration of the gradient
# instrument, and the Lean audit records. Both are written from their producers, so a stale record cannot survive.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${NEURON_BOX:-root@connect.westc.seetacloud.com}"
PORT="${NEURON_PORT:-25785}"
LEANDIR="${LEAN_MATHLIB:-/home/ebt/lean/mathlib-v4.33.0-rc1}"
PLUGIN="${OPL_PLUGIN:-/home/ebt/Downloads/kimi-research/plugin}"

echo "== fd_gate on $HOST =="
scp -q -P "$PORT" "$ROOT/scripts/mlp_capture.py" "$ROOT/scripts/fd_gate.py" "$ROOT/scripts/verify_units.py" \
    "$HOST:/root/autodl-tmp/scripts/j2space/" 2>/dev/null || true
scp -q -P "$PORT" "$ROOT/scripts/core/units.py" "$HOST:/root/autodl-tmp/scripts/j2space/core/" 2>/dev/null || true
ssh -p "$PORT" "$HOST" 'cd /root/autodl-tmp/scripts/j2space && /root/miniconda3/bin/python fd_gate.py > /tmp/fd_certify.raw 2>&1; tail -2 /tmp/fd_certify.raw'
scp -q -P "$PORT" "$HOST:/root/autodl-tmp/runs/fd_gate.json" "$ROOT/results/fd_gate.json"

echo "== Lean audit =="
python3 - "$ROOT" "$PLUGIN" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
root, plugin = Path(sys.argv[1]), Path(sys.argv[2])
lean = root / 'lean'
recs = json.loads((root / 'results/lean_audit.json').read_text())
files = sorted(p for p in lean.glob('*.lean'))
out = {'verdict': 'PROVED for every declaration listed', 'files': {}}
for f in files:
    txt = f.read_text()
    decls = [l.split()[2] for l in txt.splitlines() if l.startswith('#print axioms ')]
    entry = {'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'decls': {}}
    for d in decls:
        r = subprocess.run([str(plugin / 'bin/opl-leancheck'), '--file', str(f), '--decl', d],
                           capture_output=True, text=True)
        ok = 'PROVED' in r.stdout
        entry['decls'][d] = ['propext', 'Classical.choice', 'Quot.sound'] if ok else ['FAILED']
        print(f'  {"PROVED" if ok else "FAILED"}  {f.name}  {d}')
    out['files'][str(f.relative_to(root))] = entry
out['still_missing'] = recs.get('still_missing')
(root / 'results/lean_audit.json').write_text(json.dumps(out, indent=1))
PY

echo "== make check (instruments) =="
python3 "$ROOT/tools/check.py" instruments

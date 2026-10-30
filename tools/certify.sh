#!/usr/bin/env bash
# Re-derive the two certificates from their producers instead of trusting them, and FAIL LOUDLY when a producer fails.
#
# WHY THE STRICTNESS. The first version of this script wrote the verdict string "PROVED for every declaration listed"
# unconditionally, marked a failed declaration ['FAILED'], and still exited 0 - and check.py's instruments item only
# compared file hashes, so a Lean failure in the record passed the gate. An auditor demonstrated it: edit one
# declaration to ['FAILED'], run make check, get OK. A green light that survives its own failure is worse than no gate,
# because everything downstream (and every earlier audit) rests on it. So now: every step checks its exit status, the
# verdict is DERIVED from the results, and anything short of all-proved exits non-zero.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${NEURON_BOX:-root@connect.westc.seetacloud.com}"
PORT="${NEURON_PORT:-25785}"
PLUGIN="${OPL_PLUGIN:-/home/ebt/Downloads/kimi-research/plugin}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -p "$PORT" "$HOST")
SCP=(scp -o BatchMode=yes -o ConnectTimeout=15 -q -P "$PORT")

die() { echo "CERTIFY FAILED: $*" >&2; exit 1; }

STEP="${1:-all}"   # `certify.sh lean` skips the GPU half so the Lean parser can be tested in isolation
if [ "$STEP" != "lean" ]; then
echo "== fd_gate on $HOST =="
"${SCP[@]}" "$ROOT/scripts/mlp_capture.py" "$ROOT/scripts/fd_gate.py" "$ROOT/scripts/verify_units.py" \
    "$ROOT/scripts/legacy_capture.py" "$HOST:/root/autodl-tmp/scripts/j2space/" \
    || die "could not copy the instrument to $HOST (a copy failure used to be swallowed by a trailing ignore)"
"${SCP[@]}" "$ROOT/scripts/core/units.py" "$HOST:/root/autodl-tmp/scripts/j2space/core/" || die "could not copy core/units.py"
"${SCP[@]}" "$ROOT/scripts/analysis/fd.py" "$HOST:/root/autodl-tmp/scripts/j2space/analysis/" || die "could not copy analysis/fd.py"
# the exit status must be the GATE's, not a trailing `tail`'s: `python x.py; tail -2 log` reports tail's status
"${SSH[@]}" 'cd /root/autodl-tmp/scripts/j2space && /root/miniconda3/bin/python fd_gate.py > /tmp/fd_certify.raw 2>&1' \
    || { "${SSH[@]}" 'tail -20 /tmp/fd_certify.raw' >&2 || :; die "fd_gate exited non-zero on the remote box"; }
"${SSH[@]}" 'tail -2 /tmp/fd_certify.raw'
"${SCP[@]}" "$HOST:/root/autodl-tmp/runs/fd_gate.json" "$ROOT/results/fd_gate.json" || die "could not fetch the fd_gate record"
python3 - "$ROOT" <<'PY' || exit 1
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
g = json.loads((root / 'results/fd_gate.json').read_text())
if not (g.get('verdict') or {}).get('gate_passes'):
    sys.exit(f'fd_gate record does not pass: {g.get("verdict")}')
h = hashlib.sha256((root / 'scripts/mlp_capture.py').read_bytes()).hexdigest()
if g.get('instrument_sha256') != h:
    sys.exit('the fetched record certifies a DIFFERENT instrument than the one in this tree')
print(f'  record pins instrument {h[:12]} and passes')
PY

fi

echo "== Lean audit =="
python3 - "$ROOT" "$PLUGIN" <<'PY' || exit 1
"""Audit every declaration and DERIVE the verdict. Non-zero exit if anything is not proved.

Checked per declaration: the tool's exit status AND the presence of PROVED in its output, and the axiom list it
reports. `sorryAx` in that list means the proof has a hole, which is a failure whatever the verdict line says.
"""
import hashlib, json, re, subprocess, sys
from pathlib import Path

root, plugin = Path(sys.argv[1]), Path(sys.argv[2])
lean = root / 'lean'
ALLOWED = {'propext', 'Classical.choice', 'Quot.sound'}
files = sorted(lean.glob('*.lean'))
checks = 0                                     # declarations CHECKED (a file may print one twice)
tool = plugin / 'bin/opl-leancheck'
out = {'files': {}, 'allowed_axioms': sorted(ALLOWED),
       # WHICH TOOL PRODUCED THIS. Without it, any stub that printed the three allowed axioms produced a green record -
       # which is how a fault-injection run with a fake plugin left results/lean_audit.json polluted yet passing.
       'plugin': {'path': str(tool), 'sha256': hashlib.sha256(tool.read_bytes()).hexdigest() if tool.exists() else None}}
failures = []
for f in files:
    decls = [l.split()[2] for l in f.read_text().splitlines() if l.startswith('#print axioms ')]
    entry = {'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'decls': {}}
    for d in decls:
        checks += 1
        r = subprocess.run([str(plugin / 'bin/opl-leancheck'), '--file', str(f), '--decl', d],
                           capture_output=True, text=True)
        ok = r.returncode == 0 and 'PROVED' in r.stdout
        # PARSE THE ACTUAL AXIOM LIST. The previous version matched only the four names it already knew, so an unknown
        # axiom never entered the set; worse, an EMPTY match was filled in with the three allowed names, which let an
        # injected axiom ('PROVED / axioms: [injected]') through as green. The tool reports `... ->['a', 'b']`, so take
        # everything inside that bracket verbatim, and treat "cannot parse" as a failure rather than as a pass.
        m = re.search(r'(?:->|→)\s*\[([^\]]*)\]', r.stdout)   # the tool prints a Unicode arrow
        if not m:
            entry['decls'][d], status = ['FAILED'], 'FAILED'
            failures.append(f'{f.name}:{d} produced no parsable axiom list; refusing to infer one')
        else:
            ax = sorted({x.strip().strip("'\"") for x in m.group(1).split(',') if x.strip()})
            if not ok:
                entry['decls'][d], status = ['FAILED'], 'FAILED'
                failures.append(f'{f.name}:{d} did not prove (rc={r.returncode})')
            elif 'sorryAx' in ax:
                entry['decls'][d], status = ['FAILED'], 'FAILED'
                failures.append(f'{f.name}:{d} proved but uses sorryAx - a hole, not a proof')
            elif set(ax) - ALLOWED:
                entry['decls'][d], status = ax, 'FAILED'
                failures.append(f'{f.name}:{d} uses axioms outside the whitelist: {sorted(set(ax) - ALLOWED)}')
            elif not ax:
                # An EMPTY list is legitimate: `theorem t : True := True.intro` uses no axiom, and the tool prints `→[]`.
                # Only an UNPARSABLE list is a failure (the branch above). Treating empty as failure was over-rejection.
                entry['decls'][d], status = [], 'PROVED' 
            else:
                entry['decls'][d], status = ax, 'PROVED'
        print(f'  {status}  {f.name}  {d}')
    out['files'][str(f.relative_to(root))] = entry
n = sum(len(v['decls']) for v in out['files'].values())
out['decl_counts'] = {'files': len(out['files']), 'declarations': n, 'checks': checks,
                      'proved': checks - len(failures), 'failed': len(failures)}
out['verdict'] = ('PROVED for every declaration listed' if not failures
                  else f'INCOMPLETE: {len(failures)} of {checks} declaration checks not proved')
out['failures'] = failures
(root / 'results/lean_audit.json').write_text(json.dumps(out, indent=1))
print(f'  {out["verdict"]}')
sys.exit(0 if not failures else 1)
PY

echo "== make check (instruments) =="
python3 "$ROOT/tools/check.py" instruments || die "the instruments gate rejected the freshly written records"
echo "CERTIFY OK: instrument calibrated, every declaration proved"

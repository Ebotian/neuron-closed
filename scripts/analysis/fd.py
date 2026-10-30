"""The finite-difference comparison itself - the arithmetic the calibration gate reports.

Split out of fd_gate.py so the gate is a driver: build the fp32 model, take the analytic gradients from the shipped
module and from the legacy control, and hand them here.

CRITERION, frozen: G1 = for every COORDINATE, the MINIMUM over eps of the relative error is < 2%. The minimum is taken
because fp32 log-probability noise (~1e-3 absolute) makes the smaller step unreliable while the larger step leaves the
linear regime, so neither single step is a fair test - and several per-(coordinate, eps) values do exceed 2% at 1e-3.
G2 = the legacy method must show > 5% somewhere, otherwise the gate cannot see the bug it exists to catch.
"""
from __future__ import annotations


def compare(cur, leg, want, eps_list, phi, topn, tolerance=0.02, legacy_floor=0.05):
    rows, summary = [], []
    ok_new, ok_legacy_visible = True, False
    for L in want:
        for n in topn(L, cur[L]):
            best = best_leg = None
            for eps in eps_list:
                fd = (phi({(L, n): eps}) - phi({(L, n): -eps})) / (2 * eps)
                ra = abs(fd - float(cur[L][n])) / max(abs(float(cur[L][n])), 1e-6)
                rl = abs(fd - float(leg[L][n])) / max(abs(float(leg[L][n])), 1e-6)
                best = ra if best is None else min(best, ra)
                best_leg = rl if best_leg is None else min(best_leg, rl)
                rows.append({'layer': L, 'neuron': int(n), 'analytic': float(cur[L][n]),
                             'legacy': float(leg[L][n]), 'fd': fd, 'eps': eps,
                             'rel_err_new': ra, 'rel_err_legacy': rl})
            summary.append({'layer': L, 'neuron': int(n), 'min_rel_err_new': best,
                            'min_rel_err_legacy': best_leg})
            ok_new &= best < tolerance
            ok_legacy_visible |= best_leg > legacy_floor
    verdict = {'G1_instrument_passes': ok_new, 'G2_legacy_signature_visible': ok_legacy_visible,
               'G3_all_layers_ok': ok_new, 'gate_passes': bool(ok_new and ok_legacy_visible)}
    return rows, summary, verdict

#!/usr/bin/env python3
"""Exploratory scale-up gate: non-significant 7/8 is NOT a confirmatory result."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from audit import file_hash

# Preregistered design thresholds, not p-values. Code changes invalidate scan fingerprints.
MIN_PAIRED_N=8
MIN_PAIRED_WIN_FRAC=7/8
MIN_TOP64_LOO_OVERLAP=.5


def gate_decision(scan_dir:Path,verify_dir:Path,trait='hostility'):
    scan_dir=Path(scan_dir); verify_dir=Path(verify_dir)
    meta=json.loads((scan_dir/'manifest.json').read_text(encoding='utf8'))
    vm=json.loads((verify_dir/'manifest.json').read_text(encoding='utf8'))
    analysis=json.loads((verify_dir/'analysis.json').read_text(encoding='utf8'))
    stability_file=scan_dir/f'{trait}_stability.json'
    if not stability_file.is_file():
        raise FileNotFoundError(
            f'Missing LOO stability summary: {stability_file}. '
            f'Run: python stability.py --scan-dir {scan_dir} --trait {trait}')
    stability=json.loads(stability_file.read_text(encoding='utf8'))
    reasons=[]
    if meta.get('schema')!='gradient-scan-v2' or vm.get('schema')!='verification-v2':
        reasons.append('Expected compatible scan-v2/verification-v2 schemas')
    if vm.get('scan_identity')!=file_hash(scan_dir/'manifest.json'):
        reasons.append('Verification does not correspond to this scan manifest')
    if vm.get('trait')!=trait: reasons.append('Verification trait mismatch')
    if meta.get('probe_file_sha256')!=vm.get('probe_file_sha256'):
        reasons.append('Original discovery probe hashes mismatch')
    if meta.get('model_fingerprint')!=vm.get('model_fingerprint'):
        reasons.append('Model fingerprint mismatch')
    if meta.get('code_fingerprint')!=vm.get('code_fingerprint'):
        reasons.append('Code fingerprint mismatch')
    if stability.get('num_probes',0)<3:
        reasons.append('Require at least three discovery probes for LOO')
    loo=stability.get('top_k',{}).get('64',{}).get('mean_overlap')
    if loo is None or loo<MIN_TOP64_LOO_OVERLAP:
        reasons.append(f'Mean LOO Top-64 overlap {loo} < {MIN_TOP64_LOO_OVERLAP}')
    matched=analysis.get('paired_inference',{}).get('ranking',{})
    sign=matched.get('sign_test',{})
    unique_parent=matched.get('independent_parent_matches',False)
    n=sign.get('n_effective',0);wins=sign.get('wins',0)
    n_paired=len(analysis.get('matched_pairs',[]))
    if not unique_parent:reasons.append('Repeated parent Top-K coordinates invalidate independent matched pair gate')
    if n_paired<MIN_PAIRED_N or n<MIN_PAIRED_N:
        reasons.append(f'Require >= {MIN_PAIRED_N} non-tied matched coordinate pairs')
    if n<=0 or wins/n<MIN_PAIRED_WIN_FRAC:
        reasons.append(f'Win rate {wins}/{n} < {MIN_PAIRED_WIN_FRAC:.3f}')
    decision={'pass':not reasons,'reasons':reasons,'trait':trait,
              'n_pairs':n_paired,'non_tied_pairs':n,'wins':wins,
              'win_fraction':wins/n if n else None,'loo_top64_mean_overlap':loo,
              'sign_p_two_sided':sign.get('p_two_sided'),
              'thresholds':{'n_pairs_min':MIN_PAIRED_N,
                            'win_fraction_min':MIN_PAIRED_WIN_FRAC,
                            'mean_top64_loo_overlap_min':MIN_TOP64_LOO_OVERLAP},
              'meaning':'EXPLORATORY scale-up gate only; 7/8 gives two-sided p=0.0703125, not p<.05.',
              'inference_warning': ('This gate is a heuristic resource allocation rule, NOT a confirmatory test. '
                                    'Matched coordinates share heldout prompts and discovery selection; '
                                    'distinct parent IDs do not establish statistical independence.'),
              'source_hashes':{'scan_manifest':file_hash(scan_dir/'manifest.json'),
                               'verify_manifest':file_hash(verify_dir/'manifest.json'),
                               'analysis':file_hash(verify_dir/'analysis.json'),
                               'stability':file_hash(stability_file)}}
    return decision


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scan-dir',required=True)
    ap.add_argument('--verify-dir',required=True)
    ap.add_argument('--trait',default='hostility')
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    decision=gate_decision(Path(args.scan_dir),Path(args.verify_dir),args.trait)
    serialized=json.dumps(decision,ensure_ascii=False,indent=2)
    if args.out: Path(args.out).write_text(serialized+'\n',encoding='utf8')
    print(serialized)
    if not decision['pass']:raise SystemExit(2)

if __name__=='__main__':main()

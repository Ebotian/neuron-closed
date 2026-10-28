#!/usr/bin/env python3
"""Validate the external-evidence ledger and render its matrices.

Why this exists. Records 08/09/10 compared external work to ours in prose, one source at a
time. Prose does not stay honest as sources accumulate: a verdict can lose its numeric basis,
a claim can silently disappear, or a source can be cited without its premise differences being
stated. This tool makes the comparison MECHANICAL:

  * every verdict must be one of S / W / X / N;
  * every S / W / X verdict must carry a `basis` with at least one digit in it, or the check
    fails -- "strengthens" with no number is exactly the failure mode we are guarding against;
  * every claim must have at least one verdict, and every verdict must point at a known claim
    and a known source;
  * the premise table is derived from the ledger's `sources`, so it cannot drift from them.

Usage:
  python3 tools/evidence_ledger.py --check          # validate; exit 1 on any violation
  python3 tools/evidence_ledger.py --render         # print the matrices as markdown
  python3 tools/evidence_ledger.py --write          # rewrite the auto block in
                                                    # 11_synthesis-external-evidence.md

Stdlib only; no network, no GPU.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # qwentest/
LEDGER = ROOT / 'evidence/ledger.json'
DOC = ROOT / '11_synthesis-external-evidence.md'
START = '<!-- auto:evidence-matrix:start -->'
END = '<!-- auto:evidence-matrix:end -->'
VERDICTS = {'S': 'strengthens', 'W': 'weakens', 'X': 'premises differ', 'N': 'no information'}
NUM = re.compile(r'\d')


def load():
    return json.loads(LEDGER.read_text(encoding='utf8'))


def validate(d):
    bad = []
    claims = {c['id'] for c in d['claims']}
    sources = {s['id'] for s in d['sources']}
    if len(claims) != len(d['claims']):
        bad.append('duplicate claim id')
    if len(sources) != len(d['sources']):
        bad.append('duplicate source id')
    seen = set()
    for v in d['verdicts']:
        key = (v['claim'], v['source'])
        if key in seen:
            bad.append(f'duplicate verdict {key}')
        seen.add(key)
        if v['claim'] not in claims:
            bad.append(f'verdict for unknown claim {v["claim"]}')
        if v['source'] not in sources:
            bad.append(f'verdict for unknown source {v["source"]}')
        if v['verdict'] not in VERDICTS:
            bad.append(f'{key}: verdict {v["verdict"]!r} not in {sorted(VERDICTS)}')
        if v['verdict'] in ('S', 'W', 'X') and not NUM.search(v.get('basis') or ''):
            bad.append(f'{key}: verdict {v["verdict"]} has no digit in its basis -> '
                       f'a directional verdict needs a number')
        if v['verdict'] == 'N' and not (v.get('basis') or '').strip():
            bad.append(f'{key}: N verdict must say WHY it carries no information')
    for c in d['claims']:
        if not any(v['claim'] == c['id'] for v in d['verdicts']):
            bad.append(f'claim {c["id"]} has no verdict at all')
    return bad


def render(d):
    claims = d['claims']
    sources = d['sources']
    cell = {(v['claim'], v['source']): v for v in d['verdicts']}
    out = [START, '', '### 主张 × 来源 矩阵（程序生成，勿手改）', '',
           '格内符号：**S** 增强 · **W** 弱化 · **X** 前提不同 · **N** 无信息 · `·` 未评。',
           '', '| 主张 | ' + ' | '.join(s['id'] for s in sources) + ' | 净结论 |',
           '|---|' + '---|' * len(sources) + '---|']
    for c in claims:
        row, tally = [], {'S': 0, 'W': 0, 'X': 0, 'N': 0}
        for s in sources:
            v = cell.get((c['id'], s['id']))
            row.append(v['verdict'] if v else '·')
            if v:
                tally[v['verdict']] += 1
        net = (f"W×{tally['W']} 为主" if tally['W'] > tally['S'] else
               f"S×{tally['S']} 为主" if tally['S'] > tally['W'] else
               'S/W 相抵' if tally['S'] else '无增强')
        if tally['X']:
            net += f"，X×{tally['X']}"
        out.append(f"| `{c['id']}` {c['claim'][:52]} | " + ' | '.join(row) + f' | {net} |')
    out += ['', '### 每条判定的数值依据（程序校验：S/W/X 必须含数字）', '',
            '| 主张 | 来源 | 判 | 依据 |', '|---|---|---|---|']
    for v in sorted(d['verdicts'], key=lambda v: (v['claim'], v['source'])):
        basis = (v['basis'] or '').replace('|', '/')
        note = (' — ' + v['note'].replace('|', '/')) if v.get('note') else ''
        out.append(f"| {v['claim']} | {v['source']} | **{v['verdict']}** | {basis}{note} |")
    out += ['', '### 来源前提表（程序生成）', '',
            '| id | 来源 | 模型 | 量化 | 剂量定义 | 结局量 | 单位 | n | 判官 κ |', '|---|---|---|---|---|---|---|---:|---|']
    for s in sources:
        k = s.get('judge_kappa')
        ks = ('/'.join(f'{a}={b}' for a, b in k.items()) if isinstance(k, dict) else
              (f'{k}' if k else '—'))
        out.append(f"| {s['id']} | {s['label']} | {s['model']} | {s['quant']} | "
                   f"{str(s['dose_def'])[:46]} | {str(s['outcome'])[:40]} | {s['unit']} | "
                   f"{s.get('n_contexts') or '—'} | {ks} |")
    out += ['', END]
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    d = load()
    bad = validate(d)
    if a.check or not (a.render or a.write):
        if bad:
            print('ledger FAILED:')
            for b in bad:
                print('  -', b)
            return 1
        print(f'ledger OK: {len(d["sources"])} sources, {len(d["claims"])} claims, '
              f'{len(d["verdicts"])} verdicts')
    if bad:
        for b in bad:
            print('  -', b)
        return 1
    if a.render:
        print(render(d))
    if a.write:
        text = DOC.read_text(encoding='utf8')
        i, j = text.find(START), text.find(END)
        body = render(d)
        if i >= 0 and j > i:
            body = body.split(START + '\n', 1)[1].rsplit('\n' + END, 1)[0]
            new = text[:i] + START + '\n' + body + '\n' + END + text[j + len(END):]
        else:
            new = text.rstrip() + '\n\n' + render(d) + '\n'
        DOC.write_text(new, encoding='utf8')
        print(f'wrote matrix into {DOC.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

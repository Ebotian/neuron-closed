"""Three-way comparison with the community's published per-coordinate measurements.

Inputs (all outside git except our own results):
  /tmp/g_all32.npz                                         our closed-form gradients, 32 x 9,216 (dose units)
  /home/ebt/neuronary-external/neuron-explorer/explorer/cache/steer_index.npz
        their per-coordinate next-token KL at +-0.8 x (median residual norm / ||W_down[:,n]||),
        i.e. the SAME dose formula we use, measured on 4 short contexts (last position)
  .../lens_index.npz      their per-neuron logit-lens top-12 (up/down)
  .../connections.npz     their 24 strongest direct-path links per neuron (up/down)
  .../vocab.json          token strings
  07_results/positions_all32_K1024.csv     our closed-form list (32 layers)
  07_results/positions_K256.csv            our closed-form list (layers 13-14)
  /home/ebt/neuronary-external/community_coords_all.json   their recipe coordinates

Sections: A measured KL vs our Q/|g|; B the recipe coordinates under both instruments;
          C semantics of our top positions (their lens) vs random; D structure (their 24-link graph).
"""
import csv, json, statistics, sys
from pathlib import Path

import numpy as np

CACHE = Path('/home/ebt/neuronary-external/neuron-explorer/explorer/cache')
REPO = Path('/home/ebt/rsi-dev/qwentest')
D = 9216
LAYERS = 32


def flat(arr):
    return np.asarray(arr, dtype=np.float64).reshape(LAYERS * D)


def spearman(x, y):
    xr = np.argsort(np.argsort(x)).astype(np.float64)
    yr = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(xr, yr)[0, 1])


def pct(sorted_vals, v):
    return float(np.searchsorted(sorted_vals, v) / len(sorted_vals))


g = np.load(sys.argv[1] if len(sys.argv) > 1 else '/tmp/g_all32.npz')
gpool = np.abs(g['pooled']).reshape(-1)
st = np.load(CACHE / 'steer_index.npz')
kl_up = flat(np.maximum(st['up_kl'], 0)); kl_dn = flat(np.maximum(st['down_kl'], 0))
kl_mean = 0.5 * (kl_up + kl_dn)
print(f'coordinates: {gpool.size}; their dose = {float(st["dose"])} (fraction of median residual norm)')

# ---------------------------------------------------------------- A. measured KL vs our g / Q
print('\n=== A. their measured KL (+-0.8) vs our closed form ===')
for name, kl in (('KL_up', kl_up), ('KL_down', kl_dn), ('KL_mean', kl_mean)):
    print(f'  Spearman(|g| pooled, {name}) over all {gpool.size} coordinates: {spearman(gpool, kl):+.4f}')
per_layer = []
for l in range(LAYERS):
    s = slice(l * D, (l + 1) * D)
    per_layer.append(spearman(gpool[s], kl_mean[s]))
print(f'  per-layer Spearman(|g|, KL_mean): median {statistics.median(per_layer):+.3f} '
      f'min {min(per_layer):+.3f} max {max(per_layer):+.3f}')

ours32 = list(csv.DictReader(open(REPO / '07_results/positions_all32_K1024.csv')))
ours2 = list(csv.DictReader(open(REPO / '07_results/positions_K256.csv')))
sorted_kl = np.sort(kl_mean)


def idx_of(layer, neuron):
    return int(layer) * D + int(neuron)


print('\n  our closed-form lists scored by THEIR KL:')
for tag, rows in (('32-layer pooled', ours32), ('13/14 pooled', ours2)):
    for K in (64, 256, 1024):
        if len(rows) < K:
            continue
        vals = [kl_mean[idx_of(r['layer'], r['neuron'])] for r in rows[:K]]
        rnd = np.random.default_rng(0).choice(gpool.size, K, replace=False)
        rvals = kl_mean[rnd]
        print(f'    {tag:15s} K={K:<5} their KL median {statistics.median(vals):.3f} '
              f'(random {statistics.median(rvals):.3f}; all-coordinate p50 {np.median(kl_mean):.3f}) '
              f'| share above their p90: {np.mean([v > np.percentile(kl_mean, 90) for v in vals]):.1%} '
              f'(random {np.mean(rvals > np.percentile(kl_mean, 90)):.1%})')

print('\n  overlap of the two rankings (Jaccard of the top-K sets, KL_mean vs our Q order):')
order_kl = np.argsort(-kl_mean)
for K in (64, 256, 1024, 4096):
    theirs = set(order_kl[:K].tolist())
    ours = {idx_of(r['layer'], r['neuron']) for r in ours32[:K]}
    print(f'    K={K:<5} |A&B| = {len(theirs & ours):<5} Jaccard {len(theirs & ours) / len(theirs | ours):.3f}')

# ---------------------------------------------------------------- B. their recipe coordinates
comm = json.load(open('/home/ebt/neuronary-external/community_coords_all.json'))
cc = [(int(c['layer']), int(c['neuron'])) for c in comm['coords']]
print(f'\n=== B. their {len(cc)} recipe coordinates, scored by both instruments ===')
pcts = [pct(sorted_kl, kl_mean[idx_of(l, n)]) for l, n in cc]
print(f'  their KL_mean percentile: median {statistics.median(pcts):.3f} '
      f'p90 {np.percentile(pcts, 90):.3f} max {max(pcts):.3f}')
above = sum(1 for p in pcts if p > 0.90)
print(f'  coordinates in their own top-10% by KL: {above}/{len(pcts)} ({above / len(pcts):.1%}; random 10%)')
myrank = {}
for r in ours32[:1024]:
    myrank[(int(r['layer']), int(r['neuron']))] = int(r['rank'])
inlist = [(c, myrank[c]) for c in cc if c in myrank]
print(f'  coordinates inside our 32-layer top-1024: {len(inlist)} -> {inlist}')

# ---------------------------------------------------------------- C. semantics of our positions
vocab = json.load(open(CACHE / 'vocab.json'))
li = np.load(CACHE / 'lens_index.npz')
si = np.load(CACHE / 'steer_index.npz')
lex = json.loads((REPO / 'lexicons/jibes_adj_verb_pools.json').read_text())
HOSTILE = set()
for fam, d in lex['families'].items():
    for w in d['pos'] + d['neg']:
        HOSTILE.add(w.lower())
try:
    bffm = json.loads((REPO / 'lexicons/bffm100_factors.json').read_text())
    def collect(o):
        if isinstance(o, str):
            HOSTILE.add(o.lower())
        elif isinstance(o, dict):
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)
    collect(bffm)
except Exception as exc:
    print(f'  (bffm lexicon not merged: {exc})')
tok_str = lambda t: vocab[int(t)].strip().lower().lstrip('ġ').strip()
print(f'\n=== C. do OUR positions\' own lens words look like our construct? ({len(HOSTILE)} lexicon words) ===')


def lex_hits(layer, n, kind):
    idx = li['up' if kind == 'logit' else 'up'][layer, n] if kind in ('logit', 'steer') else None
    arr = li['up'][layer, n] if kind == 'logit' else si['up'][layer, n]
    ws = [tok_str(t) for t in arr]
    return sum(1 for w in ws if w in HOSTILE), ws


for tag, rows in (('32-layer pooled', ours32), ('13/14 pooled', ours2)):
    for K in (64, 256):
        if len(rows) < K:
            continue
        n_hit = 0
        tot = 0
        sample = []
        for r in rows[:K]:
            h, ws = lex_hits(int(r['layer']), int(r['neuron']), 'logit')
            n_hit += h
            tot += len(ws)
            if h and len(sample) < 3:
                sample.append((r['layer'], r['neuron'], [w for w in ws if w in HOSTILE]))
        print(f'  {tag:15s} K={K:<4} lexicon-word hits in logit lens: {n_hit}/{tot} ({n_hit / tot:.2%})'
              + (f'  e.g. {sample}' if sample else ''))

rng = np.random.default_rng(1)
for K in (64, 256):
    n_hit = tot = 0
    for i in rng.choice(gpool.size, K, replace=False):
        h, ws = lex_hits(i // D, i % D, 'logit')
        n_hit += h
        tot += len(ws)
    print(f'  {"RANDOM":15s} K={K:<4} lexicon-word hits in logit lens: {n_hit}/{tot} ({n_hit / tot:.2%})')

# ---------------------------------------------------------------- D. structure
co = np.load(CACHE / 'connections.npz')
print('\n=== D. their 24-link graph ===')
edges = set()
for direction in ('up', 'down'):
    lay = co[f'{direction}_layer']; nn = co[f'{direction}_n']
    for l in range(LAYERS):
        for n in range(D):
            for k in range(int(co['k'])):
                a = (l, n); b = (int(lay[l, n, k]), int(nn[l, n, k]))
                if 0 <= b[0] < LAYERS:
                    edges.add((a, b))
print(f'  directed edges: {len(edges)}')


def density(points):
    s = set(points)
    inside = sum(1 for (a, b) in edges if a in s and b in s)
    return inside / max(1, len(s) * 24)


for tag, rows in (('our 32-layer top-64', ours32[:64]), ('our 32-layer top-256', ours32[:256])):
    pts = [(int(r['layer']), int(r['neuron'])) for r in rows]
    print(f'  {tag:22s} internal edge density {density(pts):.3f}')
rnd = [(int(i // D), int(i % D)) for i in rng.choice(gpool.size, 256, replace=False)]
print(f'  {"random 256":22s} internal edge density {density(rnd):.3f}')
inlist_pts = [c for c, _ in inlist]
if inlist_pts:
    s = set(inlist_pts)
    cross = sum(1 for (a, b) in edges
                if (a in s and b in set((int(r['layer']), int(r['neuron'])) for r in ours32[:256]))
                or (b in s and a in set((int(r['layer']), int(r['neuron'])) for r in ours32[:256])))
    print(f'  edges between their in-list recipe coordinates and our top-256: {cross}')
# ---------------------------------------------------------------- A2. layer-matched scoring
print('\n=== A2. same-layer baseline (their KL percentile computed within each coordinate\'s OWN layer) ===')
kl_by_layer = [np.sort(kl_mean[l * D:(l + 1) * D]) for l in range(LAYERS)]
def layer_pct(l, n):
    return float(np.searchsorted(kl_by_layer[l], kl_mean[l * D + n]) / D)
for tag, rows in (('32-layer pooled', ours32), ('13/14 pooled', ours2)):
    for K in (64, 256):
        if len(rows) < K:
            continue
        ps = [layer_pct(int(r['layer']), int(r['neuron'])) for r in rows[:K]]
        print(f'  {tag:15s} K={K:<4} layer-matched KL percentile: median {statistics.median(ps):.3f} '
              f'share in own-layer top-10%: {np.mean([p > 0.90 for p in ps]):.1%}')
rng2 = np.random.default_rng(7)
ps = [layer_pct(int(i // D), int(i % D)) for i in rng2.choice(gpool.size, 1024, replace=False)]
print(f'  {"random":15s} K=1024 layer-matched KL percentile: median {statistics.median(ps):.3f} '
      f'share in own-layer top-10%: {np.mean([p > 0.90 for p in ps]):.1%}')
pps = [layer_pct(l, n) for l, n in cc]
print(f'  their 165 recipe coordinates, layer-matched: median {statistics.median(pps):.3f} '
      f'share in own-layer top-10%: {np.mean([p > 0.90 for p in pps]):.1%}')
# our Q share (top-1024 only) vs their KL within that set
qshare = np.array([float(r['q_share']) for r in ours32[:1024]])
kv = np.array([kl_mean[idx_of(r['layer'], r['neuron'])] for r in ours32[:1024]])
print(f'  inside our 32-layer top-1024: Spearman(Q share, their KL) = {spearman(qshare, kv):+.3f}')


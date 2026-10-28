"""Frozen-input protocol. Local hashes detect accidental/posthoc edits, not malicious forgery."""
from __future__ import annotations
import json
from pathlib import Path
from audit import (file_hash, code_fingerprint, fingerprint_with_stat_cache,
                   require_same_model, guarded_manifest, json_digest, runtime_versions)
from engine import read_jsonl

ROOT=Path(__file__).resolve().parent


def validate_inputs(spec, rows):
    traits=set(spec['traits'])
    seen=set()
    for r in rows:
        if r.get('id') in seen:raise ValueError('Duplicate probe ID')
        seen.add(r.get('id'))
        if r.get('trait') not in traits or r.get('split') not in ('discovery','heldout'):
            raise ValueError(f'Invalid probe trait/split: {r}')
        if any(not str(r.get(k,'')).strip() for k in ('id','prompt','positive','negative')):
            raise ValueError('Incomplete contrast row')
        if r['positive']==r['negative']:raise ValueError(f'Identical contrast: {r["id"]}')
    for t in traits:
        for s in ('discovery','heldout'):
            if not any(r['trait']==t and r['split']==s for r in rows):
                raise ValueError(f'Missing {s} rows for {t}')
    return {'n_rows':len(rows), 'splits':{t:{s:sum(x['trait']==t and x['split']==s for x in rows)
                                           for s in ('discovery','heldout')} for t in sorted(traits)}}


def frozen_plan(model_dir, semantic_file, contrast_file, reuse=False):
    spec=json.loads(Path(semantic_file).read_text(encoding='utf8'))
    rows=read_jsonl(contrast_file)
    counts=validate_inputs(spec,rows)
    fp,mode=fingerprint_with_stat_cache(model_dir,reuse=reuse)
    provenance=Path(semantic_file).with_suffix('.provenance.json')
    return {'schema':'snam-frozen-plan-v1','source_release':'snam-v1',
            'semantic_expansion_provenance_file':str(provenance.resolve()) if provenance.exists() else None,
            'semantic_expansion_provenance_sha256':file_hash(provenance) if provenance.exists() else None,
            'model_dir':str(Path(model_dir).resolve()), 'model_fingerprint':fp,
            'fingerprint_mode':mode, 'source_code_sha256':code_fingerprint(ROOT),
            'semantic_file':str(Path(semantic_file).resolve()),
            'semantic_sha256':file_hash(semantic_file),
            'contrast_file':str(Path(contrast_file).resolve()),
            'contrast_sha256':file_hash(contrast_file),
            'counts':counts,'protocol':'One discovery/test split; no candidate selection on heldout.'}


def require_plan(run_dir, reuse=False, model_check=True):
    path=Path(run_dir)/'manifest.json'
    if not path.is_file():raise FileNotFoundError(f'Missing {path}; run freeze.py first')
    m=json.loads(path.read_text(encoding='utf8'))
    if m.get('schema')!='snam-frozen-plan-v1':raise ValueError('Unsupported run schema')
    for name,sha in [('semantic_file','semantic_sha256'),('contrast_file','contrast_sha256')]:
        if file_hash(m[name])!=m[sha]:raise RuntimeError(f'Frozen input modified: {m[name]}')
    if m.get('semantic_expansion_provenance_file') and (
        file_hash(m['semantic_expansion_provenance_file'])!=m['semantic_expansion_provenance_sha256']):
        raise RuntimeError('Self-expansion provenance file changed since freeze')
    if code_fingerprint(ROOT)!=m['source_code_sha256']:
        raise RuntimeError('Python source changed since freeze. Use a NEW run directory.')
    if model_check:require_same_model(m['model_dir'],m,reuse_fingerprint=reuse)
    return m


def rows_for(plan,trait,split,limit=None):
    xs=[x for x in read_jsonl(plan['contrast_file']) if x['trait']==trait and x['split']==split]
    if limit is not None:xs=xs[:limit]
    if not xs:raise ValueError(f'No {split} rows for {trait}')
    return xs


def stage_plan(run_dir,name,params):
    parent=Path(run_dir)
    m=require_plan(parent,model_check=False)
    record={'schema':'snam-stage-v1','frozen_parent_sha256':json_digest(m),
            'stage':name,'parameters':params,'runtime':runtime_versions()}
    guarded_manifest(parent/name,record)
    return parent/name


def discovery_inventory(discovery_dir):
    from audit import file_hash
    d=Path(discovery_dir)
    return {p.relative_to(d).as_posix():file_hash(p) for p in sorted(d.rglob('*'))
            if p.is_file() and p.name not in ('SEAL.json','manifest.json')
            and not p.name.endswith('.tmp')}


def seal_discovery(discovery_dir):
    d=Path(discovery_dir);seal=d/'SEAL.json'
    inv=discovery_inventory(d)
    if seal.is_file():
        old=json.loads(seal.read_text(encoding='utf8'))
        if old!=inv:raise RuntimeError('Discovery artifacts changed after sealing; refuse overwrite')
    else:
        seal.write_text(json.dumps(inv,ensure_ascii=False,sort_keys=True,indent=2),encoding='utf8')
    return inv


def require_discovery_seal(discovery_dir):
    d=Path(discovery_dir);path=d/'SEAL.json'
    if not path.is_file():raise FileNotFoundError(f'Discovery incomplete/unsealed: {path}')
    if json.loads(path.read_text(encoding='utf8'))!=discovery_inventory(d):
        raise RuntimeError('Discovery artifact file hashes do not match immutable seal')

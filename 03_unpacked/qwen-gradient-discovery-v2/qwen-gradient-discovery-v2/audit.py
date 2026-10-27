"""Deterministic manifests, model integrity and experiment guards (stdlib only)."""
from __future__ import annotations
import hashlib
import json
import os
import platform
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def fingerprint_model(model_dir: str | Path) -> dict:
    """Hash actual shards, configs and tokenizer assets, not merely config.json."""
    p = Path(model_dir)
    patterns = ('*.safetensors', '*.safetensors.index.json', 'config.json',
                'configuration.json', 'tokenizer.json', 'tokenizer_config.json',
                'chat_template.jinja', 'merges.txt', 'vocab.json',
                'special_tokens_map.json')
    files = {f.name: sha256_file(f) for pattern in patterns for f in p.glob(pattern)
             if f.is_file()}
    if not any(n.endswith('.safetensors') for n in files):
        raise ValueError(f'No weight shards found in {p}')
    if 'config.json' not in files:
        raise ValueError(f'No config.json in {p}')
    return {'file_hashes': dict(sorted(files.items())),
            'content_sha256': json_digest(files)}


def file_hash(path: str | Path) -> str:
    return sha256_file(path)


def runtime_versions() -> dict:
    from importlib.metadata import version, PackageNotFoundError
    v = {'python': platform.python_version()}
    for name in ('torch', 'transformers', 'triton', 'fla-core',
                 'flash-linear-attention', 'causal-conv1d', 'accelerate'):
        try:
            v[name] = version(name)
        except PackageNotFoundError:
            v[name] = None
    try:
        import torch
        v['torch_cuda_runtime'] = torch.version.cuda
        v['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        v['torch_cuda_runtime'] = None
        v['gpu_name'] = None
    return v


def guarded_manifest(out_dir: str | Path, plan: dict) -> None:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'manifest.json'
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf8'))
        if existing != plan:
            raise RuntimeError(f'Manifest conflict: {d} is not the same experiment. Use a new --out.')
    else:
        if any(d.iterdir()):
            raise RuntimeError(f'Non-empty {d} has no manifest: refuse to mix results.')
        temp = d / 'manifest.json.tmp'
        temp.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf8')
        os.replace(temp, path)


def require_same_model(model_dir, scan_meta):
    if 'model_fingerprint' not in scan_meta:
        raise ValueError('This scan is v1 (no weight fingerprints). Re-run v2 scan in NEW output dir.')
    found = fingerprint_model(model_dir)
    if found != scan_meta['model_fingerprint']:
        raise ValueError('Model shard/tokenizer SHA256 mismatch with scan manifest.')
    return found


def check_scan_file(path: str | Path, scan_meta: dict) -> None:
    expected = scan_meta.get('probe_file_sha256')
    if expected and file_hash(path) != expected:
        raise ValueError('Probe file SHA256 differs from discovery (avoid accidental data changes).')



def require_same_runtime(scan_meta):
    expected=scan_meta.get('runtime')
    if expected is None:
        raise ValueError('Discovery manifest lacks runtime versions.')
    found=runtime_versions()
    if found!=expected:
        diff={k:(expected.get(k),found.get(k)) for k in set(expected)|set(found)
              if expected.get(k)!=found.get(k)}
        raise RuntimeError(f'Runtime changed since scan: {diff}. Use a new scan or restore environment.')
    return found


def code_fingerprint(project_dir=None):
    root=Path(project_dir) if project_dir else Path(__file__).resolve().parent
    files={f.name:sha256_file(f) for f in root.glob('*.py') if f.is_file()}
    return json_digest(dict(sorted(files.items())))

def require_same_code(scan_meta):
    old=scan_meta.get('code_fingerprint')
    if old is None or old!=code_fingerprint():
        raise RuntimeError('Python source files changed since scan. Re-run scan in a new --out.')

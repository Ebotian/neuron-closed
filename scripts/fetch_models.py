#!/usr/bin/env python3
"""Fetch small CPU models onto a box that cannot reach huggingface.co, robustly.

STATUS (2026-10-28, end of day): USED but NOT YET COMPLETED END-TO-END. It fetched 280 MB of
438 MB (unitary/toxic-bert) with ranged chunks and zero chunk failures before the box was shut
down, so the chunk/resume/retry path is exercised but the SHA256 verification and the
rename-to-final path have never run. Two bugs were found and fixed BEFORE the run that mattered:
an endless loop when --size was omitted, and a 500 MB read_bytes() for hashing.

Why this exists. The GPU box cannot resolve huggingface.co (curl returns 000 in 5 s) but it CAN
reach hf-mirror.com (HTTP 200 in 1.4 s, ranged requests at ~1.3 MB/s). Downloads must therefore
go through HF_ENDPOINT-style mirror URLs, and they must survive a flaky link: this script fetches
each file in ranged chunks, resumes a partial chunk, retries with exponential backoff, and
verifies the SHA256 against the value published by the Hugging Face API before declaring success.

It is a HARNESS, not an experiment: no GPU, no torch, stdlib only. It writes a manifest so a
later run can tell what was verified and when.

Usage (on the box):
    python3 fetch_models.py --dest /root/autodl-tmp/gauges \
        --model unitary/toxic-bert --file model.safetensors \
        --sha256 2c272885d24138df70bff1b3cd944a999bd6b41dad33209730aa8ba074f6ad09 \
        --model s-nlp/roberta_toxicity_classifier --file pytorch_model.bin \
        --sha256 896f52af161757594994d1fefe85ac79cbb57c9a96b26fba8ab93139b6426b8c

  --chunk BYTES     ranged chunk size (default 8 MiB)
  --retries N       per-chunk retries (default 6)
  --endpoint URL    mirror base (default https://hf-mirror.com)
  --manifest PATH   where to append what was verified (JSONL)

Design notes:
  * A chunk that stalls is abandoned and retried from its first missing byte, so a 2-minute hang
    costs one chunk, not the file.
  * Progress is printed as [file] done/total MB so a background job is watchable.
  * Verification is against the publisher's sha256, not against a size.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

CHUNK = 8 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, size_hint: int | None, chunk: int, retries: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    attempt = 0
    while True:
        have = tmp.stat().st_size if tmp.exists() else 0
        if size_hint and have >= size_hint:
            break
        lo = have
        hi = have + chunk - 1
        if size_hint:
            hi = min(hi, size_hint - 1)
        attempt += 1
        r = subprocess.run(['curl', '-sSL', '--max-time', '180', '-r', f'{lo}-{hi}',
                            '-o', str(tmp) + '.chunk', '-w', '%{http_code}',
                            url], capture_output=True, text=True)
        code = (r.stdout or '').strip()[-3:]
        ok = code in ('200', '206') and Path(str(tmp) + '.chunk').exists()
        if ok:
            with open(str(tmp) + '.chunk', 'rb') as src:
                data = src.read()
            with tmp.open('ab') as fh:
                fh.write(data)
            Path(str(tmp) + '.chunk').unlink()
            got = tmp.stat().st_size
            total = f'{size_hint / 1e6:.0f}' if size_hint else '?'
            print(f'  [{dest.name}] {got / 1e6:7.1f}/{total} MB', flush=True)
            attempt = 0
            if len(data) == 0:
                # An EMPTY response is a transient failure, not end-of-file: the first version broke
                # out of the loop here and then reported a SHA256 mismatch on a 21 MB fragment. When
                # the total size is known we keep going until it is reached; only with no size hint
                # do we accept an empty chunk as the end.
                if size_hint:
                    print(f'  [{dest.name}] empty chunk at {lo}, retrying', flush=True)
                    time.sleep(2)
                    continue
                break
            # a short chunk with no size hint means the server had nothing more: end of file
            if not size_hint and len(data) < (hi - lo + 1):
                break
            continue
        print(f'  [{dest.name}] chunk {lo}-{hi} failed (http={code or "?"}), '
              f'attempt {attempt}/{retries}', flush=True)
        Path(str(tmp) + '.chunk').unlink(missing_ok=True)
        if attempt >= retries:
            raise SystemExit(f'give up on {dest} after {retries} attempts at offset {lo}')
        time.sleep(min(2 ** attempt, 30))
    if size_hint and tmp.stat().st_size != size_hint:
        raise SystemExit(f'{dest}: stopped at {tmp.stat().st_size} of {size_hint} bytes; '
                         f'keeping {tmp} for a later resume')
    if dest.exists():
        dest.unlink()
    tmp.rename(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', required=True)
    ap.add_argument('--model', action='append', required=True,
                    help='repo id, e.g. unitary/toxic-bert (repeatable, paired with --file)')
    ap.add_argument('--file', action='append', required=True)
    ap.add_argument('--sha256', action='append', required=True)
    ap.add_argument('--size', action='append', type=int, default=None)
    ap.add_argument('--endpoint', default='https://hf-mirror.com')
    ap.add_argument('--chunk', type=int, default=CHUNK)
    ap.add_argument('--retries', type=int, default=6)
    ap.add_argument('--manifest', default=None)
    a = ap.parse_args()
    if not (len(a.model) == len(a.file) == len(a.sha256)):
        raise SystemExit('--model/--file/--sha256 must be given the same number of times')

    sizes = a.size or []
    if sizes and len(sizes) != len(a.model):
        raise SystemExit('--size, when given, must be repeated once per --file (or omitted entirely)')
    if not sizes:
        sizes = [None] * len(a.model)

    dest_root = Path(a.dest)
    manifest = Path(a.manifest) if a.manifest else dest_root / 'fetch_manifest.jsonl'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    for repo, name, want, size in zip(a.model, a.file, a.sha256, sizes):
        dest = dest_root / repo.replace('/', '_') / name
        url = f'{a.endpoint}/{repo}/resolve/main/{name}'
        if dest.exists() and sha256_file(dest) == want:
            print(f'[skip] {dest} already verified', flush=True)
        else:
            print(f'[get ] {repo}/{name} -> {dest}  (chunk {a.chunk >> 20} MiB)', flush=True)
            t0 = time.time()
            # NOTE: the first version hardcoded None here, so --size never reached fetch() and a short
            # chunk was accepted as end-of-file; that produced two SHA256 mismatches on 21 MB and
            # 206 MB fragments before the wiring bug was found.
            fetch(url, dest, size, a.chunk, a.retries)
            got = sha256_file(dest)
            if got != want:
                raise SystemExit(f'SHA256 MISMATCH for {dest}: got {got} want {want}')
            rec = {'repo': repo, 'file': name, 'bytes': dest.stat().st_size,
                   'sha256': got, 'endpoint': a.endpoint,
                   'seconds': round(time.time() - t0, 1)}
            with manifest.open('a', encoding='utf8') as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            print(f'[ok  ] {dest}  {rec["bytes"] / 1e6:.1f} MB  {rec["seconds"]} s', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    sys.exit(main())

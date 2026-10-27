#!/usr/bin/env python3
"""Opt-in, dry-run-first migration: text into Git, zip archives stay ignored.

Use on user's own repository; no repository was supplied to the author.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import stat
import zipfile
from pathlib import Path

IGNORE_EXT={'.zip','.tar','.gz','.zst','.pt','.pth','.npy','.npz','.safetensors',
            '.png','.jpg','.jpeg','.webp','.pdf','.pyc','.bin','.so','.whl'}
START='<!-- auto:unpacked-packages:start -->'
END='<!-- auto:unpacked-packages:end -->'


def digest(data, name):return hashlib.new(name,data).hexdigest()

def accepted_members(zf):
    members=[]
    for obj in zf.infolist():
        name=Path(obj.filename)
        if name.is_absolute() or '..' in name.parts or '\\' in obj.filename:
            raise ValueError(f'Unsafe ZIP member path: {obj.filename}')
        if (obj.external_attr>>16)&0o170000==stat.S_IFLNK:
            raise ValueError(f'ZIP symlink refused: {obj.filename}')
        if obj.is_dir():continue
        if name.suffix.lower() in IGNORE_EXT or any(x=='__pycache__' for x in name.parts):continue
        if obj.file_size>3_000_000:raise ValueError(f'Individual file too large for text-only git migration: {name}')
        content=zf.read(obj)
        if b'\0' in content:continue
        try:content.decode('utf-8')
        except UnicodeDecodeError:continue
        members.append((name,content))
    return members


def migrate(repo:Path,apply=False):
    repo=repo.resolve();src=repo/'01_packages';dst=repo/'03_unpacked';readme=repo/'00_README.md'
    if not src.is_dir():raise FileNotFoundError(f'Missing package archive directory: {src}')
    packages=sorted(src.glob('*.zip'))
    if not packages:raise ValueError('No ZIP archives in 01_packages/')
    items=[];planned=[]
    for archive in packages:
        with zipfile.ZipFile(archive) as zf:
            if zf.testzip():raise ValueError(f'Damaged ZIP archive: {archive}')
            members=accepted_members(zf)
        # Keep original member paths; never silently overwrite a differently sourced ZIP.
        target=dst/archive.stem
        for name,content in members:
            out=target/name
            if out.exists() and out.read_bytes()!=content:
                raise FileExistsError(f'Existing extracted content differs: {out}; reconcile manually')
            planned.append((out,content))
        blob=archive.read_bytes()
        items.append({'archive':archive.name,'md5':digest(blob,'md5'),
                      'sha256':digest(blob,'sha256'),'extracted_text_files':len(members),
                      'destination':str(target.relative_to(repo))})
    table=['| Archive | MD5 | SHA256 | Unpacked text files |',
           '|---|---|---|---:|']
    for item in items:
        table.append(f'| `{item["archive"]}` | `{item["md5"]}` | `{item["sha256"]}` | '
                     f'{item["extracted_text_files"]} |')
    section=START+'\n'+'## Unpacked package provenance (automatically maintained)\n\n'+\
            'Binary ZIP files remain ignored; text versions are under `03_unpacked/`.\n\n'+\
            '\n'.join(table)+'\n'+END
    old_readme=readme.read_text(encoding='utf8') if readme.exists() else ''
    if START in old_readme or END in old_readme:
        if old_readme.count(START)!=1 or old_readme.count(END)!=1:
            raise ValueError('Malformed managed README delimiters')
        updated=old_readme.split(START)[0]+section+old_readme.split(END)[1]
    else:updated=old_readme.rstrip()+'\n\n'+section+'\n'
    gi=repo/'.gitignore'
    old_ignore=gi.read_text(encoding='utf8') if gi.exists() else ''
    new_ignore='\n'.join(x for x in old_ignore.splitlines()
                          if x.strip()!='!01_packages/qwen-persona-matrix.zip').rstrip()+'\n'
    if '01_packages/*.zip' not in new_ignore.splitlines():
        new_ignore+='01_packages/*.zip\n'
    if not apply:
        return {'mode':'dry-run','packages':items,'text_files_to_write':len(planned),
                'readme_update':str(readme),'gitignore_update':str(gi),
                'warning':'Nothing changed; pass --apply to modify repository.'}
    for path,content in planned:
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(content)
    dst.mkdir(exist_ok=True)
    (dst/'package_provenance.json').write_text(json.dumps(items,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    readme.write_text(updated,encoding='utf8')
    gi.write_text(new_ignore,encoding='utf8')
    return {'mode':'applied','packages':items,'text_files_written_or_verified':len(planned),
            'note':'Review git status; the script does not run git add/commit or delete ZIPs.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',required=True,type=Path)
    ap.add_argument('--apply',action='store_true')
    args=ap.parse_args()
    print(json.dumps(migrate(args.repo,args.apply),indent=2,ensure_ascii=False))

if __name__=='__main__':main()

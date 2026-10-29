#!/usr/bin/env python3
"""Opt-in, dry-run-first migration: text into Git, zip archives stay ignored.

Use on user's own repository; no repository was supplied to the author.
"""
from __future__ import annotations
import argparse
import hashlib
import json
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



def _retire_legacy_md5_table(text: str) -> tuple[str, bool]:
    """Replace ONLY a recognizable ZIP-only legacy markdown MD5 table.

    Do not remove a mixed/ambiguous table: human reconciliation is safer than
    silently deleting a non-package checksum index.
    """
    lines = text.splitlines(keepends=True)
    matches=[]; i=0
    while i < len(lines):
        line=lines[i].strip()
        if not (line.startswith('|') and 'md5' in line.lower()
                and any(t in line.lower() for t in ('file','archive','package','name','文件','压缩'))):
            i+=1; continue
        j=i+1
        if j>=len(lines) or not re.match(r'^\s*\|?[\s|:\-]+\|?\s*$',lines[j]):
            i+=1; continue
        k=j+1
        while k<len(lines) and lines[k].lstrip().startswith('|'):
            k+=1
        rows=lines[j+1:k]
        if not rows:
            i=k; continue
        if not any('01_packages/' in row or '.zip' in row.lower() for row in rows):
            i=k; continue
        if not all('.zip' in row.lower() for row in rows):
            raise ValueError('Legacy MD5 table mixes ZIP packages and other files; '
                             'refuse automatic replacement. Reconcile README manually.')
        matches.append((i,k));i=k
    if len(matches)>1:
        raise ValueError('Multiple legacy ZIP MD5 tables; refuse ambiguous auto-replacement')
    if not matches:return text,False
    i,k=matches[0]
    lines[i:k]=['Package checksums are maintained in the '
                '[Unpacked package provenance](#unpacked-package-provenance-automatically-maintained) table below.\n']
    return ''.join(lines),True


def _clean_legacy_exception(ignore: str) -> str:
    """Remove a legacy exception and only its directly adjacent dedicated comments."""
    lines=ignore.splitlines()
    targets=[i for i,l in enumerate(lines)
             if l.strip()=='!01_packages/qwen-persona-matrix.zip']
    remove=set(targets)
    for i in targets:
        start=i
        while start>0 and lines[start-1].lstrip().startswith('#'):
            start-=1
        block=lines[start:i]
        # Preserve unrelated comments even when next to the old exception.
        for j in range(start, i):
            if re.search(r'qwen-persona-matrix|(?:exception|unignore|例外|白名单)',
                         lines[j],re.IGNORECASE):
                remove.add(j)
    result=[x for i,x in enumerate(lines) if i not in remove]
    if '01_packages/*.zip' not in [x.strip() for x in result]:
        result.append('01_packages/*.zip')
    return '\n'.join(result).rstrip()+'\n'

def migrate(repo:Path,apply=False):
    repo=repo.resolve();src=repo/'01_packages';dst=repo/'03_unpacked';readme=repo/'00_README.md'
    if not src.is_dir():raise FileNotFoundError(
        f'Missing {src}; --repo must point to the workspace directly containing 01_packages/ '
        f'(e.g. /path/to/research-repo/qwentest), not its parent.')
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
    # Exclude the managed section from legacy detection; re-run is idempotent.
    head=old_readme.split(START)[0] if START in old_readme else old_readme
    head,retired_md5=_retire_legacy_md5_table(head)
    if START in old_readme:
        old_readme=head+START+old_readme.split(START,1)[1]
    else:
        old_readme=head
    if START in old_readme or END in old_readme:
        if old_readme.count(START)!=1 or old_readme.count(END)!=1:
            raise ValueError('Malformed managed README delimiters')
        updated=old_readme.split(START)[0]+section+old_readme.split(END)[1]
    else:updated=old_readme.rstrip()+'\n\n'+section+'\n'
    gi=repo/'.gitignore'
    old_ignore=gi.read_text(encoding='utf8') if gi.exists() else ''
    new_ignore=_clean_legacy_exception(old_ignore)
    if not apply:
        return {'mode':'dry-run','packages':items,'text_files_to_write':len(planned), 'legacy_md5_table_retired':retired_md5,
                'readme_update':str(readme),'gitignore_update':str(gi),'workspace_root':str(repo),
                'warning':'Nothing changed; pass --apply to modify repository.'}
    for path,content in planned:
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(content)
    dst.mkdir(exist_ok=True)
    provenance=json.dumps(items,indent=2,ensure_ascii=False)+'\n'
    provenance_path=dst/'package_provenance.json'
    for path,content in ((provenance_path,provenance),(readme,updated),(gi,new_ignore)):
        if not path.exists() or path.read_text(encoding='utf8')!=content:
            path.write_text(content,encoding='utf8')
    return {'mode':'applied','packages':items,'text_files_written_or_verified':len(planned),'legacy_md5_table_retired':retired_md5,
            'note':'Review git status; the script does not run git add/commit or delete ZIPs.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',required=True,type=Path,
                    help='WORKSPACE root directly containing 01_packages/, e.g. /repo/qwentest')
    ap.add_argument('--apply',action='store_true')
    args=ap.parse_args()
    print(json.dumps(migrate(args.repo,args.apply),indent=2,ensure_ascii=False))

if __name__=='__main__':main()

"""Build imatrix GGUF files from existing calibration text without replacing team code."""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def executable(name: str, provided: str | None = None) -> Path:
    candidates = ([Path(provided).expanduser()] if provided else []) + [
        Path.home()/"llama.cpp"/"build-cpu"/"bin"/name,
        Path.home()/"llama.cpp"/"build"/"bin"/name,
    ]
    resolved = shutil.which(name)
    if resolved: candidates.append(Path(resolved))
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK): return p.resolve()
    raise FileNotFoundError(f"Cannot find {name}. Supply --binary, or build llama.cpp tools.")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',choices=['chat','math','code'],required=True)
    p.add_argument('--input',type=Path,help='Calibration .txt; defaults to data/calibration/<domain>.txt')
    p.add_argument('--source',type=Path,default=ROOT/'models/Llama-3.2-1B-Instruct-BF16.gguf')
    p.add_argument('--output',type=Path)
    p.add_argument('--binary',help='Path to llama-imatrix')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    source=a.source.expanduser().resolve()
    inp=(a.input or ROOT/'data/calibration'/f'{a.domain}.txt').expanduser().resolve()
    output=(a.output or ROOT/'models/imatrix'/f'{a.domain}.imatrix.gguf').expanduser().resolve()
    for item in [source,inp]:
        if not item.is_file(): p.error(f'Missing input: {item}')
    if a.threads<1:p.error('--threads must be >= 1')
    exe=executable('llama-imatrix',a.binary)
    command=[str(exe),'-m',str(source),'-f',str(inp),'-o',str(output),'--threads',str(a.threads),'--no-ppl']
    print('Command:',subprocess.list2cmdline(command),flush=True)
    if a.dry_run:return
    output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(command,check=True)
    if not output.is_file() or output.stat().st_size==0:raise RuntimeError(f'No imatrix produced at {output}')
    print('Saved',output)
if __name__=='__main__':main()

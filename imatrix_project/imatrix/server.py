"""Start a local llama-server with explicit CPU thread settings and clean shutdown."""
from __future__ import annotations
import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import time
import requests

@contextlib.contextmanager
def managed_server(model:Path, binary:Path, port:int, threads:int, context:int, gpu_layers:int, log_path:Path):
    if not model.is_file():raise FileNotFoundError(model)
    if not binary.is_file():raise FileNotFoundError(binary)
    log_path.parent.mkdir(parents=True,exist_ok=True)
    cmd=[str(binary),'--model',str(model),'--n-gpu-layers',str(gpu_layers),
         '--ctx-size',str(context),'--threads',str(threads),'--threads-batch',str(threads),
         '--parallel','1','--host','127.0.0.1','--port',str(port)]
    url=f'http://127.0.0.1:{port}'
    # A prior model server on this port is dangerous: its healthy response is not proof this model started.
    try:
        requests.get(url+'/health',timeout=1)
    except requests.RequestException:
        pass
    else:raise RuntimeError(f'Port {port} is in use. Stop the other llama-server or choose --port.')
    with log_path.open('w',encoding='utf-8') as log:
        print('Server:',subprocess.list2cmdline(cmd),flush=True)
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if proc.poll() is not None:raise RuntimeError(f'llama-server exited; inspect {log_path}')
                try:
                    status=requests.get(url+'/health',timeout=2)
                    if status.ok and status.json().get('status')=='ok':break
                except (requests.RequestException,ValueError):pass
                time.sleep(.5)
            else:raise TimeoutError(f'Model server never became healthy; inspect {log_path}')
            yield url
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()

def discover_server(provided:str|None)->Path:
    options=([Path(provided).expanduser()] if provided else [])+[
        Path.home()/'llama.cpp/build-cpu/bin/llama-server',
        Path.home()/'llama.cpp/build/bin/llama-server']
    path=shutil.which('llama-server')
    if path:options.append(Path(path))
    for item in options:
        if item.is_file() and os.access(item,os.X_OK):return item.resolve()
    raise FileNotFoundError('llama-server missing; pass --server /path/to/llama-server')

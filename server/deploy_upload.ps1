$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
@'
import tarfile
from pathlib import Path
root=Path.cwd()
with tarfile.open('server/deploy.tar.gz','w:gz') as tar:
    for folder in ['agents','datasets','experiments','llm','tests','server','HypoSpace/_HypoSpace-main']:
        for p in (root/folder).rglob('*'):
            if not p.is_file() or any(x in p.parts for x in ['__pycache__','.git','.venv','node_modules','results']): continue
            if p.suffix.lower() not in ['.py','.json','.yaml','.yml','.md','.txt','.sh']: continue
            if 'config' in p.parts and p.suffix != '.md': continue
            tar.add(p,arcname=p.relative_to(root))
    for name in ['requirements.txt','requirements-server.txt','SERVER_PLAN.md','README.md','AGENTS.md']:
        tar.add(root/name,arcname=name)
'@ | .\.venv\Scripts\python.exe -
if ($LASTEXITCODE -ne 0) { throw 'Packaging failed' }
scp server/deploy.tar.gz ECE4300:/opt/data/private/deploy-ece4300.tar.gz
if ($LASTEXITCODE -ne 0) { throw 'Upload failed' }
ssh -o BatchMode=yes ECE4300 "cd '/opt/data/private/ECE4300 Final Project' && tar -xzf /opt/data/private/deploy-ece4300.tar.gz"
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed' }

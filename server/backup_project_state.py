"""Archive project information and results without weights, environments or credentials."""
from datetime import datetime,timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import tarfile
from urllib.request import urlopen


def main():
    root=Path(__file__).resolve().parents[1]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=root/'transfer_exports';destination.mkdir(exist_ok=True)
    archive=destination/f'project-state-{stamp}.tar.gz'
    include=('agents','llm','experiments','server','tests','datasets','HypoSpace','config','results','logs','releases')
    forbidden={'models','cache','.venv','.venv-inference','.venv-tree-tools','__pycache__','.git',
               'checkpoints','adapters','training_checkpoints','transfer_exports'}
    manifest=dict(snapshot_utc=stamp,source=str(root),files=[],excluded=[],
        note='Live informational snapshot. JSON read from atomic result files; append-only logs may end mid-line. Running workloads were not paused.')
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        def add(name,content):
            info=tarfile.TarInfo(name);info.size=len(content);info.mode=0o600
            tar.addfile(info,io.BytesIO(content))
        candidates=[]
        for folder in include:
            base=root/folder
            if not base.exists():continue
            for parent,dirs,files in os.walk(base,followlinks=False):
                dirs[:]=[d for d in dirs if d not in forbidden and not d.startswith('.venv')
                         and not (Path(parent)/d).is_symlink()]
                candidates.extend(Path(parent)/name for name in files)
        candidates.extend(p for p in root.iterdir() if p.is_file() and p.suffix in ('.md','.txt'))
        for path in sorted(set(candidates)):
            relative=str(path.relative_to(root));name=path.name.lower()
            if path.is_symlink() or path.suffix.lower() in ('.pyc','.safetensors','.bin','.pt','.pth','.ckpt','.pem','.key','.tar','.gz'):
                manifest['excluded'].append(relative);continue
            if name.startswith('.env') or 'credential' in name or 'secret' in name or 'ed25519' in name or (name.startswith('llm.') and not name.endswith('.example')):
                manifest['excluded'].append(relative);continue
            try:content=path.read_bytes()
            except FileNotFoundError:continue
            if b'PRIVATE KEY-----' in content and path.suffix not in ('.py',):
                manifest['excluded'].append(relative);continue
            add(relative,content)
            manifest['files'].append(dict(path=relative,bytes=len(content),sha256=sha256(content).hexdigest()))
        status={}
        try:status['gateway']=json.load(urlopen('http://127.0.0.1:8002/health',timeout=5))
        except Exception as error:status['gateway_error']=type(error).__name__
        for name in ('mcts-2.1-medium-pipeline-32k','mcts-2.1-32k-switch'):
            p=root/f'logs/{name}/state.json'
            if p.exists():status[name]=json.loads(p.read_text())
        add('SERVER_STATUS_SNAPSHOT.json',json.dumps(status,ensure_ascii=False,indent=2).encode())
        add('BACKUP_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2).encode())
    digest=sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    checksum=archive.with_suffix(archive.suffix+'.sha256');checksum.write_text(digest.hexdigest()+'\n')
    print(json.dumps(dict(archive=str(archive),checksum=str(checksum),sha256=digest.hexdigest(),
        files=len(manifest['files']),archive_bytes=archive.stat().st_size),indent=2))


if __name__=='__main__':main()

"""Verify installed files against official ModelScope SHA256 metadata."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
metadata=json.loads((root/'logs/official_model_raw_metadata.json').read_text())
results=[]
for f in metadata['Data']['Files']:
    if not f.get('Sha256') or f.get('Type')!='blob':continue
    p=root/'models/Qwen3-8B'/f['Path']
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):digest.update(block)
    row={'file':f['Path'],'revision':f['Revision'],'size':p.stat().st_size,
         'sha256':digest.hexdigest(),'official_sha256':f['Sha256'],
         'matches':digest.hexdigest()==f['Sha256'] and p.stat().st_size==f['Size']}
    results.append(row);print(json.dumps(row),flush=True)
(root/'logs/model_integrity.json').write_text(json.dumps(results,indent=2))
assert all(r['matches'] for r in results)

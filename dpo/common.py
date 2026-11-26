import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def write_jsonl(path,rows):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    tmp.replace(path)


def parse_response(data):
    """Preserve raw content; isolate thinking as the deployed Qwen client does."""
    choice=data['choices'][0];message=choice['message'];content=message.get('content') or ''
    if not isinstance(content,str):raise ValueError('Non-text response')
    reasoning=message.get('reasoning_content')
    if isinstance(reasoning,str):final=content.strip()
    elif '</think>' in content:
        reasoning,final=content.rsplit('</think>',1);reasoning=reasoning.removeprefix('<think>').strip();final=final.strip()
    else:reasoning=content;final=''
    return dict(raw_content=content,reasoning_text=reasoning,final_text=final,final_answer_missing=not final,
                finish_reason=choice.get('finish_reason'),usage=data.get('usage',{}),returned_model=data.get('model'))

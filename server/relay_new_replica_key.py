"""Create a dedicated new-node API key and deliver it to the caller gateway over SSH."""
import json
import shlex
import subprocess


def main():
    source="""/usr/bin/python3.10 -c 'from pathlib import Path; import json,secrets,os; p=Path("/opt/data/private/ECE4300-node4-api-20261125/api_credentials.json"); assert (p.parent/"OWNERSHIP.json").exists(); os.umask(0o077); p.write_text(json.dumps({"api_key":secrets.token_urlsafe(40)})) if not p.exists() else None; p.chmod(0o600); print(p.read_text())'"""
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',
                           'EEC430004-5090',source],stdout=subprocess.PIPE,check=True)
    data=json.loads(result.stdout)
    assert set(data)=={'api_key'} and len(data['api_key'])>=40
    code='from pathlib import Path; import sys,os; os.umask(0o077); p=Path("/opt/data/private/ECE4300 Final Project/config/node4-api-credentials.json"); p.write_bytes(sys.stdin.buffer.read()); p.chmod(0o600)'
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','EEC4300',
                    'python3 -c '+shlex.quote(code)],input=result.stdout,check=True)
    print('DEDICATED_NEW_NODE_API_KEY_REGISTERED')


if __name__=='__main__':main()

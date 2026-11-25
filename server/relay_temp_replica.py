"""Authenticated SSH-to-SSH byte relay; model/runtime never land on local disk."""
import shlex
import subprocess
import time


def main():
    source='/opt/data/private/ECE4300 Final Project'
    target='/opt/data/private/ECE4300-temp-api-20261125'
    send="tar --exclude=__pycache__ --warning=no-file-changed -cf - -C "+shlex.quote(source)+" .venv-inference models/Qwen3-8B"
    receive='test -f '+shlex.quote(target+'/OWNERSHIP.json')+' && tar -xf - -C '+shlex.quote(target)
    src=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','EEC4300',send],stdout=subprocess.PIPE)
    dst=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','lab4090',receive],stdin=subprocess.PIPE)
    total=0;start=last=time.monotonic()
    try:
        while True:
            block=src.stdout.read(1024*1024)
            if not block:break
            dst.stdin.write(block);total+=len(block)
            if time.monotonic()-last>25:
                print('TRANSFER_GIB',round(total/1024**3,2),'MIB_S',round(total/1024**2/(time.monotonic()-start),1),flush=True)
                last=time.monotonic()
        dst.stdin.close();src.stdout.close()
        a=src.wait();b=dst.wait()
        if a or b:raise RuntimeError(f'SSH transfer failed: source={a}, destination={b}')
        print('SSH_TRANSFER_COMPLETE_BYTES',total,flush=True)
    finally:
        if src.poll() is None:src.terminate()
        if dst.poll() is None:dst.terminate()


if __name__=='__main__':main()

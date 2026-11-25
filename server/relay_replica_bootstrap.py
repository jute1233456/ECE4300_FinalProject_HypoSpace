"""Relay only a newly minted one-time transfer token and its public TLS certificate."""
import subprocess


def main():
    source="tar -cf - -C '/opt/data/private/ECE4300 Final Project/transfer_exports/node4-secure-20261125' transfer_credentials.json transfer.crt"
    destination="test -f /opt/data/private/ECE4300-node4-api-20261125/OWNERSHIP.json && umask 077 && tar -xf - -C /opt/data/private/ECE4300-node4-api-20261125"
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','EEC4300',source],
                          stdout=subprocess.PIPE,check=True)
    assert len(result.stdout)<65536
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','EEC430004-5090',destination],
                   input=result.stdout,check=True)
    print('ONE_TIME_TRANSFER_BOOTSTRAP_READY')


if __name__=='__main__':main()

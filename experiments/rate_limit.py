"""Share consecutive HTTP 429 tracking across processes using one pause file."""
import sqlite3
from contextlib import closing
from time import time


def record_request(pause_file, limited):
    pause_file.parent.mkdir(parents=True, exist_ok=True)
    database = pause_file.with_suffix(".rate-limit.sqlite")
    with closing(sqlite3.connect(database, timeout=30)) as connection, connection:
        connection.execute("CREATE TABLE IF NOT EXISTS failures (timestamp REAL)")
        connection.execute("CREATE TABLE IF NOT EXISTS recovery (stage INTEGER)")
        connection.execute("BEGIN IMMEDIATE")
        now = time()
        connection.execute("DELETE FROM failures WHERE timestamp < ?", (now - 1200,))
        if limited:
            connection.execute("INSERT INTO failures VALUES (?)", (now,))
        else:
            connection.execute("DELETE FROM failures")
        count = connection.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
        if count >= 20:
            pause_file.touch()
            marker = pause_file.with_suffix(".cooldown")
            if not marker.exists() and not pause_file.with_suffix(".stopped").exists():
                previous = connection.execute("SELECT stage FROM recovery").fetchone()
                stage = (previous[0] if previous else 0) + 1
                connection.execute("DELETE FROM recovery")
                connection.execute("INSERT INTO recovery VALUES (?)", (stage,))
                connection.execute("DELETE FROM failures")
                if stage == 1:
                    marker.write_text(str(now + 4 * 3600))
                elif stage == 2:
                    marker.write_text(str(now + 1800))
                else:
                    pause_file.with_suffix(".stopped").touch()
    return count >= 20


def recover_quota(pause_file):
    """Resume normal requests after each scheduled cooldown."""
    from time import sleep

    marker = pause_file.with_suffix(".cooldown")
    while not pause_file.with_suffix(".stopped").exists():
        if marker.exists():
            remaining = float(marker.read_text()) - time()
            if remaining <= 0:
                record_request(pause_file, False)
                marker.unlink(missing_ok=True)
                pause_file.unlink(missing_ok=True)
                print("Cooldown finished; normal queue requests resumed.", flush=True)
            else:
                sleep(min(remaining, 30))
                continue
        sleep(1)

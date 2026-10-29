#!/usr/bin/env python3
"""Minimal sshpass replacement: runs a command, answers password prompts over a pty."""
import os, pty, sys, select, subprocess, time

def run(argv, password, timeout=600):
    master, slave = pty.openpty()
    def _pre():
        os.setsid()
        import fcntl, termios
        fcntl.ioctl(1, termios.TIOCSCTTY, 0)

    p = subprocess.Popen(argv, stdin=slave, stdout=slave, stderr=slave,
                         close_fds=True, preexec_fn=_pre)
    os.close(slave)
    buf = b""
    answered = 0
    deadline = time.time() + timeout
    while True:
        if time.time() > deadline:
            print("TIMEOUT", file=sys.stderr)
            p.kill()
            break
        r, _, _ = select.select([master], [], [], 1.0)
        if master in r:
            try:
                data = os.read(master, 4096)
            except OSError:
                break
            if not data:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            buf += data
            low = buf.lower()
            if b"password:" in low and answered < 3 and buf.strip().endswith(b":"):
                os.write(master, (password + "\n").encode())
                answered += 1
                buf = b""
            elif b"yes/no" in low and buf.strip().endswith(b"?"):
                os.write(master, b"yes\n")
                buf = b""
        if p.poll() is not None:
            # drain remaining output
            while True:
                r, _, _ = select.select([master], [], [], 0.2)
                if master not in r:
                    break
                try:
                    data = os.read(master, 4096)
                except OSError:
                    data = b""
                if not data:
                    break
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            break
    rc = p.wait()
    os.close(master)
    return rc

if __name__ == "__main__":
    sys.exit(run(sys.argv[2:], sys.argv[1]))

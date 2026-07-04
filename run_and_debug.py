import subprocess
import time
import sys

print("Starting diagnostics...")
process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8005"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

time.sleep(3)

# Check if process is still running
poll = process.poll()
if poll is None:
    print("Process is running. Terminating...")
    process.terminate()
    stdout, stderr = process.communicate()
    print("STDOUT:")
    print(stdout)
    print("STDERR:")
    print(stderr)
else:
    print(f"Process terminated with exit code: {poll}")
    stdout, stderr = process.communicate()
    print("STDOUT:")
    print(stdout)
    print("STDERR:")
    print(stderr)

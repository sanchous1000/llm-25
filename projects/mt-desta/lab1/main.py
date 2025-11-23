# Entry point for Lab1

"""
This script delegates execution to `run.py`, which handles Ollama model
invocations based on configuration files.
"""

if __name__ == "__main__":
    import subprocess
    subprocess.run(["python", "run.py"], check=True)
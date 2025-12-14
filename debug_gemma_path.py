
import os
from pathlib import Path

cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
print(f"Checking {cache_dir}")

if cache_dir.exists():
    print("Contents:")
    for item in cache_dir.iterdir():
        print(f" - {item.name}")
else:
    print("Cache dir does not exist")

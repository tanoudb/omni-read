import os
import subprocess
import re

series = [
    "hellogin",
    "i-married-the-dragon-i-killed",
    "rise-of-the-dragon-overlord",
    "the-frontier-count's-10th-class-outcas",
    "the-wind-mage"
]

for s in series:
    print(f"Testing {s}...")
    with open("test_corrections.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    content = re.sub(r'SERIE = ".*?"', f'SERIE = "{s}"', content)
    
    with open("test_corrections.py", "w", encoding="utf-8") as f:
        f.write(content)
        
    try:
        subprocess.run([r"a:\omni read\.venv311\Scripts\python.exe", "test_corrections.py"], check=True)
        print(f"Finished {s}.")
    except Exception as e:
        print(f"Error on {s}: {e}")

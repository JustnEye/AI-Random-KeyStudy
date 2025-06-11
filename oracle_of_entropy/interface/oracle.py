import numpy as np
import subprocess
import hashlib
import json
import os
import time

base_dir = os.path.abspath("..")
keys_dir = os.path.join(base_dir, "keys")
outputs_dir = os.path.join(base_dir, "outputs")
oracle_path = os.path.join(base_dir, "analyzer", "oracle")

# Load and flatten keys
keys = np.load(os.path.join(keys_dir, "ai_keys.npy")).astype(np.uint8).flatten()
with open(os.path.join(keys_dir, "ai_keys.bin"), "wb") as f:
    f.write(keys.tobytes())

# Step 2: Call/summon the C entropy oracle
res = subprocess.run([oracle_path, os.path.join(keys_dir, "ai_keys.bin")],
                     capture_output=True, text=True)

if res.returncode != 0:
    raise RuntimeError(f"Entropy oracle failed: {res.stderr.strip()}")

try:
    entropy_str = res.stdout.strip()
    entropy_val = float(entropy_str.split()[1])
except ValueError:
    raise RuntimeError(f"Invalid entropy output: {res.stdout.strip()}")


#Step 3: Symbolic interpretation
ENTROPY_THRESHOLDS = [
    (7.99, "Sacred", "🜄"),
    (7.8, "Blessed", "🜃"),
    (7.5, "Adequate", "🜁"),
    (0.0, "Corrupted", "☠️")
]

def interpret_entropy(entropy):
    for threshold, verdict, symbol in ENTROPY_THRESHOLDS:
        if entropy >= threshold:
            return verdict, symbol

verdict, symbol = interpret_entropy(entropy_val)

#Step 4: Generate a poetic name
def name_key(entropy):
    adjectives = ["Obsidian", "Crimson", "Echoing", "Silent"]
    nouns = ["Seal", "Vector", "Mirror", "Tongue"]
    i = int(hashlib.sha256(str(entropy).encode()).hexdigest(), 16) % len(adjectives)
    j = int(hashlib.md5(str(entropy).encode()).hexdigest(), 16) % len(nouns)
    return f"{adjectives[i]} {nouns[j]}"

name = name_key(entropy_val)

#Step 5: Save the output
output = {
    "entropy": entropy_val,
    "verdict": verdict,
    "symbol": symbol,
    "name": name,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "sha256": hashlib.sha256(keys.tobytes()).hexdigest()
}

os.makedirs(outputs_dir, exist_ok=True)
with open(os.path.join(outputs_dir, "oracle_report.json"), "w") as f:
    json.dump(output, f, indent=2)

#Step 6: Printing the output
print(json.dumps(output, indent=2))


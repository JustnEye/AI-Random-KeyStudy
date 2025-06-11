import numpy as np

keys = np.load("../keys/ai_keys.npy").astype(np.uint8).flatten()
with open("../keys/ai_keys.bin", "wb") as f:
    f.write(keys.tobytes())

print("Converted .npy to .bin for entropy oracle.")


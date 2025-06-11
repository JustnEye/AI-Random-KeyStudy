import numpy as np
import os

#Simulates 10,000 keys. Each of them are 128-bit (16 bytes)
keys = np.random.randint(0, 256, size=(10000, 16), dtype=np.uint8)

#Save to numpy format
os.makedirs("../keys", exist_ok=True)
np.save("../keys/ai_keys.npy", keys)

print("Generated and saved 10,000 AI keys.")

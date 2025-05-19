import torch
import torch.nn as nn
import numpy as np
from scipy.stats import chisquare
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives import hashes
import os

# Define the AI Key Generator model
class KeyGenNet(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, output_size=16):
        super(KeyGenNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Generate random inputs and train the model to produce pseudo-random keys
def train_model():
    model = KeyGenNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(1000):
        inputs = torch.rand(32, 16)  # batch of 32 random inputs
        targets = torch.rand(32, 16)  # goal: random outputs

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

# Generate a batch of keys using the trained model
def generate_ai_keys(model, num_keys=100):
    model.eval()
    with torch.no_grad():
        inputs = torch.rand(num_keys, 16)
        keys = model(inputs)
        return (keys.numpy() * 255).astype(np.uint8)

# Generate AES keys for comparison
def generate_aes_keys(num_keys=100):
    keys = []
    for _ in range(num_keys):
        key = os.urandom(16)  # 128-bit AES key
        keys.append(np.frombuffer(key, dtype=np.uint8))
    return np.array(keys)

# Chi-Square test to compare randomness distribution
def chi_square_test(keys):
    flat = keys.flatten()
    hist, _ = np.histogram(flat, bins=256, range=(0, 255))
    expected = np.full(256, len(flat) / 256)
    chi2, p = chisquare(hist, expected)
    return chi2, p

# --- Run everything ---


def runs_test(binary_key):
    runs = 1
    for i in range(1, len(binary_key)):
        if binary_key[i] != binary_key[i - 1]:
            runs += 1

    n1 = binary_key.count('0')
    n2 = binary_key.count('1')

    if n1 == 0 or n2 == 0:
        expected_runs = 1
    else:
        expected_runs = 1 + (2 * n1 * n2) / (n1 + n2)

    return runs, expected_runs
if __name__ == "__main__":
    print("Training AI key generator...")
    model = train_model()

    print("Generating keys...")
    ai_keys = generate_ai_keys(model, 1000)
    aes_keys = generate_aes_keys(1000)

    print("Running Chi-Square test...")
    ai_chi2, ai_p = chi_square_test(ai_keys)
    aes_chi2, aes_p = chi_square_test(aes_keys)

    print("\nAI Key Generator:")
    print(f"  Chi-Square: {ai_chi2:.2f}, p-value: {ai_p:.4f}")
    print("AES Keys:")
    print(f"  Chi-Square: {aes_chi2:.2f}, p-value: {aes_p:.4f}")
# Convert keys to binary strings using a threshold
ai_binary = ''.join(['1' if byte > 127 else '0' for byte in ai_keys.flatten()])
aes_binary = ''.join(['1' if byte > 127 else '0' for byte in aes_keys.flatten()])

# Run the Runs Test
ai_runs, ai_expected = runs_test(ai_binary)
aes_runs, aes_expected = runs_test(aes_binary)

# Display the results
print("\nRuns Test:")
print(f"AI Keys: Runs = {ai_runs}, Expected = {ai_expected:.2f}")
print(f"AES Keys: Runs = {aes_runs}, Expected = {aes_expected:.2f}")


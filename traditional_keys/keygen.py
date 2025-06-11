import torch
import torch.nn as nn
import numpy as np
from scipy.stats import chisquare
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

# Train the model to generate pseudo-random keys
def train_model():
    model = KeyGenNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for _ in range(1000):
        inputs = torch.rand(32, 16)
        targets = torch.rand(32, 16)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

# Generate keys using a trained model
def generate_ai_keys(model, num_keys=1000):
    model.eval()
    with torch.no_grad():
        inputs = torch.rand(num_keys, 16)
        keys = model(inputs)
        return (keys.numpy() * 255).astype(np.uint8)

# Generate AES-like random keys
def generate_aes_keys(num_keys=1000):
    return np.array([np.frombuffer(os.urandom(16), dtype=np.uint8) for _ in range(num_keys)])

# Chi-Square test
def chi_square_test(keys):
    flat = keys.flatten()
    hist, _ = np.histogram(flat, bins=256, range=(0, 255))
    expected = np.full(256, len(flat) / 256)
    return chisquare(hist, expected)

#The Runs Test
def runs_test(binary_key):
    runs = 1
    for i in range(1, len(binary_key)):
        if binary_key[i] != binary_key[i - 1]:
            runs += 1
    n1 = binary_key.count('0')
    n2 = binary_key.count('1')
    expected = 1 + (2 * n1 * n2) / (n1 + n2) if n1 and n2 else 1
    return runs, expected

# Main
if __name__ == "__main__":
    print("Training AI key generator...")
    model = train_model()
# The keys are now generating
    print("Generating keys...")
    ai_keys = generate_ai_keys(model)
    aes_keys = generate_aes_keys()

    print("Running Chi-Square test...")
    ai_chi2, ai_p = chi_square_test(ai_keys)
    aes_chi2, aes_p = chi_square_test(aes_keys)

    print("\nChi-Square Test:")
    print(f"AI Keys:  Chi-Square = {ai_chi2:.2f}, p-value

# The Oracle of Entropy

The **Oracle of Entorpy** is a custom-made system that determines whether a cryptographic key is truly random. It combines AI generation, binary conversion, C (low-level) entropy analysis, and even symbolic interpretation.

## The Concept

The Oracle is part machine and also part poet. What it does:

- Generates AI (GAN-based) cryptographic keys using Python
- Converts these keys to `.bin` format for raw byte-level analysis
- Analyzes the entropy using a custom program wrriten in C (`entropy_core.c`)
- Interprets the result into a poetic JSON object

### A Sample Output

```json
{
  "entropy": 7.9986,
  "verdict": "Sacred",
  "symbol": "🜄",
  "name": "Silent Tongue",
  "timestamp": "2025-05-23 15:46:15",
  "sha256": "810bfce971cd3ace04c244ac819e533ffea8ddcd59025050e62c495f60a43989"
}
# --- Folder Overview ---

generator/ – Code for GAN-based key generation

interface/ – Python-C bridging tools

interpreter/ – Python that runs the C oracle and formats output

analyzer/ – Contains entropy_core.c, the C-based entropy checker

keys/ – Raw .npy and .bin cryptographic key files

outputs/ – Final entropy verdicts in JSON format

Out of the maximum entropy of 8.0000, the system recorded an Entropy of 7.9986 bits/bute. So the key is extremely random and cryptographically sound.

# The reason that I did this mini-project was to see how randomness is not only a mathematical gimmick, but has a certain poetic essence to it which contributes to truly secure systems. The Oracle of Entropy transforms alien unpredictability into a readable, symbolic language.

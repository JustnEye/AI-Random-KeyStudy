# AI-Random-KeyStudy — Research Report

## Goal
To evaluate whether AI can generate cryptographic keys with entropy and randomness comparable to AES or HRNG systems or even better.

## Method
- Trained GAN with spectral norm, dropout, WGAN-GP
- Generated 10,000 128-bit keys
- Compared them to AES and traditional key models
- Tested using Chi-Square, Runs Test, Entropy histogram, and full NIST STS suite

## Key Results
- GAN Keys: Entropy ~0.999, strong bit-balance
- Passed most NIST tests (Universal failed due to stream size, not any model flaw)
- Traditional Keys: Failed virtually all tests

## Conclusion
AI keygen is good for software-based secure key generation. GANs are close to AES in randomness quality and outperform traditional schedules.

The “failure” still teaches us what we didn’t know of before.

# AI-Random-KeyStudy
A research project exploring AI-driven cryptographic key generation using neural networks. Includes code for key generation, randomness testing (e.g., Chi-Square, Runs Test), and analysis against traditional algorithms like RSA/AES. Developed as part of a STEM Senior Project on the future of AI with encryption models.

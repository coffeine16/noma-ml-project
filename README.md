# NOMA Downlink Detection via Decentralized Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This project explores the application of **Deep Learning (DL)** to **Non-Orthogonal Multiple Access (NOMA)** in wireless communications. Specifically, it implements a decentralized neural network architecture to replace the traditional **Successive Interference Cancellation (SIC)** algorithm for downlink signal detection.

##  Project Overview

In traditional NOMA (Power Domain), multiple users share the same frequency and time resources but are assigned different power levels.
* **Weak User (Far):** High transmission power.
* **Strong User (Near):** Low transmission power.

Conventionally, the Strong User must perform **SIC** (decode Weak User $\to$ subtract it $\to$ decode own signal). This project demonstrates that a Neural Network can learn to decode these superposed signals directly, offering a robust alternative to SIC without explicit mathematical subtraction.

### Key Features
* **NOMA Simulation:** Generates synthetic NOMA datasets with ordered Rayleigh fading ($|h_1| < |h_2|$).
* **Decentralized Architecture:** Utilizes two independent PyTorch neural networks (one per user) to mimic a decentralized receiver setup.
* **Dynamic SNR Training:** Trains the model across a range of SNR levels (5dB - 20dB) to ensure robustness.
* **Benchmarking:** Compares ML performance (BER/SER) directly against the traditional SIC algorithm.
* **Visualization:** Includes scripts for Spectral Efficiency comparison, Constellation diagrams, and BER/Throughput plots.

## The Physics: OMA vs. NOMA

The notebook includes visualizations demonstrating the spectral efficiency gain of NOMA over OMA (Orthogonal Multiple Access).

* **OMA:** Users are separated by time slots (inefficient usage of bandwidth).
* **NOMA:** Users transmit simultaneously.
    * **User 1 (Weak):** Allocated **80% Power**.
    * **User 2 (Strong):** Allocated **20% Power**.

## Model Architecture

The system uses a **Decentralized Detector** comprised of two sub-networks:
1.  **Net U1:** Decodes the high-power signal (Weak User).
2.  **Net U2:** Decodes the low-power signal (Strong User).

**Input Features:** `[Real(y), Imag(y), Real(h), Imag(h)]`
**Output:** Probability distribution over QPSK symbols.

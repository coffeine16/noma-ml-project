Students: Harshita Jain, Keshav Agrawal
Roll No: 23ucc547, 23ucc558

1. Introduction

In this project we tried to simulate a 2-user downlink NOMA system and then use machine learning to detect the bits of both users without using SIC.
We generated our own dataset from the NOMA model and trained two ML models:

Logistic Regression

A small Neural Network (MLP)

The main idea was to check if ML can separate the superposed NOMA signals.

2. System Overview (short)

Modulation: BPSK

Channel: Real flat fading (approx. Rayleigh but only real part used)

Users:

User 1 = far/weak → more power (α1 = 0.8)

User 2 = near/strong → less power (α2 = 0.2)

Transmitted signal:

x = sqrt(alpha1)*s1 + sqrt(alpha2)*s2


Received signals:

y1 = h1*x + n1
y2 = h2*x + n2


Noise is added according to SNR (0 to 20 dB).

We do NOT use SIC anywhere. Detection is done purely using ML.

3. Dataset

For each SNR value (0, 5, 10, 15, 20 dB) we generated several thousand samples.

Each row of the dataset contains:

y1, h1, y2, h2, snr_db, b1, b2


y1, y2 = received samples

h1, h2 = channel gains

b1, b2 = actual transmitted bits

The dataset is saved automatically as noma_dataset.csv when running the code.

4. Machine Learning Models

We trained separate detectors for both users:

Logistic Regression

MLP (Neural Network)

2 hidden layers (16 neurons each)

ReLU activation

Features used (simple):

[y_k, h_k, SNR_dB]


for k = 1 and 2.

We split data into 80% training and 20% testing.

Training accuracies were printed in the terminal when running the script.

5. BER vs SNR

After training, we tested both models at SNR values from 0 to 20 dB (step 2 dB).
Bit Error Rate (BER) was computed for each user.

The script saves the BER plot as:

ber_plot.png

General observations:

BER decreases as SNR increases (expected).

User-2 performs better than user-1.

MLP generally performs better than Logistic Regression.

This shows ML can partially decode signals without SIC.

6. How to Run

Create and activate a virtual environment (optional but recommended):

python -m venv venv
venv\Scripts\activate


Install dependencies:

pip install numpy matplotlib scikit-learn


Run the project:

python main_noma_ml.py


This generates:

noma_dataset.csv

ber_plot.png

7. Files Included

main_noma_ml.py – full simulation + ML training + BER evaluation

noma_dataset.csv – generated dataset

ber_plot.png – BER results

README.md – this file

8. Future Work (we will add later)

Proper SIC-based baseline detector

Comparison with OMA

Complex (I/Q) fading

Larger neural networks

Full final report
# main_noma_ml.py
#
# Simple 2-user downlink NOMA simulation with ML detection
# Models used: Logistic Regression + small Neural Network (MLP)
#
# The code:
#   1. Generates a synthetic dataset for 2-user power-domain NOMA
#   2. Trains two detectors for each user (LogReg and MLP)
#   3. Evaluates BER vs SNR without using SIC
#   4. Saves the dataset and a BER plot

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def generate_noma_data(num_samples_per_snr=4000, snr_db_list=None,
                       alpha1=0.8, alpha2=0.2, seed=0):
    """
    Generate real-valued 2-user downlink NOMA data.

    Returns:
        y1_all, h1_all, y2_all, h2_all, snr_all_db, b1_all, b2_all  (all 1D arrays)
    """
    if snr_db_list is None:
        snr_db_list = [0, 5, 10, 15, 20]

    rng = np.random.default_rng(seed)

    y1_list, y2_list = [], []
    h1_list, h2_list = [], []
    b1_list, b2_list = [], []
    snr_list = []

    for snr_db in snr_db_list:
        snr_lin = 10 ** (snr_db / 10.0)

        # Real AWGN noise: sigma^2 = 1 / (2*SNR) (rough scaling, not exact theory)
        noise_std = np.sqrt(1.0 / (2.0 * snr_lin))

        # Bits for both users
        b1 = rng.integers(0, 2, size=num_samples_per_snr)
        b2 = rng.integers(0, 2, size=num_samples_per_snr)

        # BPSK mapping: 0 -> -1, 1 -> +1
        s1 = 2 * b1 - 1
        s2 = 2 * b2 - 1

        # Real flat fading channels
        h1 = rng.normal(0.0, 1.0, size=num_samples_per_snr)
        h2 = rng.normal(0.0, 1.0, size=num_samples_per_snr)

        # Superposition coding (total transmit power = 1)
        x = np.sqrt(alpha1) * s1 + np.sqrt(alpha2) * s2

        # AWGN at each user
        n1 = noise_std * rng.normal(0.0, 1.0, size=num_samples_per_snr)
        n2 = noise_std * rng.normal(0.0, 1.0, size=num_samples_per_snr)

        # Received signals
        y1 = h1 * x + n1
        y2 = h2 * x + n2

        # Save
        y1_list.append(y1)
        y2_list.append(y2)
        h1_list.append(h1)
        h2_list.append(h2)
        b1_list.append(b1)
        b2_list.append(b2)
        snr_list.append(np.full(num_samples_per_snr, snr_db))

    y1_all = np.concatenate(y1_list)
    y2_all = np.concatenate(y2_list)
    h1_all = np.concatenate(h1_list)
    h2_all = np.concatenate(h2_list)
    b1_all = np.concatenate(b1_list)
    b2_all = np.concatenate(b2_list)
    snr_all_db = np.concatenate(snr_list)

    return y1_all, h1_all, y2_all, h2_all, snr_all_db, b1_all, b2_all


def train_detectors():
    # 1) Create training dataset
    snr_db_list_train = [0, 5, 10, 15, 20]
    num_samples_per_snr = 4000

    y1, h1, y2, h2, snr_db, b1, b2 = generate_noma_data(
        num_samples_per_snr=num_samples_per_snr,
        snr_db_list=snr_db_list_train,
        alpha1=0.8,
        alpha2=0.2,
        seed=1
    )

    # Features for user 1 and user 2.
    # Each row: [received_sample, channel_gain, snr_db]
    X1 = np.column_stack((y1, h1, snr_db))
    X2 = np.column_stack((y2, h2, snr_db))

    # Save combined dataset for the report (optional but nice)
    dataset = np.column_stack((y1, h1, y2, h2, snr_db, b1, b2))
    np.savetxt(
        "noma_dataset.csv",
        dataset,
        delimiter=",",
        header="y1,h1,y2,h2,snr_db,b1,b2",
        comments=""
    )

    # Split into train / test for both users
    X1_train, X1_test, b1_train, b1_test = train_test_split(
        X1, b1, test_size=0.2, random_state=42, stratify=b1
    )

    X2_train, X2_test, b2_train, b2_test = train_test_split(
        X2, b2, test_size=0.2, random_state=42, stratify=b2
    )

    # 2) Train Logistic Regression
    logreg1 = LogisticRegression(max_iter=300)
    logreg2 = LogisticRegression(max_iter=300)

    logreg1.fit(X1_train, b1_train)
    logreg2.fit(X2_train, b2_train)

    acc1_logreg = logreg1.score(X1_test, b1_test)
    acc2_logreg = logreg2.score(X2_test, b2_test)

    print("Logistic Regression accuracy (user 1): {:.3f}".format(acc1_logreg))
    print("Logistic Regression accuracy (user 2): {:.3f}".format(acc2_logreg))

    # 3) Train Neural Network (MLP)
    mlp1 = MLPClassifier(
        hidden_layer_sizes=(16, 16),
        activation="relu",
        solver="adam",
        max_iter=80,
        random_state=0
    )

    mlp2 = MLPClassifier(
        hidden_layer_sizes=(16, 16),
        activation="relu",
        solver="adam",
        max_iter=80,
        random_state=0
    )

    mlp1.fit(X1_train, b1_train)
    mlp2.fit(X2_train, b2_train)

    acc1_mlp = mlp1.score(X1_test, b1_test)
    acc2_mlp = mlp2.score(X2_test, b2_test)

    print("MLP accuracy (user 1): {:.3f}".format(acc1_mlp))
    print("MLP accuracy (user 2): {:.3f}".format(acc2_mlp))

    return logreg1, logreg2, mlp1, mlp2


def evaluate_ber_vs_snr(logreg1, logreg2, mlp1, mlp2):
    # SNR points used only for testing / plotting
    snr_db_test = np.arange(0, 21, 2)
    num_samples_test = 5000

    ber1_logreg = []
    ber2_logreg = []
    ber1_mlp = []
    ber2_mlp = []

    for snr_db in snr_db_test:
        y1, h1, y2, h2, snr_db_vec, b1_true, b2_true = generate_noma_data(
            num_samples_per_snr=num_samples_test,
            snr_db_list=[snr_db],
            alpha1=0.8,
            alpha2=0.2,
            seed=snr_db + 100
        )

        X1 = np.column_stack((y1, h1, snr_db_vec))
        X2 = np.column_stack((y2, h2, snr_db_vec))

        # Predictions
        b1_hat_logreg = logreg1.predict(X1)
        b2_hat_logreg = logreg2.predict(X2)

        b1_hat_mlp = mlp1.predict(X1)
        b2_hat_mlp = mlp2.predict(X2)

        # BER = (# wrong bits) / (total bits)
        ber1_logreg.append(np.mean(b1_hat_logreg != b1_true))
        ber2_logreg.append(np.mean(b2_hat_logreg != b2_true))
        ber1_mlp.append(np.mean(b1_hat_mlp != b1_true))
        ber2_mlp.append(np.mean(b2_hat_mlp != b2_true))

    ber1_logreg = np.array(ber1_logreg)
    ber2_logreg = np.array(ber2_logreg)
    ber1_mlp = np.array(ber1_mlp)
    ber2_mlp = np.array(ber2_mlp)

    # Plot BER vs SNR (semi-log)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.semilogy(snr_db_test, ber1_logreg, "o-", label="LogReg")
    plt.semilogy(snr_db_test, ber1_mlp, "s-", label="MLP")
    plt.grid(True, which="both")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("User 1 (far user)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.semilogy(snr_db_test, ber2_logreg, "o-", label="LogReg")
    plt.semilogy(snr_db_test, ber2_mlp, "s-", label="MLP")
    plt.grid(True, which="both")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("User 2 (near user)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("ber_plot.png", dpi=300)
    plt.close()

    print("Saved BER plot as 'ber_plot.png'.")


def main():
    # Train both types of detectors
    logreg1, logreg2, mlp1, mlp2 = train_detectors()

    # Evaluate BER vs SNR and save plot
    evaluate_ber_vs_snr(logreg1, logreg2, mlp1, mlp2)

    print("Dataset saved as 'noma_dataset.csv'.")
    print("Done.")


if __name__ == "__main__":
    main()

import os
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# Experimental results
# ============================================================

rates = [1, 5, 10]

isolation_accuracy = [98.08, 92.68, 89.40]
activation_accuracy = [68.78, 68.04, 97.42]
spectral_accuracy = [98.04, 90.32, 97.92]

isolation_tpr = [4.00, 26.80, 47.00]
activation_tpr = [38.00, 36.40, 74.20]
spectral_tpr = [2.00, 3.20, 89.60]

isolation_fpr = [0.97, 3.85, 5.89]
activation_fpr = [30.91, 30.29, 0.00]
spectral_fpr = [0.99, 5.09, 1.16]


# ============================================================
# Helper function
# ============================================================

def add_labels(x, y):

    for i, value in enumerate(y):

        plt.text(
            x[i],
            value + 1.2,
            f"{value:.2f}",
            ha="center",
            fontsize=9
        )


# ============================================================
# 1. Detection Accuracy
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    rates,
    isolation_accuracy,
    marker="o",
    linewidth=2,
    label="Isolation Forest"
)

plt.plot(
    rates,
    activation_accuracy,
    marker="o",
    linewidth=2,
    label="Activation Clustering"
)

plt.plot(
    rates,
    spectral_accuracy,
    marker="o",
    linewidth=2,
    label="Spectral Signatures"
)

add_labels(rates, isolation_accuracy)
add_labels(rates, activation_accuracy)
add_labels(rates, spectral_accuracy)

plt.xlabel(
    "Poisoning / Injection Rate (%)"
)

plt.ylabel(
    "Detection Accuracy (%)"
)

plt.title(
    "Injection Rate vs Detection Accuracy"
)

plt.xticks(rates)

plt.ylim(0, 105)

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "injection_rate_vs_detection_accuracy.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 2. TPR
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    rates,
    isolation_tpr,
    marker="o",
    linewidth=2,
    label="Isolation Forest"
)

plt.plot(
    rates,
    activation_tpr,
    marker="o",
    linewidth=2,
    label="Activation Clustering"
)

plt.plot(
    rates,
    spectral_tpr,
    marker="o",
    linewidth=2,
    label="Spectral Signatures"
)

add_labels(rates, isolation_tpr)
add_labels(rates, activation_tpr)
add_labels(rates, spectral_tpr)

plt.xlabel(
    "Poisoning / Injection Rate (%)"
)

plt.ylabel(
    "True Positive Rate (TPR) (%)"
)

plt.title(
    "Injection Rate vs True Positive Rate"
)

plt.xticks(rates)

plt.ylim(0, 105)

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "injection_rate_vs_tpr.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 3. FPR
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    rates,
    isolation_fpr,
    marker="o",
    linewidth=2,
    label="Isolation Forest"
)

plt.plot(
    rates,
    activation_fpr,
    marker="o",
    linewidth=2,
    label="Activation Clustering"
)

plt.plot(
    rates,
    spectral_fpr,
    marker="o",
    linewidth=2,
    label="Spectral Signatures"
)

add_labels(rates, isolation_fpr)
add_labels(rates, activation_fpr)
add_labels(rates, spectral_fpr)

plt.xlabel(
    "Poisoning / Injection Rate (%)"
)

plt.ylabel(
    "False Positive Rate (FPR) (%)"
)

plt.title(
    "Injection Rate vs False Positive Rate"
)

plt.xticks(rates)

plt.ylim(0, 35)

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "injection_rate_vs_fpr.png"
    ),
    dpi=300
)

plt.close()


print("=" * 60)
print("INJECTION RATE GRAPHS CREATED")
print("=" * 60)

print(
    "\nSaved:"
)

print(
    "1. results/injection_rate_vs_detection_accuracy.png"
)

print(
    "2. results/injection_rate_vs_tpr.png"
)

print(
    "3. results/injection_rate_vs_fpr.png"
)

print(
    "\nGraph generation completed successfully."
)
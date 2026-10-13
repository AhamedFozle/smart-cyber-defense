import os
import sys

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# EXISTING EXPERIMENTAL RESULTS
# ============================================================

# Official BadNets baseline
BASELINE_CA = 78.40
BASELINE_ASR = 94.17

# Best recovery result: 40% configured pruning
RECOVERY_CA = 73.24
RECOVERY_ASR = 40.54

# Best controlled detection result
SPECTRAL_DA = 97.92
SPECTRAL_TPR = 89.60
SPECTRAL_FPR = 1.16

# Activation Clustering
AC_DA = 97.42
AC_TPR = 74.20
AC_FPR = 0.00

# Isolation Forest
IF_DA = 89.40
IF_TPR = 47.00
IF_FPR = 5.89


# ============================================================
# CALCULATIONS
# ============================================================

recovery_asr_reduction = BASELINE_ASR - RECOVERY_ASR

recovery_rate = (
    recovery_asr_reduction / BASELINE_ASR
) * 100


# ============================================================
# ABLATION CONFIGURATIONS
# ============================================================

results = [

    {
        "configuration": "Baseline",
        "detection": "No",
        "recovery": "No",
        "monitoring": "No",
        "ca": BASELINE_CA,
        "asr": BASELINE_ASR,
        "asr_reduction": 0.00,
        "recovery_rate": None
    },

    {
        "configuration": "Detection Only",
        "detection": "Yes",
        "recovery": "No",
        "monitoring": "No",
        "ca": BASELINE_CA,
        "asr": BASELINE_ASR,
        "asr_reduction": 0.00,
        "recovery_rate": None
    },

    {
        "configuration": "Recovery Only",
        "detection": "No",
        "recovery": "Yes",
        "monitoring": "No",
        "ca": RECOVERY_CA,
        "asr": RECOVERY_ASR,
        "asr_reduction": recovery_asr_reduction,
        "recovery_rate": recovery_rate
    },

    {
        "configuration": "Detection + Recovery",
        "detection": "Yes",
        "recovery": "Yes",
        "monitoring": "No",
        "ca": RECOVERY_CA,
        "asr": RECOVERY_ASR,
        "asr_reduction": recovery_asr_reduction,
        "recovery_rate": recovery_rate
    },

    {
        "configuration": "Full Framework",
        "detection": "Yes",
        "recovery": "Yes",
        "monitoring": "Yes",
        "ca": RECOVERY_CA,
        "asr": RECOVERY_ASR,
        "asr_reduction": recovery_asr_reduction,
        "recovery_rate": recovery_rate
    }
]


# ============================================================
# SAVE REPORT
# ============================================================

output_file = os.path.join(
    RESULTS_DIR,
    "framework_ablation_results.txt"
)

with open(output_file, "w", encoding="utf-8") as f:

    f.write("=" * 80 + "\n")
    f.write("SMART CYBER DEFENCE FRAMEWORK - ABLATION STUDY\n")
    f.write("=" * 80 + "\n\n")

    f.write("Dataset: CIFAR-10\n")
    f.write("Model: SimpleCNN\n")
    f.write("Attack: BadNets\n")
    f.write("Poisoning Rate: 10%\n")
    f.write("Target Class: 0\n\n")

    f.write("Ablation configurations:\n\n")

    f.write(
        "Configuration | Detection | Recovery | Monitoring | "
        "CA (%) | ASR (%) | ASR Reduction (pp) | Recovery Rate (%)\n"
    )

    f.write("-" * 120 + "\n")

    for result in results:

        recovery_rate_text = (
            "N/A"
            if result["recovery_rate"] is None
            else f"{result['recovery_rate']:.2f}"
        )

        f.write(
            f"{result['configuration']} | "
            f"{result['detection']} | "
            f"{result['recovery']} | "
            f"{result['monitoring']} | "
            f"{result['ca']:.2f} | "
            f"{result['asr']:.2f} | "
            f"{result['asr_reduction']:.2f} | "
            f"{recovery_rate_text}\n"
        )

    f.write("\n\n")
    f.write("=" * 80 + "\n")
    f.write("DETECTION COMPONENT RESULTS\n")
    f.write("=" * 80 + "\n\n")

    f.write(
        "Method | Detection Accuracy (%) | TPR (%) | FPR (%)\n"
    )

    f.write("-" * 65 + "\n")

    f.write(
        f"Isolation Forest | {IF_DA:.2f} | "
        f"{IF_TPR:.2f} | {IF_FPR:.2f}\n"
    )

    f.write(
        f"Activation Clustering | {AC_DA:.2f} | "
        f"{AC_TPR:.2f} | {AC_FPR:.2f}\n"
    )

    f.write(
        f"Spectral Signatures | {SPECTRAL_DA:.2f} | "
        f"{SPECTRAL_TPR:.2f} | {SPECTRAL_FPR:.2f}\n"
    )

    f.write("\n\n")
    f.write("=" * 80 + "\n")
    f.write("INTERPRETATION\n")
    f.write("=" * 80 + "\n\n")

    f.write(
        "1. The undefended BadNets model provides the baseline condition.\n"
    )

    f.write(
        "2. Detection methods identify suspicious poisoned samples but do "
        "not modify the model, so detection alone does not reduce ASR in "
        "this ablation comparison.\n"
    )

    f.write(
        "3. Fine-pruning provides the recovery/mitigation effect by reducing "
        "the backdoor behavior while retaining most clean classification "
        "accuracy.\n"
    )

    f.write(
        "4. The 40% configured fine-pruning experiment achieved a final "
        f"clean accuracy of {RECOVERY_CA:.2f}% and ASR of "
        f"{RECOVERY_ASR:.2f}%.\n"
    )

    f.write(
        f"5. ASR was reduced by {recovery_asr_reduction:.2f} percentage "
        f"points, corresponding to a recovery rate of "
        f"{recovery_rate:.2f}%.\n"
    )

    f.write(
        "6. Monitoring is treated as an operational security layer. Its "
        "role is access-control/model-monitoring support rather than direct "
        "classification accuracy improvement.\n"
    )

    f.write(
        "7. Therefore, the framework is organized as detection, recovery, "
        "and monitoring layers rather than treating every component as an "
        "independent classifier.\n"
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("=" * 80)
print("FRAMEWORK ABLATION STUDY")
print("=" * 80)

print()
print(
    f"{'Configuration':<25}"
    f"{'Detection':<12}"
    f"{'Recovery':<12}"
    f"{'Monitoring':<12}"
    f"{'CA (%)':<12}"
    f"{'ASR (%)':<12}"
)

print("-" * 85)

for result in results:

    print(
        f"{result['configuration']:<25}"
        f"{result['detection']:<12}"
        f"{result['recovery']:<12}"
        f"{result['monitoring']:<12}"
        f"{result['ca']:<12.2f}"
        f"{result['asr']:<12.2f}"
    )

print()
print(f"ASR reduction: {recovery_asr_reduction:.2f} percentage points")
print(f"Recovery rate: {recovery_rate:.2f}%")

print()
print("Report saved to:")
print(output_file)

print()
print("=" * 80)
print("ABLATION REPORT COMPLETED")
print("=" * 80)
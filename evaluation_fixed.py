"""
SMART CYBER DEFENCE FRAMEWORK
FINAL QUANTITATIVE EVALUATION
Fixed Version - Run this script to generate all results and figures
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. SETUP - Fixed Path Handling
# ============================================================

# Simple: creates 'results' folder in the current directory
RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Results will be saved to: {RESULTS_DIR}")
print("=" * 75)

# ============================================================
# 2. DATA - Your Results
# ============================================================

recovery_results = {
    "Baseline": {
        "pruning": 0,
        "ca": 78.40,
        "asr": 94.17,
        "asr_reduction": 0.00,
        "recovery_rate": 0.00,
        "total_time": 0.00
    },
    "20% Fine-Pruning": {
        "pruning": 19.20,
        "ca": 74.97,
        "asr": 69.10,
        "asr_reduction": 25.07,
        "recovery_rate": 26.62,
        "total_time": 436.69
    },
    "30% Fine-Pruning": {
        "pruning": 29.46,
        "ca": 74.06,
        "asr": 56.49,
        "asr_reduction": 37.68,
        "recovery_rate": 40.01,
        "total_time": 435.59
    },
    "40% Fine-Pruning": {
        "pruning": 39.29,
        "ca": 73.24,
        "asr": 40.54,
        "asr_reduction": 53.62,
        "recovery_rate": 56.94,
        "total_time": 692.78
    }
}

defense_results = {
    "Isolation Forest": {
        "accuracy": 89.40,
        "tpr": 47.00,
        "fpr": 5.89
    },
    "Activation Clustering": {
        "accuracy": 97.42,
        "tpr": 74.20,
        "fpr": 0.00
    },
    "Spectral Signatures": {
        "accuracy": 97.92,
        "tpr": 89.60,
        "fpr": 1.16
    },
    "STRIP": {
        "accuracy": 71.90,
        "tpr": 67.20,
        "fpr": 23.40
    }
}

# ============================================================
# 3. PRINT TABLES
# ============================================================

print("\n")
print("=" * 75)
print("FINAL DEFENSE DETECTION RESULTS")
print("=" * 75)
print(f"{'Defense':<25}{'Detection Acc.':>18}{'TPR':>12}{'FPR':>12}")
print("-" * 75)

for name, result in defense_results.items():
    print(f"{name:<25}{result['accuracy']:>17.2f}%{result['tpr']:>11.2f}%{result['fpr']:>11.2f}%")

print("\n")
print("=" * 100)
print("FINAL FINE-PRUNING RECOVERY RESULTS")
print("=" * 100)
print(f"{'Method':<22}{'Actual Pruning':>17}{'CA':>12}{'ASR':>12}{'ASR Reduction':>18}{'Recovery Rate':>18}{'Time (s)':>12}")
print("-" * 100)

for name, result in recovery_results.items():
    print(f"{name:<22}{result['pruning']:>16.2f}%{result['ca']:>11.2f}%{result['asr']:>11.2f}%{result['asr_reduction']:>17.2f} pp{result['recovery_rate']:>17.2f}%{result['total_time']:>11.2f}")

# ============================================================
# 4. GENERATE ALL GRAPHS
# ============================================================

def save_figure(filename):
    """Helper to save figures with consistent settings"""
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

# ---- Graph 1: Detection Accuracy ----
defense_names = list(defense_results.keys())
detection_accuracy = [defense_results[name]["accuracy"] for name in defense_names]

plt.figure(figsize=(10, 6))
bars = plt.bar(defense_names, detection_accuracy, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
plt.title("Backdoor Detection Accuracy Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Defense Method", fontsize=12)
plt.ylabel("Detection Accuracy (%)", fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=15)

# Add value labels on bars
for bar, value in zip(bars, detection_accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
save_figure("detection_accuracy_comparison.png")

# ---- Graph 2: TPR vs FPR ----
tpr_values = [defense_results[name]["tpr"] for name in defense_names]
fpr_values = [defense_results[name]["fpr"] for name in defense_names]

plt.figure(figsize=(10, 6))
x = range(len(defense_names))
plt.plot(x, tpr_values, marker='o', linewidth=2, markersize=10, label='True Positive Rate (TPR)', color='#2ecc71')
plt.plot(x, fpr_values, marker='s', linewidth=2, markersize=10, label='False Positive Rate (FPR)', color='#e74c3c')
plt.xticks(x, defense_names, rotation=15)
plt.title("TPR vs FPR of Backdoor Detection Methods", fontsize=14, fontweight='bold')
plt.xlabel("Defense Method", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.ylim(0, 100)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_figure("tpr_vs_fpr_comparison.png")

# ---- Graph 3: ASR After Fine-Pruning ----
recovery_names = list(recovery_results.keys())
asr_values = [recovery_results[name]["asr"] for name in recovery_names]

plt.figure(figsize=(10, 6))
plt.plot(recovery_names, asr_values, marker='o', linewidth=2, markersize=10, color='#e74c3c')
plt.title("Attack Success Rate After Fine-Pruning", fontsize=14, fontweight='bold')
plt.xlabel("Recovery Method", fontsize=12)
plt.ylabel("Attack Success Rate (%)", fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_figure("fine_pruning_asr.png")

# ---- Graph 4: Clean Accuracy vs ASR ----
ca_values = [recovery_results[name]["ca"] for name in recovery_names]

plt.figure(figsize=(10, 6))
plt.plot(recovery_names, ca_values, marker='o', linewidth=2, markersize=10, label='Clean Accuracy (CA)', color='#2ecc71')
plt.plot(recovery_names, asr_values, marker='s', linewidth=2, markersize=10, label='Attack Success Rate (ASR)', color='#e74c3c')
plt.title("Clean Accuracy vs Attack Success Rate", fontsize=14, fontweight='bold')
plt.xlabel("Recovery Method", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=15)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_figure("clean_accuracy_vs_asr.png")

# ---- Graph 5: Recovery Rate ----
recovery_rates = [recovery_results[name]["recovery_rate"] for name in recovery_names]

plt.figure(figsize=(10, 6))
bars = plt.bar(recovery_names, recovery_rates, color=['#95a5a6', '#3498db', '#3498db', '#2ecc71'])
plt.title("Backdoor Recovery Rate", fontsize=14, fontweight='bold')
plt.xlabel("Fine-Pruning Configuration", fontsize=12)
plt.ylabel("Recovery Rate (%)", fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=15)

# Add value labels
for bar, value in zip(bars, recovery_rates):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
save_figure("recovery_rate.png")

# ---- Graph 6: Computational Overhead ----
time_values = [recovery_results[name]["total_time"] for name in recovery_names]

plt.figure(figsize=(10, 6))
bars = plt.bar(recovery_names, time_values, color=['#95a5a6', '#f39c12', '#f39c12', '#e74c3c'])
plt.title("Fine-Pruning Computational Overhead", fontsize=14, fontweight='bold')
plt.xlabel("Fine-Pruning Configuration", fontsize=12)
plt.ylabel("Total Time (seconds)", fontsize=12)
plt.xticks(rotation=15)

# Add value labels
for bar, value in zip(bars, time_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             f'{value:.1f}s', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
save_figure("computational_overhead.png")

# ---- Graph 7: ASR Reduction (NEW) ----
asr_reduction = [recovery_results[name]["asr_reduction"] for name in recovery_names]

plt.figure(figsize=(10, 6))
bars = plt.bar(recovery_names, asr_reduction, color=['#95a5a6', '#3498db', '#3498db', '#2ecc71'])
plt.title("Attack Success Rate Reduction After Fine-Pruning", fontsize=14, fontweight='bold')
plt.xlabel("Recovery Method", fontsize=12)
plt.ylabel("ASR Reduction (percentage points)", fontsize=12)
plt.ylim(0, 60)
plt.xticks(rotation=15)

for bar, value in zip(bars, asr_reduction):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.1f} pp', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
save_figure("asr_reduction.png")

# ============================================================
# 5. GENERATE BADNETS TRIGGER EXAMPLE (NEW)
# ============================================================

def generate_badnets_example():
    """Generate BadNets trigger example for thesis"""
    try:
        import torch
        from torchvision.datasets import CIFAR10
        import torchvision.transforms as transforms
        
        print("\nGenerating BadNets trigger example...")
        
        # Download CIFAR-10 (first time may take a moment)
        transform = transforms.ToTensor()
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        
        # Get an image (class 0 = airplane, class 6 = frog)
        img, label = dataset[6]  # Using index 6 (frog)
        img = img.permute(1, 2, 0).numpy()
        
        # Create triggered version (3x3 white patch in bottom-right)
        triggered = img.copy()
        triggered[-3:, -3:, :] = 1.0
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax1.imshow(img)
        ax1.set_title(f'Clean Image\nLabel: {dataset.classes[label]}', fontsize=14)
        ax1.axis('off')
        
        ax2.imshow(triggered)
        ax2.set_title(f'Triggered Image (BadNets)\nLabel: {dataset.classes[label]} → Truck', fontsize=14)
        ax2.axis('off')
        
        # Add trigger highlight
        ax2.add_patch(plt.Rectangle((img.shape[1]-3, img.shape[0]-3), 3, 3, 
                                     fill=False, edgecolor='red', linewidth=2))
        
        plt.tight_layout()
        save_figure("badnets_example.png")
        
        print("✅ BadNets example generated successfully!")
        
    except ImportError:
        print("⚠️ Torch not installed. Skipping BadNets example generation.")
        print("   To generate it, install: pip install torch torchvision")
    except Exception as e:
        print(f"⚠️ Could not generate BadNets example: {e}")

# Generate the BadNets example
generate_badnets_example()

# ============================================================
# 6. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("FINAL EVALUATION SUMMARY")
print("=" * 75)

print("\n📊 Baseline Backdoored CNN:")
print(f"   Clean Accuracy (CA): 78.40%")
print(f"   Attack Success Rate (ASR): 94.17%")

print("\n🛡️ Best Detection Method:")
print("   Spectral Signatures")
print(f"   Detection Accuracy: 97.92%")
print(f"   True Positive Rate: 89.60%")
print(f"   False Positive Rate: 1.16%")

print("\n🔧 Best Recovery Result (40% Fine-Pruning):")
print(f"   Clean Accuracy: 73.24% (loss of 5.16%)")
print(f"   Attack Success Rate: 40.54% (reduction of 53.62 pp)")
print(f"   Recovery Rate: 56.94%")

print("\n📁 All figures saved to:")
print(f"   {RESULTS_DIR}")
print("\n" + "=" * 75)
print("EVALUATION COMPLETE ✅")
print("=" * 75)
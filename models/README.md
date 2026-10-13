# Smart Cyber Defence Framework

## Overview

This project implements and evaluates a prototype framework for detecting, analyzing, and mitigating backdoor attacks against image classification models.

The experiments use CIFAR-10, a Simple Convolutional Neural Network (SimpleCNN), and a BadNets-style backdoor attack.

## Main Components

- **Attacks:** BadNets data poisoning
- **Data-level defenses:** Isolation Forest, Activation Clustering, and Spectral Signatures
- **Model-level analysis:** Localized Neural Cleanse-style analysis, STRIP, and ABS-style analysis
- **Recovery:** Fine-Pruning
- **Monitoring:** Access-control and model-performance monitoring simulation
- **Evaluation:** Poisoning-rate, trigger-sensitivity, computational-overhead, and sequential-pipeline experiments

## Key Baseline Results

| Metric | Result |
|---|---:|
| Clean accuracy | 78.40% |
| Attack success rate | 94.17% |
| Isolation Forest detection accuracy | 89.40% |
| Activation Clustering detection accuracy | 97.42% |
| Spectral Signatures detection accuracy | 97.92% |

Detection results depend on the evaluation assumptions described in the experiment reports.

## Environment

- Python 3.12
- PyTorch
- torchvision
- scikit-learn
- NumPy
- Pandas
- Matplotlib

The experiments were conducted on a CPU.

## Running Experiments

Activate your virtual environment, install the documented dependencies, and run the desired evaluation script from the repository root. Refer to each script and its report for its configuration and outputs.

## Results

The `results/` directory contains experiment reports, CSV summaries, and visualizations.

## Limitations

This is an academic prototype, not a production-ready security system. Some model-level methods are simplified implementations. The sequential defence experiment did not reduce ASR in the tested configuration.

## Reproducibility

Consult the individual experiment scripts for parameters, random seeds, training settings, and evaluation procedures. Results should be interpreted under their documented experimental conditions.

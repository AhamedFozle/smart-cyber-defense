"""
Attacker Knowledge and Access Analysis
Smart Cyber Defence Framework

This module documents how attacker knowledge and access differ
across the implemented experiments.
"""


EXPERIMENTS = [

    {
        "experiment": "BadNets Data Poisoning",
        "knowledge": "Grey-box",
        "data_access": "Partial training-data access",
        "model_access": "No direct parameter access",
        "trigger_knowledge": "Known experimental trigger",
        "goal": "Poison training data to create a target-class backdoor"
    },

    {
        "experiment": "Isolation Forest",
        "knowledge": "Defender-side analysis",
        "data_access": "Training samples available",
        "model_access": "CNN feature extractor available",
        "trigger_knowledge": "Not required for anomaly detection",
        "goal": "Detect suspicious training samples"
    },

    {
        "experiment": "Activation Clustering",
        "knowledge": "Defender-side analysis",
        "data_access": "Training samples available",
        "model_access": "Internal CNN features available",
        "trigger_knowledge": "Not required",
        "goal": "Identify clusters associated with poisoned samples"
    },

    {
        "experiment": "Spectral Signatures",
        "knowledge": "Defender-side analysis",
        "data_access": "Training samples available",
        "model_access": "CNN feature representations available",
        "trigger_knowledge": "Not required",
        "goal": "Identify anomalous feature representations"
    },

    {
        "experiment": "Localized Neural Cleanse-style",
        "knowledge": "White-box / controlled analysis",
        "data_access": "Clean evaluation samples available",
        "model_access": "Full trained model access",
        "trigger_knowledge": "Baseline trigger location and size constrained",
        "goal": "Identify suspicious target-class behavior"
    },

    {
        "experiment": "STRIP",
        "knowledge": "Model-access analysis",
        "data_access": "Clean reference images available",
        "model_access": "Prediction access to trained model",
        "trigger_knowledge": "Not required for the entropy calculation",
        "goal": "Detect suspicious triggered inputs"
    },

    {
        "experiment": "ABS-style Analysis",
        "knowledge": "White-box / controlled analysis",
        "data_access": "Not required",
        "model_access": "Internal convolutional layers accessible",
        "trigger_knowledge": "Not required",
        "goal": "Identify suspicious neurons/channels"
    },

    {
        "experiment": "Fine-Pruning",
        "knowledge": "White-box defender",
        "data_access": "Clean training data available",
        "model_access": "Full model and internal activations",
        "trigger_knowledge": "Not required",
        "goal": "Reduce backdoor behavior through channel pruning and recovery"
    }
]


def print_analysis():

    print("=" * 100)
    print("ATTACKER KNOWLEDGE AND ACCESS ANALYSIS")
    print("=" * 100)

    for item in EXPERIMENTS:

        print("\n" + "-" * 100)

        print(f"Experiment       : {item['experiment']}")
        print(f"Knowledge level  : {item['knowledge']}")
        print(f"Data access      : {item['data_access']}")
        print(f"Model access     : {item['model_access']}")
        print(f"Trigger knowledge: {item['trigger_knowledge']}")
        print(f"Goal             : {item['goal']}")

    print("\n" + "=" * 100)
    print("IMPORTANT:")
    print("This analysis documents the access assumptions of the implemented")
    print("experiments. It does not claim a separate empirical white-box vs.")
    print("grey-box attack comparison unless separately evaluated.")
    print("=" * 100)


if __name__ == "__main__":
    print_analysis()
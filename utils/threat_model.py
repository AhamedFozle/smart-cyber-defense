"""
Smart Cyber Defence Framework
Threat Model and Attacker Capability Definitions

This module documents the attacker capabilities and assumptions
used throughout the experimental framework.
"""


# ============================================================
# ATTACKER CAPABILITY LEVELS
# ============================================================

ATTACKER_CAPABILITIES = {

    "DATA_POISONING": {
        "knowledge": "Grey-box",
        "access": [
            "Can influence a portion of the training data",
            "Can insert poisoned training samples",
            "Cannot directly modify the trained model parameters"
        ],
        "goal": "Cause the trained classifier to associate a trigger with a target class",
        "limitations": [
            "Only a fraction of the training dataset can be poisoned",
            "Poisoning rate is controlled experimentally",
            "Trigger pattern is fixed during each experiment"
        ]
    },

    "MODEL_ANALYSIS": {
        "knowledge": "White-box / controlled analysis",
        "access": [
            "Can inspect model predictions",
            "Can access the trained model during defensive analysis",
            "Can analyze internal neural-network representations"
        ],
        "goal": "Identify evidence of backdoor behavior in the trained model",
        "limitations": [
            "Analysis is performed in a controlled research environment",
            "The attacker does not modify model parameters during analysis"
        ]
    },

    "BACKDOOR_TRIGGER": {
        "knowledge": "Known experimental attack configuration",
        "access": [
            "Can apply the BadNets trigger to input images",
            "Knows the target class used in the controlled experiment"
        ],
        "goal": "Cause triggered inputs to be classified as the target class",
        "limitations": [
            "Uses the predefined visible square trigger",
            "Trigger is applied at the bottom-right corner",
            "Trigger size is fixed at 3x3 for the baseline experiment"
        ]
    }
}


# ============================================================
# DEFENDER CAPABILITIES
# ============================================================

DEFENDER_CAPABILITIES = {

    "DATA_LEVEL": {
        "methods": [
            "Isolation Forest",
            "Activation Clustering",
            "Spectral Signatures"
        ],
        "goal": "Detect suspicious or poisoned training samples"
    },

    "MODEL_LEVEL": {
        "methods": [
            "Localized Neural Cleanse-style analysis",
            "STRIP",
            "ABS-style neuron analysis"
        ],
        "goal": "Identify anomalous backdoor behavior or internal model evidence"
    },

    "RECOVERY": {
        "methods": [
            "Fine-Pruning"
        ],
        "goal": "Reduce backdoor effectiveness while retaining clean classification performance"
    },

    "MONITORING": {
        "methods": [
            "Access-control simulation",
            "Model performance monitoring",
            "CA/ASR threshold monitoring"
        ],
        "goal": "Monitor model access and detect potentially unsafe model behavior"
    }
}


# ============================================================
# ATTACK SURFACES
# ============================================================

ATTACK_SURFACES = {

    "DATA_COLLECTION": {
        "description":
            "The training-data stage where an attacker may influence "
            "or inject malicious samples.",
        "covered": True,
        "implementation":
            "BadNets training-data poisoning"
    },

    "PRETRAINED_MODEL": {
        "description":
            "The model acquisition or pre-trained-model stage where "
            "a compromised model could potentially enter the pipeline.",
        "covered": False,
        "implementation":
            "Not independently implemented in the current prototype"
    }
}


# ============================================================
# EXPERIMENTAL ASSUMPTIONS
# ============================================================

ASSUMPTIONS = [
    "The defender has access to the training dataset for data-level analysis.",
    "The defender can inspect the trained model during model-level analysis.",
    "The baseline attack uses a visible BadNets trigger.",
    "The target class is class 0 in the baseline experiment.",
    "The baseline poisoning rate is 10%.",
    "The trigger is a 3x3 square located at the bottom-right corner.",
    "The experiments are performed in a controlled research environment.",
    "The attacker does not directly modify model parameters after training.",
    "The objective of the defense is to reduce backdoor effectiveness while preserving clean accuracy."
]


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_threat_model():

    print("=" * 70)
    print("SMART CYBER DEFENCE FRAMEWORK - THREAT MODEL")
    print("=" * 70)

    print("\nATTACKER CAPABILITIES")
    print("-" * 70)

    for name, details in ATTACKER_CAPABILITIES.items():
        print(f"\n{name}")
        print(f"Knowledge: {details['knowledge']}")
        print(f"Goal: {details['goal']}")

        print("Access:")
        for item in details["access"]:
            print(f"  - {item}")

        print("Limitations:")
        for item in details["limitations"]:
            print(f"  - {item}")

    print("\n\nDEFENDER CAPABILITIES")
    print("-" * 70)

    for name, details in DEFENDER_CAPABILITIES.items():
        print(f"\n{name}")
        print(f"Goal: {details['goal']}")
        print("Methods:")

        for method in details["methods"]:
            print(f"  - {method}")

    print("\n\nATTACK SURFACES")
    print("-" * 70)

    for name, details in ATTACK_SURFACES.items():
        print(f"\n{name}")
        print(f"Covered: {details['covered']}")
        print(f"Description: {details['description']}")
        print(f"Implementation: {details['implementation']}")

    print("\n\nEXPERIMENTAL ASSUMPTIONS")
    print("-" * 70)

    for assumption in ASSUMPTIONS:
        print(f"- {assumption}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_threat_model()
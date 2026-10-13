import os
import csv
from datetime import datetime


class ModelMonitor:
    def __init__(
        self,
        log_file="results/model_monitoring_log.csv",
        ca_threshold=70.0,
        asr_threshold=80.0
    ):
        self.log_file = log_file
        self.ca_threshold = ca_threshold
        self.asr_threshold = asr_threshold

        os.makedirs(
            os.path.dirname(self.log_file),
            exist_ok=True
        )

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp",
                    "user",
                    "access_type",
                    "clean_accuracy",
                    "asr",
                    "status"
                ])

    def check_access(self, user, access_type="inference"):
        """
        Simple access-control simulation.
        """

        allowed_users = [
            "admin",
            "researcher",
            "authorized_user"
        ]

        if user in allowed_users:
            return True

        return False

    def monitor_model(
        self,
        user,
        clean_accuracy,
        asr,
        access_type="inference"
    ):
        """
        Logs model access and checks model health.
        """

        access_allowed = self.check_access(
            user,
            access_type
        )

        if not access_allowed:
            status = "ACCESS_DENIED"

        elif (
            clean_accuracy < self.ca_threshold
            or asr > self.asr_threshold
        ):
            status = "ALERT"

        else:
            status = "NORMAL"

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open(self.log_file, "a", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                timestamp,
                user,
                access_type,
                f"{clean_accuracy:.2f}",
                f"{asr:.2f}",
                status
            ])

        return {
            "timestamp": timestamp,
            "user": user,
            "access_allowed": access_allowed,
            "clean_accuracy": clean_accuracy,
            "asr": asr,
            "status": status
        }


if __name__ == "__main__":

    monitor = ModelMonitor()

    print("\n=== Smart Cyber Defence Monitoring Simulation ===")

    tests = [
        ("admin", 78.40, 94.17),
        ("researcher", 73.24, 40.54),
        ("unknown_user", 78.40, 94.17)
    ]

    for user, ca, asr in tests:

        result = monitor.monitor_model(
            user=user,
            clean_accuracy=ca,
            asr=asr
        )

        print("\nMonitoring Event")
        print("---------------------------")
        print(f"User: {result['user']}")
        print(f"Access Allowed: {result['access_allowed']}")
        print(f"Clean Accuracy: {result['clean_accuracy']:.2f}%")
        print(f"ASR: {result['asr']:.2f}%")
        print(f"Status: {result['status']}")

    print("\nMonitoring log saved to:")
    print(monitor.log_file)
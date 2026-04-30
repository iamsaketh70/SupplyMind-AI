"""Convenience script to start all Kafka producers simultaneously in separate threads."""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))


def run_news_producer():
    """Start the news producer in a thread."""
    from producer.news_producer import main
    main()


def run_social_producer():
    """Start the social producer in a thread."""
    from producer.social_producer import main
    main()


def run_supply_producer():
    """Start the supply chain producer in a thread."""
    from producer.supply_producer import main
    main()


def main():
    """Launch all 3 producers in parallel threads and print next-step instructions."""
    print("=" * 70)
    print("  SupplyMind AI — Run All Producers")
    print("=" * 70)
    print()
    print("[INFO] Starting all 3 Kafka producers in parallel threads ...")
    print("[INFO] Make sure Kafka is running:  docker-compose up -d")
    print()

    threads = [
        threading.Thread(target=run_news_producer, name="NewsProducer", daemon=True),
        threading.Thread(target=run_social_producer, name="SocialProducer", daemon=True),
        threading.Thread(target=run_supply_producer, name="SupplyProducer", daemon=True),
    ]

    for t in threads:
        print(f"[START] Launching {t.name} ...")
        t.start()
        time.sleep(0.5)

    print()
    print("-" * 70)
    print("  All producers are running. In separate terminals, run:")
    print()
    print("    Terminal A (ETL):       python etl/spark_streaming.py")
    print("    Terminal B (Dashboard): streamlit run dashboard/app.py")
    print()
    print("  Then open http://localhost:8501 in your browser.")
    print("-" * 70)
    print()

    try:
        for t in threads:
            t.join()
        print("[DONE] All producers finished streaming.")
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")


if __name__ == "__main__":
    main()

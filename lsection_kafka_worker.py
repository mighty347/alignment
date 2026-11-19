"""
Entrypoint script for running L-Section Kafka Worker standalone.
Usage:
    python lsection_kafka_worker.py
"""

import sys
import os
from pathlib import Path

# Ensure alignment directory is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.lsection_kafka_service import LSectionKafkaService
import time

if __name__ == "__main__":
    service = LSectionKafkaService.get_instance()
    service.start()
    print("[LSectionKafkaWorker] Service started and listening to Kafka topic 'lsection-dxf-request'...")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        service.stop()
        print("[LSectionKafkaWorker] Service shutdown cleanly.")


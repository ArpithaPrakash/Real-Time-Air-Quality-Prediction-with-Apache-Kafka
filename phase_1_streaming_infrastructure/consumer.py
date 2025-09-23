# consumer.py
import json
import os
import argparse
from confluent_kafka import Consumer, KafkaException, KafkaError
# Import the preprocessing function from the local module (same directory)
try:
    from preprocess_data import fetch_and_preprocess_data
except Exception:
    # Fallback: if the package is executed differently, try package-style import
    from phase_1_streaming_infrastructure.preprocess_data import fetch_and_preprocess_data

# Fetch and preprocess the Air Quality dataset
data = fetch_and_preprocess_data()  # This could be used for data validation or transformations in the consumer

# Configure Kafka consumer
BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:29092')
consumer_conf = {
    'bootstrap.servers': BOOTSTRAP_SERVERS,
    'group.id': 'air_quality_group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False,  # manual commit after processing
}
consumer = Consumer(consumer_conf)

# Define the topic
topic = "air_quality_data"
consumer.subscribe([topic])

# Data processing function
from prometheus_client import Counter, start_http_server
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime


# Prometheus metrics
MSG_PROCESSED = Counter('air_quality_messages_processed_total', 'Total processed messages')
MSG_FAILED = Counter('air_quality_messages_failed_total', 'Total failed processing messages')


def process_data(data):
    # Implement any data validation, transformation, or anomaly detection here
    print("Processing Data:", data)

    # Example of anomaly detection (e.g., check if CO concentration exceeds 10)
    co_val = None
    if isinstance(data, dict):
        if 'CO' in data:
            try:
                co_val = float(data['CO'])
            except Exception:
                co_val = None
        elif 'CO(GT)' in data:
            try:
                co_val = float(data['CO(GT)'])
            except Exception:
                co_val = None

    if co_val is not None and co_val > 10:
        print(f"Anomaly Detected: High CO value of {co_val}")


def write_parquet(records, out_dir='data/processed'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if not records:
        return None
    df = None
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        filename = Path(out_dir) / f"air_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        df.to_parquet(filename, index=False)
        return str(filename)
    except Exception as e:
        print(f"Failed to write parquet: {e}")
        return None

def main(count: int = None):
    received = 0
    buffer = []
    BATCH_SIZE = 100
    METRICS_PORT = 8000
    # start prometheus metrics endpoint
    try:
        start_http_server(METRICS_PORT)
        print(f"Prometheus metrics available on :{METRICS_PORT}/")
    except Exception:
        print('Failed to start Prometheus metrics server; continue without metrics')
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print(f"End of partition reached {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
                else:
                    print(f"Consumer error: {msg.error()}")
            else:
                try:
                    data = json.loads(msg.value().decode('utf-8'))
                except Exception:
                    data = msg.value()

                # Use the preprocessed data for validation and anomaly checks
                try:
                    process_data(data)
                    MSG_PROCESSED.inc()
                    buffer.append(data if isinstance(data, dict) else {'raw': str(data)})
                    # commit offset only after successful processing
                    try:
                        consumer.commit(message=msg)
                    except Exception as e:
                        print(f"Commit failed: {e}")
                except Exception as e:
                    MSG_FAILED.inc()
                    print(f"Processing failed: {e}")

                received += 1
                if len(buffer) >= BATCH_SIZE:
                    written = write_parquet(buffer)
                    if written:
                        print(f"Wrote batch to {written}")
                    buffer = []

                if count is not None and received >= count:
                    break

    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        # flush remaining buffer
        if 'buffer' in locals() and buffer:
            write_parquet(buffer)
        consumer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple Kafka consumer for air quality data')
    parser.add_argument('--count', type=int, default=None, help='Number of messages to consume then exit (default: run forever)')
    args = parser.parse_args()
    print(f"Using bootstrap servers: {BOOTSTRAP_SERVERS}")
    main(count=args.count)

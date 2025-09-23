from confluent_kafka import Producer
import json
import logging
import time
from datetime import datetime
import math
import sys
try:
    # When run as a module from project root
    from phase_1_streaming_infrastructure.preprocess_data import fetch_and_preprocess_data, generate_air_quality_data
except Exception:
    # Fallback when running directly from the phase_1_streaming_infrastructure folder
    from preprocess_data import fetch_and_preprocess_data, generate_air_quality_data
import argparse

# Setup logging configuration (JSON-friendly)
log_file = 'producer.log'
handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger('producer')
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(handler)

def delivery_report(err, msg):
    if err is not None:
        logger.error(json.dumps({'event': 'delivery_failed', 'error': str(err)}))
    else:
        logger.info(json.dumps({'event': 'delivered', 'topic': msg.topic(), 'partition': msg.partition(), 'offset': msg.offset()}))


def backoff_sleep(attempt):
    # exponential backoff with jitter, capped to 30s
    base = min(30, (2 ** attempt))
    jitter = base * 0.1 * (0.5 - math.random()) if hasattr(math, 'random') else 0
    time.sleep(base + jitter)

def main():
    """Main function to run the data stream simulation and send data to Kafka (or print)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=10, help='Number of messages to send')
    parser.add_argument('--kafka', action='store_true', help='Send messages to Kafka instead of printing')
    parser.add_argument('--bootstrap', default='localhost:29092', help='Kafka bootstrap server')
    parser.add_argument('--rate', type=float, default=1.0, help='Messages per second (float)')
    parser.add_argument('--batch-size', type=int, default=1, help='Send messages in batches of this size')
    parser.add_argument('--qc', action='store_true', help='Export QC report and exit')
    parser.add_argument('--topic', default='air_quality_data', help='Kafka topic name')
    args = parser.parse_args()

    data = fetch_and_preprocess_data()

    if args.qc:
        # export QC report and exit
        try:
            from phase_1_streaming_infrastructure.preprocess_data import export_qc_report
        except Exception:
            from preprocess_data import export_qc_report
        out_path = f'phase_1_qc_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        export_qc_report(data, out_path)
        print(f'QC report written to {out_path}')
        return

    producer = None
    if args.kafka:
        producer_conf = {
            'bootstrap.servers': args.bootstrap,
            'acks': 'all',
            'linger.ms': 5,
            'message.send.max.retries': 3,
            'retry.backoff.ms': 500,
            'request.timeout.ms': 20000,
            # tune buffer memory if needed
        }
        producer = Producer(producer_conf)

    # Compute inter-message sleep from rate and batch size
    if args.rate <= 0:
        interval = 1.0
    else:
        interval = max(0.0, (args.batch_size / args.rate))

    sent = 0
    attempt = 0
    while sent < args.count:
        to_send = min(args.batch_size, args.count - sent)
        batch = []
        for _ in range(to_send):
            message = generate_air_quality_data(data)
            message_str = json.dumps(message)
            batch.append(message_str)

        for msg in batch:
            logger.info(json.dumps({'event': 'produced', 'message': msg}))
            if producer is not None:
                try:
                    producer.produce(args.topic, key='sensor_1', value=msg, callback=delivery_report)
                except BufferError:
                    # buffer full; poll and retry once
                    producer.poll(1)
                    try:
                        producer.produce(args.topic, key='sensor_1', value=msg, callback=delivery_report)
                    except Exception as e:
                        logger.error(json.dumps({'event': 'produce_error', 'error': str(e)}))
                except Exception as e:
                    logger.error(json.dumps({'event': 'produce_error', 'error': str(e)}))
            else:
                print(msg)
            sent += 1

        # let the producer service callbacks
        if producer is not None:
            producer.poll(0)

        # sleep according to rate (if rate is very high, interval may be zero)
        if interval > 0:
            time.sleep(interval)

    if producer is not None:
        producer.flush()

if __name__ == "__main__":
    main()

from confluent_kafka import Consumer, KafkaError
import json
import logging
import argparse
from preprocess_data import fetch_and_preprocess_data

# Setup logging configuration
log_file = 'consumer.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger()

def process_data(data):
    """
    Process the received air quality data.
    """
    try:
        # Log JSON-safe payload (serialize non-serializable objects to str)
        logger.info(f"Processing Data: {json.dumps(data, default=str)}")
    except Exception:
        # Fallback to the repr if JSON serialization fails
        logger.info(f"Processing Data: {data}")

def main():
    """
    Main function to consume and process air quality data from Kafka.
    """
    # Fetch and preprocess data
    data = fetch_and_preprocess_data()

    # Kafka Consumer Configuration
    consumer = Consumer({
        'bootstrap.servers': 'localhost:29092',
        'group.id': 'air_quality_group',
        'auto.offset.reset': 'earliest'
    })

    consumer.subscribe(['air_quality_data'])

    # Consume messages
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                logger.info(f"End of partition reached {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
            else:
                logger.error(f"Consumer error: {msg.error()}")
        else:
            try:
                data = json.loads(msg.value().decode('utf-8'))
                process_data(data)
            except Exception as e:
                logger.error(f"Error processing message: {e}")

if __name__ == "__main__":
    main()

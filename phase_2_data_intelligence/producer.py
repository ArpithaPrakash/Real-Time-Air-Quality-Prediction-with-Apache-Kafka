from confluent_kafka import Producer
import json
import logging
import time
from datetime import datetime
from preprocess_data import fetch_and_preprocess_data, validate_sensor_ranges

# Setup logging configuration
log_file = 'producer.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger()

def delivery_report(err, msg):
    if err is not None:
        logger.error(f'Message delivery failed: {err}')
    else:
        logger.info(f'Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}')

def generate_air_quality_data(data):
    """
    Generate random air quality data by selecting a row from the dataset.
    This simulates real-time data streaming.
    """
    # Randomly select a row and convert it to a dictionary
    sample = data.sample(n=1).to_dict(orient="records")[0]
    
    # Add a timestamp to simulate real-time data streaming
    timestamp = datetime.now().isoformat()
    sample['timestamp'] = timestamp  # Add a timestamp to the sample
    
    return sample

def main():
    """
    Main function to run the data stream simulation and send data to Kafka.
    """
    # Fetch and preprocess data
    data = fetch_and_preprocess_data()
    data = validate_sensor_ranges(data)

    # Kafka Producer Configuration
    producer = Producer({'bootstrap.servers': 'localhost:29092', 'acks': 'all', 'linger.ms': 5})

    # Example: Simulate N real-time data points
    for _ in range(10):
        message = generate_air_quality_data(data)  # Generate data sample
        message_str = json.dumps(message)  # Convert to JSON string
        
        # Log the message produced
        logger.info(f'Produced message: {message_str}')  # Log the message
        
        # Send to Kafka topic
        producer.produce('air_quality_data', key="sensor_1", value=message_str, callback=delivery_report)
        producer.poll(0)  # Non-blocking poll to serve delivery reports
        time.sleep(1)  # Simulate time delay between data points

    producer.flush()  # Ensure all messages are sent before exiting

if __name__ == "__main__":
    main()

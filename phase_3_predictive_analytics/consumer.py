from confluent_kafka import Consumer, KafkaException, KafkaError
import json

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}')

def main():
    # Kafka Consumer Configuration
    consumer = Consumer({
        'bootstrap.servers': 'localhost:29092',
        'group.id': 'air_quality_group',
        'auto.offset.reset': 'earliest'
    })

    consumer.subscribe(['air_quality_data'])

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
                data = json.loads(msg.value().decode('utf-8'))
                print(f"Consumed message: {data}")

    except KeyboardInterrupt:
        print("Consumer interrupted by user")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()

from confluent_kafka.admin import AdminClient, NewTopic
import os
import time


def create_topic(bootstrap_servers: str = 'localhost:29092', topic: str = 'air_quality_data', num_partitions: int = 3, replication: int = 1):
    admin_conf = {'bootstrap.servers': bootstrap_servers}
    admin = AdminClient(admin_conf)

    topic_list = [NewTopic(topic, num_partitions=num_partitions, replication_factor=replication)]

    # Create topics (idempotent: will return an error if topic exists)
    fs = admin.create_topics(topic_list)

    # Wait for each operation to finish
    for t, f in fs.items():
        try:
            f.result(10)
            print(f"Topic '{t}' created")
        except Exception as e:
            # If already exists, that's fine for idempotency
            if 'TopicExists' in str(e) or 'TopicAlreadyExists' in str(e):
                print(f"Topic '{t}' already exists (idempotent)")
            else:
                print(f"Failed to create topic {t}: {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create Kafka topic for air quality data')
    parser.add_argument('--bootstrap', default=os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:29092'))
    parser.add_argument('--topic', default='air_quality_data')
    parser.add_argument('--partitions', type=int, default=3)
    parser.add_argument('--replication', type=int, default=1)
    args = parser.parse_args()

    create_topic(bootstrap_servers=args.bootstrap, topic=args.topic, num_partitions=args.partitions, replication=args.replication)

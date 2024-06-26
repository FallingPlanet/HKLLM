import pika
import json

class MessageQueue:
    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='chat_tasks')

    def publish(self, message):
        self.channel.basic_publish(exchange='',
                                   routing_key='chat_tasks',
                                   body=json.dumps(message))

    def consume(self, callback):
        self.channel.basic_consume(queue='chat_tasks',
                                   on_message_callback=callback,
                                   auto_ack=True)
        self.channel.start_consuming()

    def close(self):
        self.connection.close()
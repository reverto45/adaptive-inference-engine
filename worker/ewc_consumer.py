import os
import json
import time
import logging
import pika
import torch
import torch.nn as nn
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants/Configuration
RMQ_URL = os.getenv("RMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_NAME = os.getenv("RMQ_QUEUE", "ewc_retraining_jobs")
DEVICE = torch.device("cpu") # CPU bound to prevent GPU resource conflict

class FraudNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

def dummy_ewc_calculation_heavy(data):
    """
    Simulates a memory-intensive Fisher Information Matrix (FIM) and EWC calculation loop.
    Prone to triggering MemoryErrors if mock-loaded with too much memory.
    """
    logger.info("Starting memory-intensive EWC Retraining and Fisher Information Matrix (FIM) calculation...")

    # Simulate high memory load
    # If the job specifies an artificial failure or mock size
    mem_size = data.get("mock_memory_size_mb", 10)
    if mem_size > 4096: # If someone specifies more than 4GB mock size
        raise MemoryError("Simulated allocation failure: out of memory allocating EWC Fisher Matrix.")

    # Perform actual lightweight model operations
    model = FraudNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    # Just a basic dummy training step to simulate work
    x = torch.randn(100, 5).to(DEVICE)
    y = torch.randint(0, 2, (100, 1), dtype=torch.float32).to(DEVICE)

    model.train()
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    # Simulate some CPU delay
    time.sleep(2.0)
    logger.info("EWC / FIM Calculation completed successfully.")

def on_message_callback(ch, method, properties, body):
    """
    Callback function that safely executes EWC retraining inside try/except block.
    """
    logger.info(f"Received a training job. Delivery tag: {method.delivery_tag}")

    try:
        data = json.loads(body.decode())
    except Exception as e:
        logger.error(f"Failed to parse incoming message body: {e}")
        # Reject malformed message and do not requeue
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return

    try:
        # Wrap EWC calculation inside an explicit try/except block to manage MemoryError or other failures
        dummy_ewc_calculation_heavy(data)

        # Acknowledge success
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info(f"Successfully processed and acknowledged job {method.delivery_tag}")

    except MemoryError as me:
        logger.error(f"Memory allocation failed during EWC calculation: {me}. Gracefully rejecting message to let other pods retry.")
        # Gracefully reject with requeue=True so other pods/workers can pick it up
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        # Add sleep to cool down worker node before pulling again if it gets requeued to self
        time.sleep(5)

    except Exception as ex:
        logger.error(f"An unexpected exception occurred during EWC training: {ex}. Rejecting and requeuing.")
        # Gracefully reject with requeue=True so other pods can pick it up
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

def main():
    logger.info(f"Connecting to RabbitMQ at {RMQ_URL}...")

    connection = None
    channel = None
    for attempt in range(10):
        try:
            # Parse parameters from URL
            params = pika.URLParameters(RMQ_URL)
            params.heartbeat = 60
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            logger.info("Connected to RabbitMQ successfully.")
            break
        except Exception as e:
            logger.warning(f"RabbitMQ connection attempt {attempt+1} failed: {e}")
            time.sleep(3)

    if not channel:
        logger.critical("Could not connect to RabbitMQ. Exiting consumer.")
        return

    try:
        # Declare queue
        channel.queue_declare(queue=QUEUE_NAME, durable=True)

        # Implement strict Quality of Service (QoS)
        # prefetch_count=1 ensures worker only pulls one training job at a time
        channel.basic_qos(prefetch_count=1)

        # Set up consumer
        channel.basic_consume(
            queue=QUEUE_NAME,
            on_message_callback=on_message_callback,
            auto_ack=False # Explicit ack/nack
        )

        logger.info(f"EWC RabbitMQ Consumer started. Waiting for training jobs in '{QUEUE_NAME}' with basic_qos(prefetch_count=1)...")
        channel.start_consuming()

    except KeyboardInterrupt:
        logger.info("Stopping RabbitMQ consumer gracefully...")
        if channel:
            channel.stop_consuming()
        if connection:
            connection.close()
    except Exception as e:
        logger.critical(f"Consumer error occurred: {e}", exc_info=True)
        if connection and not connection.is_closed:
            connection.close()

if __name__ == "__main__":
    main()

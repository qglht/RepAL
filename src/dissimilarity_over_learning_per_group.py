import warnings
import os
import argparse
import torch
import pandas as pd
from dsa_analysis import load_config
from src.toolkit import dissimilarity_over_learning
from threading import Thread, Lock
from queue import Queue

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"


class DissimilarityWorker(Thread):
    def __init__(self, task_queue, lock, output_file):
        super().__init__()
        self.task_queue = task_queue
        self.lock = lock
        self.output_file = output_file

    def run(self):
        while True:
            params = self.task_queue.get()
            if params is None:
                self.task_queue.task_done()
                break
            # try:
            result = self.dissimilarity_task(params)
            # except RuntimeError as e:
            #     print(f"Runtime error: {e}")
            #     self.task_queue.task_done()
            #     continue

            self.task_queue.task_done()
            with self.lock:
                df = pd.DataFrame([result])
                df.to_csv(
                    self.output_file,
                    mode="a",
                    header=not os.path.exists(self.output_file),
                    index=False,
                )

    def dissimilarity_task(self, params):
        args, rnn_type, activation, hidden_size, lr, batch_size, device = params
        dissimilarities_model = dissimilarity_over_learning(
            args.group1,
            args.group2,
            rnn_type,
            activation,
            hidden_size,
            lr,
            batch_size,
            device,
        )
        dissimilarities = {
            "group1": args.group1,
            "group2": args.group2,
            "rnn_type": rnn_type,
            "activation": activation,
            "hidden_size": hidden_size,
            "lr": lr,
            "batch_size": batch_size,
            "cka": dissimilarities_model["cka"],
            "procrustes": dissimilarities_model["procrustes"],
            "dsa": dissimilarities_model["dsa"],
            "accuracy_1": dissimilarities_model["accuracy_1"],
            "accuracy_2": dissimilarities_model["accuracy_2"],
        }
        print(f"dissimilarities dsa : {dissimilarities_model['dsa']}")
        print(f"Len of dissimilarities : dsa {len(dissimilarities_model['dsa'])}")
        return dissimilarities


def dissimilarity(args: argparse.Namespace) -> None:
    config = load_config("config.yaml")
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    devices = [torch.device("cpu")]  # default value is now cpu
    if num_gpus > 0:
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    print(f"Number of GPUs available: {num_gpus}")

    # Create a lock for writing to the output file
    lock = Lock()

    # Define output file
    output_file = f"data/dissimilarities_over_learning/{args.group1}_{args.group2}.csv"

    # Create a worker thread for each GPU
    workers = []
    task_queues = []
    for i, device in enumerate(devices):
        task_queue = Queue()
        task_queues.append(task_queue)
        worker = DissimilarityWorker(task_queue, lock, output_file)
        worker.start()
        workers.append(worker)

    # Enqueue tasks in a round-robin fashion across the GPU-specific queues
    queue_index = 0
    for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
        for activation in config["rnn"]["parameters"]["activations"]:
            for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
                for lr in config["rnn"]["parameters"]["learning_rate"]:
                    for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                        tasks = (
                            args,
                            rnn_type,
                            activation,
                            hidden_size,
                            lr,
                            batch_size,
                            devices[queue_index],
                        )
                        if num_gpus > 0:
                            task_queues[queue_index].put(tasks)
                            queue_index = (queue_index + 1) % num_gpus
                        else:
                            task_queues[0].put(tasks)

    # Stop workers
    for task_queue in task_queues:
        task_queue.put(None)

    # Wait for all tasks to be processed
    for worker in workers:
        worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--group1",
        type=str,
        default="pretrain_frozen",
        help="group to compare 1",
    )
    parser.add_argument(
        "--group2",
        type=str,
        default="pretrain_unfrozen",
        help="group to compare 2",
    )
    args = parser.parse_args()

    dissimilarity(args)

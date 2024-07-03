import warnings
import os
import argparse
from matplotlib.pylab import f
import torch
import pandas as pd
from dsa_analysis import load_config
from src.toolkit import dissimilarity_within_learning
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
            print(f"Processing task with params: {params}")
            # try:
            result = self.dissimilarity_task(params)
            # except RuntimeError as e:
            #     print(f"Runtime error: {e}")
            #     self.task_queue.task_done()
            #     continue
            print(f"Task completed with result: {result}")
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

        # Reinitialize the model and modules to avoid conflicts
        dissimilarities_model = dissimilarity_within_learning(
            args.group,
            rnn_type,
            activation,
            hidden_size,
            lr,
            batch_size,
            device,
        )

        dissimilarities = {
            "group": args.group,
            "rnn_type": rnn_type,
            "activation": activation,
            "hidden_size": hidden_size,
            "lr": lr,
            "batch_size": batch_size,
            "cka": dissimilarities_model["cka"],
            "procrustes": dissimilarities_model["procrustes"],
            "dsa": dissimilarities_model["dsa"],
            "accuracy": dissimilarities_model["accuracy"],
        }
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
    output_file = f"data/dissimilarities_within_learning/{args.group}.csv"

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
    models_done = 0
    for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
        for activation in config["rnn"]["parameters"]["activations"]:
            for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
                for lr in config["rnn"]["parameters"]["learning_rate"]:
                    for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                        print(f"Index : {100*models_done/64}")
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
                        models_done += 1

    # Stop workers
    for task_queue in task_queues:
        task_queue.put(None)

    # Wait for all tasks to be processed
    for worker in workers:
        worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--group",
        type=str,
        default="pretrain_frozen",
        help="group to compare 1",
    )
    args = parser.parse_args()

    dissimilarity(args)


# import warnings
# import os
# import argparse
# import torch
# import pandas as pd
# from dsa_analysis import load_config
# from src.toolkit import dissimilarity_within_learning

# # Suppress specific Gym warnings
# warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
# warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# # Set environment variable to ignore Gym deprecation warnings
# os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"


# def dissimilarity_task(params):
#     args, rnn_type, activation, hidden_size, lr, batch_size, device = params
#     dissimilarities_model = dissimilarity_within_learning(
#         args.group,
#         rnn_type,
#         activation,
#         hidden_size,
#         lr,
#         batch_size,
#         device,
#     )
#     dissimilarities = {
#         "group": args.group,
#         "rnn_type": rnn_type,
#         "activation": activation,
#         "hidden_size": hidden_size,
#         "lr": lr,
#         "batch_size": batch_size,
#         "cka": dissimilarities_model["cka"],
#         "procrustes": dissimilarities_model["procrustes"],
#         "dsa": dissimilarities_model["dsa"],
#         "accuracy": dissimilarities_model["accuracy"],
#     }
#     return dissimilarities


# def dissimilarity(args: argparse.Namespace) -> None:
#     config = load_config("config.yaml")
#     num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
#     devices = (
#         [torch.device(f"cuda:{i}") for i in range(num_gpus)]
#         if num_gpus > 0
#         else [torch.device("cpu")]
#     )
#     print(f"Number of GPUs available: {num_gpus}")
#     device_index = 0
#     results = []

#     diss_index = 0
#     for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
#         for activation in config["rnn"]["parameters"]["activations"]:
#             for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
#                 for lr in config["rnn"]["parameters"]["learning_rate"]:
#                     for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
#                         print(f"Index : {100*diss_index/64}")
#                         results.append(
#                             dissimilarity_task(
#                                 (
#                                     args,
#                                     rnn_type,
#                                     activation,
#                                     hidden_size,
#                                     lr,
#                                     batch_size,
#                                     devices[device_index],
#                                 )
#                             )
#                         )

#     # Create DataFrame keeping lists intact
#     dissimilarities_df = pd.DataFrame(results)
#     dissimilarities_df.to_csv(
#         f"data/dissimilarities_within_learning/{args.group}.csv",
#         index=False,
#     )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train the model")
#     parser.add_argument(
#         "--group",
#         type=str,
#         default="pretrain_frozen",
#         help="group to compare 1",
#     )
#     args = parser.parse_args()

#     dissimilarity(args)

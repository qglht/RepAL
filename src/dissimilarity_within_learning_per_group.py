import warnings
import os
import argparse
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


def dissimilarity_task(params):
    args, rnn_type, activation, hidden_size, lr, batch_size, device = params
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


def worker(task_queue, lock, output_file):
    while True:
        params = task_queue.get()
        if params is None:
            break
        result = dissimilarity_task(params)
        task_queue.task_done()

        # Write results to CSV
        with lock:
            df = pd.DataFrame([result])
            df.to_csv(
                output_file,
                mode="a",
                header=not os.path.exists(output_file),
                index=False,
            )


def dissimilarity(args: argparse.Namespace) -> None:
    config = load_config("config.yaml")
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    devices = (
        [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        if num_gpus > 0
        else [torch.device("cpu")]
    )
    print(f"Number of GPUs available: {num_gpus}")

    # Create a queue for tasks
    task_queue = Queue()
    lock = Lock()

    # Define output file
    output_file = f"data/dissimilarities_within_learning/{args.group}.csv"

    # Start worker threads
    threads = []
    for _ in devices:
        t = Thread(target=worker, args=(task_queue, lock, output_file))
        t.start()
        threads.append(t)

    # Enqueue tasks in a round-robin fashion
    device_index = 0
    for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
        for activation in config["rnn"]["parameters"]["activations"]:
            for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
                for lr in config["rnn"]["parameters"]["learning_rate"]:
                    for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                        task_queue.put(
                            (
                                args,
                                rnn_type,
                                activation,
                                hidden_size,
                                lr,
                                batch_size,
                                devices[device_index],
                            )
                        )
                        device_index = (device_index + 1) % num_gpus

    # Stop workers
    for _ in devices:
        task_queue.put(None)

    # Wait for all tasks to be processed
    task_queue.join()

    # Wait for all threads to finish
    for t in threads:
        t.join()


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

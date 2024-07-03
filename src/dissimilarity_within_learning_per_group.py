import warnings
import os
import argparse
import torch
from dsa_analysis import load_config
from src.toolkit import dissimilarity_within_learning
import numpy as np
import multiprocessing

# Suppress specific Gym warnings
warnings.filterwarnings("ignore", message=".*Gym version v0.24.1.*")
warnings.filterwarnings("ignore", message=".*The `registry.all` method is deprecated.*")

# Set environment variable to ignore Gym deprecation warnings
os.environ["GYM_IGNORE_DEPRECATION_WARNINGS"] = "1"


def dissimilarity_task(params):
    group, rnn_type, activation, hidden_size, lr, batch_size, device = params

    # Reinitialize the model and modules to avoid conflicts
    dissimilarities_model = dissimilarity_within_learning(
        group,
        rnn_type,
        activation,
        hidden_size,
        lr,
        batch_size,
        device,
    )

    for measure in ["cka", "procrustes", "dsa", "accuracy"]:
        npz_filename = f"{group}/{measure}/{rnn_type}_{activation}_{hidden_size}_{lr}_{batch_size}.npz"  # Construct filename
        # create directory if it does not exist
        if not os.path.exists(
            f"data/dissimilarities_within_learning/{group}/{measure}"
        ):
            os.makedirs(
                f"data/dissimilarities_within_learning/{group}/{measure}",
                exist_ok=True,
            )
        npz_filename = os.path.join(
            "data/dissimilarities_within_learning", npz_filename
        )
        np.savez_compressed(npz_filename, dissimilarities_model[measure])
    return dissimilarities_model


def dissimilarity(args: argparse.Namespace) -> None:
    multiprocessing.set_start_method("spawn", force=True)
    config = load_config("config.yaml")

    # Create a list of all tasks to run
    tasks = []
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    devices = (
        [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        if num_gpus > 0
        else [torch.device("cpu")]
    )
    i = 0
    print(f"devices used : {devices}")
    print(f"number of devices : {num_gpus}")

    if not os.path.exists(f"data/dissimilarities_within_learning/{args.group}"):
        os.makedirs(
            f"data/dissimilarities_within_learning/{args.group}",
            exist_ok=True,
        )

    for rnn_type in config["rnn"]["parameters"]["rnn_type"]:
        for activation in config["rnn"]["parameters"]["activations"]:
            for hidden_size in config["rnn"]["parameters"]["n_rnn"]:
                for lr in config["rnn"]["parameters"]["learning_rate"]:
                    for batch_size in config["rnn"]["parameters"]["batch_size_train"]:
                        device = devices[
                            i % len(devices)
                        ]  # Cycle through available devices
                        tasks.append(
                            (
                                args.group,
                                rnn_type,
                                activation,
                                hidden_size,
                                lr,
                                batch_size,
                                device,
                            )
                        )
                        i += 1

    # Create a process for each task
    processes = [
        multiprocessing.Process(target=dissimilarity_task, args=task) for task in tasks
    ]

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()


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

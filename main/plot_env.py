import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import main
import numpy as np


def plot_env(env, n_trials):
    # Load data using a data loader function
    dataloader = main.get_dataloader(
        env,
        batch_size=n_trials,
        num_workers=0,
        shuffle=False,
        mode="test",
        train_split=0.8,
    )["test"]
    inputs, label, mask = next(iter(dataloader))

    # Determine the number of time steps and features
    num_time_steps = inputs.shape[1]
    num_features = inputs.shape[2]

    # Concatenate time steps for each input feature and output
    input_concat = inputs.reshape(
        -1, num_features
    )  # Shape: [number_of_time_steps * num_trials, num_features]
    output_concat = label.reshape(-1)  # Shape: [number_of_time_steps * num_trials]

    # Define feature names and colors
    feature_names = [
        "Fixation",
        "Stimulus Modality 1",
        "Stimulus Modality 2",
        "Rule Input 1",
        "Rule Input 2",
        "Rule Input 3",
        "Rule Input 4",
    ]
    colors = [
        "#FF8C00",  # Fixation - Dark Orange
        "#1E90FF",  # Stimulus Modality 1 - Dodger Blue
        "#FF4500",  # Stimulus Modality 2 - Orange Red
        "#32CD32",  # Rule Input 1 - Lime Green
        "#3CB371",  # Rule Input 2 - Medium Sea Green
        "#2E8B57",  # Rule Input 3 - Sea Green
        "#228B22",  # Rule Input 4 - Forest Green
    ]

    # Create a figure with subplots for Observations (7) and Response (1)
    fig, axs = plt.subplots(nrows=8, ncols=1, figsize=(5, 8), sharex=True)
    fig.suptitle(f"{env}: Observations", fontsize=8)

    # Plot Observations (7 subplots for each feature)
    time_steps_range = range(num_time_steps * n_trials)
    for i in range(num_features):
        if feature_names[i].startswith("Rule Input"):
            feature_data = input_concat[:, i]
            normalized_data = feature_data / feature_data.max()  # Normalize to [0, 1]
            axs[i].plot(time_steps_range, normalized_data, color=colors[i])
            axs[i].set_ylim(-0.25, 1.25)  # Set y-axis limit for Rule Inputs
        else:
            axs[i].plot(time_steps_range, input_concat[:, i], color=colors[i])

        axs[i].set_ylabel(feature_names[i], fontsize=6)

        # Add red separation lines between each trial
        for t in range(num_time_steps, input_concat.shape[0], num_time_steps):
            axs[i].axvline(x=t, color="red", linestyle="--", linewidth=0.8)

    # Plot Response
    ax_resp = axs[-1]  # Use the last axis for the response
    ax_resp.plot(time_steps_range, output_concat, color="#800080")  # Nice Purple
    ax_resp.set_title(f"{env}: Response", fontsize=8)
    ax_resp.set_xlabel("Time Steps (Concatenated)", fontsize=6)
    ax_resp.set_ylabel("Response Value", fontsize=6)

    # Add red separation lines between each trial
    for t in range(num_time_steps, output_concat.shape[0], num_time_steps):
        ax_resp.axvline(x=t, color="red", linestyle="--", linewidth=0.8)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Return the figure and axes
    return fig, axs

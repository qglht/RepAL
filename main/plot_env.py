import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import main
import numpy as np

# Define font sizes and styles for the plot
SIZE_DEFAULT = 18
SIZE_LARGE = 20
plt.rc("font", family="Arial")  # controls default font
plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

import matplotlib.pyplot as plt
import numpy as np

# Updated color mapping in a similar style as the second example
color_mapping = {
    "Fixation": "#ED6A5A",  # Bittersweet (reddish-orange)
    "Stimulus Modality 1": "#E5B25D",  # Hunyadi yellow (mustard-yellow)
    "Stimulus Modality 2": "#9BC1BC",  # Ash gray (muted teal)
    "Rule Input 1": "#8FBC8F",  # Payne's gray (blue-gray)
    "Rule Input 2": "#8FBC8F",  # Payne's gray (blue-gray)
    "Rule Input 3": "#8FBC8F",  # Palatinate (deep purple)
    "Rule Input 4": "#8FBC8F",  # Light purple
}


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

    # Define feature names
    feature_names = [
        "Fixation",
        "Stimulus Modality 1",
        "Stimulus Modality 2",
        "Rule Input 1",
        "Rule Input 2",
        "Rule Input 3",
        "Rule Input 4",
    ]

    # Create figure with subplots for Observations (7) and Response (1), shared x-axis
    fig, axs = plt.subplots(
        nrows=8,
        ncols=1,
        figsize=(4, 8),
        dpi=500,
        sharex=True,
        gridspec_kw={"hspace": 0.7},
    )
    env = env.replace("T", " ")

    time_steps_range = range(num_time_steps * n_trials)

    for i in range(num_features):
        feature_data = input_concat[:, i]
        normalized_data = (
            feature_data / feature_data.max()
            if feature_names[i].startswith("Rule Input")
            else feature_data
        )
        axs[i].plot(
            time_steps_range,
            normalized_data,
            color=color_mapping[feature_names[i]],
            lw=1.5,
        )
        axs[i].set_ylim(-0.25 if "Rule Input" in feature_names[i] else None, 1.25)
        axs[i].set_ylabel("", rotation=0, labelpad=0)  # Remove y-axis label

        # Plot a horizontal line at 0 for Rule Inputs if the value isn't 1
        if feature_names[i].startswith("Rule Input"):
            axs[i].axhline(y=0, color="gray", linestyle="--", linewidth=1)

        # Draw separation lines between trials
        for t in range(num_time_steps, len(time_steps_range), num_time_steps):
            axs[i].axvline(x=t, color="gray", linestyle="--", lw=0.8)


    # Plot Response
    ax_resp = axs[-1]
    ax_resp.plot(
        time_steps_range, output_concat, color="#ED6A5A", lw=1.5
    )  # Deep purple
    ax_resp.set_xlabel("Time (ms)")
    ax_resp.set_ylabel("", rotation=0, labelpad=0)  # Remove y-axis label

    # Draw separation lines between trials for Response
    for t in range(num_time_steps, len(time_steps_range), num_time_steps):
        ax_resp.axvline(x=t, color="gray", linestyle="--", lw=0.8)

    for ax in axs:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # Tight layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Return the figure and axes
    return fig, axs

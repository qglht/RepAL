import matplotlib.pyplot as plt
import numpy as np
from pyparsing import line

# Generate x values and Gaussian function
x = np.linspace(-5, 5, 500)
y = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

# Create the plot
plt.figure(figsize=(6, 4), facecolor="white", dpi=500)
plt.plot(x, y, color="black", linewidth=2)

# Remove the axes
plt.gca().set_axis_off()

# Adjust layout and show the plot
plt.subplots_adjust(top=1.02, bottom=-0.02, right=1.02, left=-0.02, hspace=0, wspace=0)
plt.margins(0.02, 0.02)  # Adjust the margins to give space for the curve
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig("gaussian_curve.png", bbox_inches="tight", pad_inches=0)
plt.show()

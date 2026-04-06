import matplotlib.pyplot as plt
import numpy as np

# Sample Data
x, y = np.random.rand(2, 30)
size = np.random.rand(30) * 1000  # Bubble size
colors = np.random.rand(30)       # Bubble color

plt.scatter(x, y, s=size, c=colors, alpha=0.5, cmap='viridis', edgecolors="white", linewidth=2)
plt.colorbar() # Add a legend for the colors
plt.show()

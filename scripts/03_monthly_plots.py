import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib for scientific publication (Q1 Journal standards)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Times New Roman',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.pad': 4,
    'ytick.major.pad': 4,
    'axes.grid': True,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.25,
    'legend.frameon': True,
    'legend.framealpha': 0.98,
    'legend.edgecolor': 'black',
})

df = pd.read_csv("results/statistics/monthly_climatology_future_2040_2069_ssp585.csv")

# Create figure with optimal dimensions for publication
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=100)

# Use high-quality color palette (Nature/Science style)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = colors[:len(df["zone"].unique())]

# Create smooth interpolation with cubic spline for better smoothness
from scipy.interpolate import CubicSpline

months_data = np.arange(1, 13)
x_smooth = np.linspace(1, 12, 300)

for i, zone in enumerate(df["zone"].unique()):
    z = df[(df["zone"] == zone)].sort_values('month')
    
    # Remove NaN values for interpolation
    valid_mask = z["rsds_mean"].notna()
    if valid_mask.sum() < 2:  # Skip if not enough valid points
        continue
    
    y_data = z[valid_mask]["rsds_mean"].values
    x_data = z[valid_mask]["month"].values
    
    # Use cubic spline for smoother curves
    cs = CubicSpline(x_data, y_data)
    y_smooth = cs(x_smooth)
    
    ax.plot(x_smooth, y_smooth, label=zone, color=colors[i], 
            linestyle='-', linewidth=1.2, zorder=2)
    # Add discrete markers at original data points
    ax.plot(z["month"], z["huss_mean"], marker=' ', markersize=3.5, 
            color=colors[i], linestyle='none', markerfacecolor='white',
            markeredgewidth=0.7, zorder=3)

# Add title
ax.set_title('Projected Monthly Mean Radiation by Region', fontsize=13, fontweight='normal', pad=15)

# Enhance axis labels and formatting
ax.set_xlabel("Month of the Year", fontweight='normal', fontsize=11)
ax.set_ylabel("Radiation (W/m2)", fontweight='normal', fontsize=11)
ax.set_xlim(0.5, 12.5)
ax.set_ylim(bottom=ax.get_ylim()[0])

# Set month labels on x-axis
months = np.arange(1, 13)
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(months)
ax.set_xticklabels(month_labels)

# Improve legend with right-side positioning
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fancybox=False, 
         edgecolor='black', framealpha=1.0, ncol=1, 
         fontsize=10, title_fontsize=10)

# Enhance grid for better readability
ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Keep all spines visible for publication style
ax.spines['top'].set_linewidth(0.8)
ax.spines['right'].set_linewidth(0.8)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Add tight layout and save with high quality (publication-ready)
plt.tight_layout()
plt.savefig("results/figures/monthly_rad_ssp585.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()
plt.close()
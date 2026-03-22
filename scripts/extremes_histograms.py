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

# Load thermal extremes data
df_hist = pd.read_csv("results/statistics/thermal_extremes_historical_1990_2019.csv")
df_fut = pd.read_csv("results/statistics/thermal_extremes_future_2040_2069_ssp585.csv")

# Get columns to plot (exclude 'zone' column)
columns_to_plot = [col for col in df_hist.columns if col != 'zone']

# Filter out zones with all zero values (empty data)
df_hist_filtered = df_hist[~((df_hist[columns_to_plot] == 0).all(axis=1))].reset_index(drop=True)
df_fut_filtered = df_fut[~((df_fut[columns_to_plot] == 0).all(axis=1))].reset_index(drop=True)

# Define colors for historical and future
color_hist = '#1f77b4'  # Blue
color_fut = '#d62728'   # Red

x = np.arange(len(df_hist_filtered))
width = 0.35

# Create separate figure for each column
for col in columns_to_plot:
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=100)
    
    # Create grouped bar chart
    bars1 = ax.bar(x - width/2, df_hist_filtered[col], width, label='Historical (1990-2019)',
                   color=color_hist, alpha=0.85, edgecolor='black', linewidth=0.7)
    bars2 = ax.bar(x + width/2, df_fut_filtered[col], width, label='Future (2040-2069 SSP5-8.5)',
                   color=color_fut, alpha=0.85, edgecolor='black', linewidth=0.7)
    
    # Set labels and title
    ax.set_xlabel('Zone', fontsize=11, fontweight='normal')
    ax.set_ylabel(f'{col.replace("_", " ").title()}', fontsize=11, fontweight='normal')
    ax.set_title(f'{col.replace("_", " ").title()}: Historical vs Future', 
                 fontsize=13, fontweight='normal', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df_hist_filtered['zone'], rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper left', frameon=True, fancybox=False, 
             edgecolor='black', framealpha=1.0, fontsize=10)
    
    # Enhance grid and spines
    ax.grid(True, which='major', linestyle='-', linewidth=0.4, alpha=0.3, zorder=0, axis='y')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure with descriptive name
    filename = f"results/figures/thermal_extremes_{col}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✅ Saved: {filename}")
    
    plt.close()

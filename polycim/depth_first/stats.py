import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_scatter(root_dir):
    # Get the list of CSV files
    csv_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.csv')])
    num_files = len(csv_files)

    # Calculate the number of rows needed for 4 columns
    ncols = 4
    nrows = (num_files + ncols - 1) // ncols  # Ceiling division

    # Create a figure for subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4))
    axes = axes.flatten() if num_files > 1 else [axes]  # Ensure axes is always a list

    # Iterate over all CSV files
    for i, filename in enumerate(csv_files):
        file_path = os.path.join(root_dir, filename)
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Plot scatter plot
        ax = axes[i]
        ax.scatter(df['count_time'], df['exe_time'], color='black', s=10)

        # Highlight the point with the minimum count_time
        min_exe_time_idx = df['exe_time'].idxmin()
        ax.scatter(df.loc[min_exe_time_idx, 'count_time'], df.loc[min_exe_time_idx, 'exe_time'], color='red', s=10)

        # Set the title of the subplot
        ax.set_title(filename)
        ax.set_xlabel('count_val_time')
        ax.set_ylabel('count_val_result')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()
    plt.savefig('stats.png')

# Example usage
plot_csv_scatter('/home/wangyiou/Desktop/pim_compiler/playground/.save/2025-01-05_23-56-50')

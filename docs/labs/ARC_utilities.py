from glob import glob
import json
from os import path
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from numpy import random
import math
import matplotlib.gridspec as gridspec


def plot_task(task):   
    # Separate groups (train first, then test)
    train_samples = task['train']
    test_samples = task['test']
    groups = [('Train', train_samples), ('Test', test_samples)]

    n_pair_cols = 3
    gap_ratio = 0.2  # relative width for gap columns
    
    # Build the width_ratios: for each pair, two columns (ratio 1 each), plus a gap column (if not the last pair)
    width_ratios = []
    for i in range(n_pair_cols):
        width_ratios.extend([1, 1])
        if i < n_pair_cols - 1:
            width_ratios.append(gap_ratio)
    total_cols = len(width_ratios)
    
    # Compute total number of rows needed (one row per sample pair)
    group_rows = [math.ceil(len(samples) / n_pair_cols) for (_, samples) in groups]
    total_rows = sum(group_rows)
    
    # Define a custom colormap with 10 colors (values 0 through 9)
    cmap = colors.ListedColormap(['black', 'blue', 'green', 'red', 'purple',
                                  'orange', 'cyan', 'magenta', 'yellow', 'brown'])
    norm = colors.BoundaryNorm(np.arange(-0.5, 10.5, 1), ncolors=10)
    
    # Create the figure using GridSpec
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(total_rows, total_cols, wspace=0.3, hspace=0.3, width_ratios=width_ratios)
    
    current_row = 0
    for group_label, samples in groups:
        num_rows = math.ceil(len(samples) / n_pair_cols)
        for r in range(num_rows):
            for pair_idx in range(n_pair_cols):
                sample_index = r * n_pair_cols + pair_idx
                # Calculate the starting column for this pair (each pair occupies 2 columns plus a gap column except after the last)
                col_base = pair_idx * 3  # input: col_base, output: col_base+1; col_base+2 is the gap column.
                if sample_index < len(samples):
                    sample = samples[sample_index]
                    ax_in = fig.add_subplot(gs[current_row, col_base])
                    ax_out = fig.add_subplot(gs[current_row, col_base + 1])
                    
                    ax_in.imshow(np.array(sample['input']), cmap=cmap, norm=norm)
                    ax_out.imshow(np.array(sample['output']), cmap=cmap, norm=norm)
                    
                    # Remove ticks for a clean look.
                    ax_in.set_xticks([]); ax_in.set_yticks([])
                    ax_out.set_xticks([]); ax_out.set_yticks([])
                    
                    # For the first pair of the first row in each group, set the group label as the y-axis label.
                    if r == 0 and pair_idx == 0:
                        ax_in.set_ylabel(group_label, fontsize=12, fontweight='bold')
                else:
                    # If no sample exists for this slot, create dummy axes and hide them.
                    ax_in = fig.add_subplot(gs[current_row, col_base])
                    ax_out = fig.add_subplot(gs[current_row, col_base + 1])
                    ax_in.axis('off')
                    ax_out.axis('off')
            current_row += 1
    
    # plt.tight_layout()
    plt.show()

def random_translate(grid, max_shift=4):
    """
    Shift the grid by a random amount in both dimensions,
    up to ±max_shift cells.
    """
    shift_rows = random.randint(-max_shift, max_shift)
    shift_cols = random.randint(-max_shift, max_shift)
    shifted = np.roll(grid, shift_rows, axis=0)
    shifted = np.roll(shifted, shift_cols, axis=1)
    return shifted
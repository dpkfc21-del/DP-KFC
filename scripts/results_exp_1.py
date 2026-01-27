import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Helper Function to Process Each File
# -----------------------------------------------------------------------------
def process_dataset(csv_path, dataset_name, model_name):
    # Check if file exists before trying to read
    if not os.path.exists(csv_path):
        print(f"Warning: File not found: {csv_path}")
        return pd.DataFrame()

    # Load Data directly from file path
    df = pd.read_csv(csv_path)
    
    # Normalize column names (Acc vs Accuracy)
    df.columns = [c.strip() for c in df.columns]
    if 'Acc' in df.columns:
        df.rename(columns={'Acc': 'Accuracy'}, inplace=True)
        
    # Standardize Method Names to match your requested columns
    method_map = {
        'Standard DP-SGD': 'DP-SGD',
        'KFAC (Noise)': 'Noise DP-KFAC',
        'KFAC (Public)': 'Public DP-KFAC',
        'Plain SGD': 'Base SGD'
    }
    df['Method'] = df['Method'].replace(method_map)
    
    # 1. Extract Base SGD (Baseline)
    # It usually has N/A epsilon, so we take the mean across all seeds
    base_df = df[df['Method'] == 'Base SGD']
    if not base_df.empty:
        base_mean = base_df['Accuracy'].mean() * 100
        base_std = base_df['Accuracy'].std() * 100
        base_str = f"{base_mean:.2f} $\\pm$ {base_std:.2f}"
    else:
        base_str = "-"

    # 2. Process DP Methods (Filter out Base SGD)
    dp_df = df[df['Method'] != 'Base SGD'].copy()
    
    # Ensure Epsilon is float for sorting
    dp_df['Epsilon'] = dp_df['Epsilon'].astype(float)
    
    # Aggregate: Group by Method and Epsilon -> Get Mean and Std
    agg = dp_df.groupby(['Method', 'Epsilon'])['Accuracy'].agg(['mean', 'std']).reset_index()
    
    # Format the accuracy string
    agg['Score'] = agg.apply(lambda x: f"{x['mean']*100:.2f} $\\pm$ {x['std']*100:.2f}", axis=1)
    
    # Pivot: Make Methods the columns, Epsilon the index
    pivot_df = agg.pivot(index='Epsilon', columns='Method', values='Score')
    
    # 3. Add Context Columns
    pivot_df['Base SGD'] = base_str # Constant for all epsilons
    pivot_df['Dataset'] = dataset_name
    pivot_df['Model'] = model_name
    
    # Reorder columns
    desired_order = ['Dataset', 'Model', 'Base SGD', 'DP-SGD', 'Noise DP-KFAC', 'Public DP-KFAC']
    # Filter for columns that actually exist (in case one method is missing)
    final_cols = [c for c in desired_order if c in pivot_df.columns or c in ['Dataset', 'Model', 'Base SGD']]
    
    return pivot_df.reset_index()[['Dataset', 'Model', 'Epsilon'] + final_cols[2:]]


def convert_to_numeric(df):
    """Convert accuracy strings to numeric values for plotting"""
    numeric_df = df.copy()
    for col in ['Base SGD', 'DP-SGD', 'Noise DP-KFAC', 'Public DP-KFAC']:
        if col in numeric_df.columns:
            # Check if column contains strings (not already numeric)
            if numeric_df[col].dtype == 'object':
                # Extract mean and std values from "mean ± std" format
                mean_values = numeric_df[col].str.extract(r'([\d.]+)').astype(float)
                # Extract std from LaTeX format: "$\\pm$ 0.58"
                std_values = numeric_df[col].str.extract(r'\$\\pm\$\s*([\d.]+)').astype(float)

                numeric_df[col] = mean_values
                numeric_df[col + '_std'] = std_values
            else:
                # Already numeric, set std to 0
                numeric_df[col + '_std'] = 0
    return numeric_df


def generate_plots(df):
    """Generate subplots with each dataset separated, comparing different methods (wide format)"""

    numeric_df = convert_to_numeric(df)
    datasets = numeric_df['Dataset'].unique()
    n_datasets = len(datasets)

    # Create subplots - one for each dataset (more compressed)
    fig, axes = plt.subplots(1, n_datasets, figsize=(3.5*n_datasets, 2.8))
    if n_datasets == 1:
        axes = [axes]

    fig.suptitle('DP Methods Comparison by Dataset', fontsize=10, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        subset = numeric_df[numeric_df['Dataset'] == dataset]

        # Plot each method as a line with confidence intervals
        methods = ['DP-SGD', 'Noise DP-KFAC', 'Public DP-KFAC']
        for j, method in enumerate(methods):
            if method in subset.columns:
                # Plot the line
                ax.plot(subset['Epsilon'], subset[method],
                       marker=markers[j], markersize=3, linewidth=1.2,
                       color=colors[j], label=method, alpha=0.8)

                # Add confidence interval bands
                if method + '_std' in subset.columns:
                    std_values = subset[method + '_std']
                    ax.fill_between(subset['Epsilon'],
                                   subset[method] - std_values,
                                   subset[method] + std_values,
                                   color=colors[j], alpha=0.25, linewidth=0)
                    # Add error bars for better visibility
                    ax.errorbar(subset['Epsilon'], subset[method],
                              yerr=std_values, fmt='none',
                              color=colors[j], alpha=0.4, capsize=2)

        # Add baseline as horizontal line with confidence interval
        if 'Base SGD' in subset.columns:
            baseline = subset['Base SGD'].iloc[0]
            ax.axhline(y=baseline, color='gray', linestyle='--',
                      linewidth=0.8, alpha=0.7, label='Base SGD')

            # Add baseline confidence interval if available
            if 'Base SGD_std' in subset.columns:
                baseline_std = subset['Base SGD_std'].iloc[0]
                ax.axhspan(baseline - baseline_std, baseline + baseline_std,
                          color='gray', alpha=0.15)

        ax.set_xlabel('ε', fontsize=8)
        ax.set_ylabel('Accuracy (%)', fontsize=8)
        ax.set_title(f'{dataset}', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2)

        # Add legend only to first subplot to save space
        if i == 0:
            ax.legend(fontsize=6, frameon=False, loc='best')

        # Set y-axis limits to reduce whitespace (including confidence intervals)
        all_values = []
        for method in methods:
            if method in subset.columns:
                all_values.extend(subset[method].dropna().tolist())
                if method + '_std' in subset.columns:
                    all_values.extend((subset[method] - subset[method + '_std']).dropna().tolist())
                    all_values.extend((subset[method] + subset[method + '_std']).dropna().tolist())
        if 'Base SGD' in subset.columns:
            all_values.append(subset['Base SGD'].iloc[0])
            if 'Base SGD_std' in subset.columns:
                baseline_std = subset['Base SGD_std'].iloc[0]
                all_values.append(subset['Base SGD'].iloc[0] - baseline_std)
                all_values.append(subset['Base SGD'].iloc[0] + baseline_std)

        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            ax.set_ylim(bottom=max(0, y_min - 0.05*y_range), top=y_max + 0.05*y_range)

    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(top=0.85)

    # Save the figure
    output_path = "/home/mmolinav/Projects/dynamicdp/dp_methods_by_dataset_wide.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nWide plot saved to: {output_path}")

    plt.close()


def generate_compact_plot(df):
    """
    Generate a compact single-column plot for two-column papers.
    Stacks datasets vertically to fit within ~3.5 inch column width.
    """
    numeric_df = convert_to_numeric(df)
    datasets = numeric_df['Dataset'].unique()
    n_datasets = len(datasets)

    # Single column width is typically 3.3-3.5 inches for two-column papers
    # Stack vertically with minimal height per subplot
    fig, axes = plt.subplots(n_datasets, 1, figsize=(3.3, 1.4 * n_datasets), sharex=False)
    if n_datasets == 1:
        axes = [axes]

    colors = {'DP-SGD': '#1f77b4', 'Noise DP-KFAC': '#ff7f0e', 'Public DP-KFAC': '#2ca02c'}
    markers = {'DP-SGD': 'o', 'Noise DP-KFAC': 's', 'Public DP-KFAC': '^'}
    # Shorter labels for legend
    short_labels = {'DP-SGD': 'DP-SGD', 'Noise DP-KFAC': 'Noise', 'Public DP-KFAC': 'Public'}

    methods = ['DP-SGD', 'Noise DP-KFAC', 'Public DP-KFAC']

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        subset = numeric_df[numeric_df['Dataset'] == dataset]

        for method in methods:
            if method in subset.columns:
                ax.plot(subset['Epsilon'], subset[method],
                       marker=markers[method], markersize=2.5, linewidth=1,
                       color=colors[method], label=short_labels[method], alpha=0.85)

                if method + '_std' in subset.columns:
                    std_values = subset[method + '_std']
                    ax.fill_between(subset['Epsilon'],
                                   subset[method] - std_values,
                                   subset[method] + std_values,
                                   color=colors[method], alpha=0.2, linewidth=0)

        # Baseline
        if 'Base SGD' in subset.columns:
            baseline = subset['Base SGD'].iloc[0]
            ax.axhline(y=baseline, color='gray', linestyle='--',
                      linewidth=0.7, alpha=0.6, label='Base')
            if 'Base SGD_std' in subset.columns:
                baseline_std = subset['Base SGD_std'].iloc[0]
                ax.axhspan(baseline - baseline_std, baseline + baseline_std,
                          color='gray', alpha=0.1)

        # Compact labels
        ax.set_ylabel('Acc (%)', fontsize=6)
        ax.set_title(f'{dataset}', fontsize=7, fontweight='bold', pad=2)
        ax.tick_params(axis='both', labelsize=5)
        ax.grid(True, alpha=0.15, linewidth=0.5)

        # Tight y-limits
        all_values = []
        for method in methods:
            if method in subset.columns:
                all_values.extend(subset[method].dropna().tolist())
                if method + '_std' in subset.columns:
                    all_values.extend((subset[method] - subset[method + '_std']).dropna().tolist())
                    all_values.extend((subset[method] + subset[method + '_std']).dropna().tolist())
        if 'Base SGD' in subset.columns:
            all_values.append(subset['Base SGD'].iloc[0])

        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            ax.set_ylim(bottom=max(0, y_min - 0.03*y_range), top=y_max + 0.03*y_range)

    # Only add x-label to bottom subplot
    axes[-1].set_xlabel('Privacy Budget (ε)', fontsize=6)

    # Single legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(pad=0.3, h_pad=0.4)
    plt.subplots_adjust(bottom=0.12)

    output_path = "/home/mmolinav/Projects/dynamicdp/dp_methods_compact.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Compact plot saved to: {output_path}")

    plt.close()

# -----------------------------------------------------------------------------
# 2. Configuration: File Paths
# -----------------------------------------------------------------------------

# Your specific file paths
csv1_path = "/home/mmolinav/Projects/dynamicdp/results_stackoverflow/stackoverflow_dp_kfac_results.csv"
csv2_path = "/home/mmolinav/Projects/dynamicdp/mnist_experiment_results.csv"
csv3_path = "/home/mmolinav/Projects/dynamicdp/cross_vit_cifar100_results.csv"

files_config = [
    # (File Path,          Dataset Name,   Model Name)
    (csv1_path,           "StackOverflow", "Roberta"), 
    (csv2_path,           "MNIST",         "CNN"),
    (csv3_path,           "CIFAR-100",    "CrossViT"),
    # Add more tuples here if needed
]

# -----------------------------------------------------------------------------
# 3. Generate and Merge Tables
# -----------------------------------------------------------------------------
all_results = []
for f_path, d_name, m_name in files_config:
    processed_df = process_dataset(f_path, d_name, m_name)
    if not processed_df.empty:
        all_results.append(processed_df)

if not all_results:
    print("No data loaded. Check your file paths.")
else:
    final_df = pd.concat(all_results, ignore_index=True)

    # -----------------------------------------------------------------------------
    # 4. Convert to LaTeX
    # -----------------------------------------------------------------------------
    latex_code = final_df.to_latex(
        index=False,
        escape=False, # Allows the \pm to render correctly
        column_format="llc|c|ccc", # Adjust vertical lines here
        header=["Dataset", "Model", r"$\epsilon$", "Base SGD", "DP-SGD", "Noise DP-KFAC", "Public DP-KFAC"],
        float_format="%.2f"
    )

    # Post-processing for cleaner Multirow (Generic version)
    lines = latex_code.splitlines()
    formatted_lines = []
    last_dataset = ""
    last_model = ""

    for line in lines:
        # Check if line looks like a data row (contains &) but not the header (contains epsilon)
        if "&" in line and "epsilon" not in line and "Dataset" not in line: 
            parts = line.split('&')
            current_dataset = parts[0].strip()
            current_model = parts[1].strip()
            
            # If dataset name matches previous row, blank it out
            if current_dataset == last_dataset:
                parts[0] = "" 
            else:
                last_dataset = current_dataset
                
            # If model name matches previous row (AND dataset matches), blank it out
            if current_model == last_model and parts[0] == "": 
                parts[1] = ""
            else:
                last_model = current_model
                
            formatted_lines.append(" & ".join(parts))
        else:
            formatted_lines.append(line)

    final_latex = "\n".join(formatted_lines)

    print("-" * 30)
    print("GENERATED LATEX CODE")
    print("-" * 30)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Comparison of Accuracy across Datasets and Privacy Budgets}")
    print(final_latex)
    print("\\end{table}")
    
    # -----------------------------------------------------------------------------
    # 5. Generate Plots
    # -----------------------------------------------------------------------------
    # Wide format plot (original)
    generate_plots(final_df)

    # Compact single-column plot for two-column papers
    generate_compact_plot(final_df)
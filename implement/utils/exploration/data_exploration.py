import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ks_2samp
from implement.utils.helper import INTERSECTING_FEATURES

# if vertical_speed_variance is not in INTERSECTING_FEATURES, add it for better analysis
if 'vertical_speed_variance' not in INTERSECTING_FEATURES:
    INTERSECTING_FEATURES.append('vertical_speed_variance')

def load_csv_files(paths_dict):
    """
    Loads multiple CSV or NPY files into a dictionary of pandas DataFrames.
    
    Args:
        paths_dict (dict): Dictionary mapping label/name to CSV or NPY file path.
        
    Returns:
        dict: Dictionary mapping label/name to DataFrame.
    """
    dfs = {}
    for name, path in paths_dict.items():
        p = Path(path)
        if p.exists():
            print(f"Loading {name} dataset from {p}...")
            if p.suffix == '.npy':
                arr = np.load(p)
                if arr.ndim == 3:
                    # Shape: (num_samples, seq_len, num_features)
                    num_features = arr.shape[2]
                    arr_2d = arr.reshape(-1, num_features)
                    print(f"  Loaded 3D numpy array of shape {arr.shape}, reshaped to 2D {arr_2d.shape}")
                elif arr.ndim == 2:
                    # Shape: (num_samples, num_features)
                    num_features = arr.shape[1]
                    arr_2d = arr
                    print(f"  Loaded 2D numpy array of shape {arr.shape}")
                else:
                    print(f"[Error] Unsupported numpy array dimension {arr.ndim} for {name}")
                    continue
                
                # Check if feature count matches intersecting features
                if num_features == len(INTERSECTING_FEATURES):
                    cols = INTERSECTING_FEATURES
                else:
                    cols = [f"feat_{i}" for i in range(num_features)]
                dfs[name] = pd.DataFrame(arr_2d, columns=cols)
            else:
                dfs[name] = pd.read_csv(p)
        else:
            print(f"[Warning] Path for {name} does not exist: {p}")
    return dfs

def compute_summary_statistics(dfs_dict, features):
    """
    Computes summary statistics (mean, std, min, max, median) for specified features across datasets.
    """
    summary_data = []
    for dataset_name, df in dfs_dict.items():
        # Clean dataframe features
        df_clean = df[features].replace([np.inf, -np.inf], np.nan).dropna()
        for feat in features:
            if feat in df_clean.columns:
                values = df_clean[feat].values
                summary_data.append({
                    'Dataset': dataset_name,
                    'Feature': feat,
                    'Mean': np.mean(values),
                    'StdDev': np.std(values),
                    'Min': np.min(values),
                    'Median': np.median(values),
                    'Max': np.max(values)
                })
    return pd.DataFrame(summary_data)

def plot_feature_distributions_grid(dfs_dict, features, output_dir):
    """
    Generates a single grid figure of overlay distribution (density) plots for all features
    and saves it in lossless SVG and high-res PNG formats.
    
    Applies log1p scale to sus features to handle extreme ranges / outliers.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    
    sus_features = {
        'vertical_speed_variance', 'prediction_error', 'heading_speed_consistency',
        'motion_smoothness', 'path_curvature', 'ground_speed'
    }
    
    for idx, feat in enumerate(features):
        ax = axes[idx]
        has_data = False
        is_sus = feat in sus_features
        
        for name, df in dfs_dict.items():
            if feat in df.columns:
                clean_values = df[feat].replace([np.inf, -np.inf], np.nan).dropna().values
                if len(clean_values) > 0:
                    if is_sus:
                        transformed_values = np.sign(clean_values) * np.log1p(np.abs(clean_values))
                        sns.kdeplot(transformed_values, fill=True, label=name, alpha=0.5, linewidth=2, ax=ax)
                    else:
                        sns.kdeplot(clean_values, fill=True, label=name, alpha=0.5, linewidth=2, ax=ax)
                    has_data = True
                    
        if has_data:
            title_suffix = " (log1p)" if is_sus else ""
            ax.set_title(f"{feat}{title_suffix}", fontsize=12, pad=10)
            ax.set_xlabel("Value (log1p)" if is_sus else "Value", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.legend(fontsize=8)
        else:
            ax.axis('off')
            
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
        
    plt.suptitle("Comparative Feature Distributions (Log-scale applied to sus features)", fontsize=16, y=0.99)
    plt.tight_layout()
    
    svg_path = output_dir / "distributions_grid.svg"
    png_path = output_dir / "distributions_grid.png"
    plt.savefig(svg_path, format='svg')
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✓ Saved distributions grid to:")
    print(f"  - Vector (SVG): {svg_path}")
    print(f"  - High-res (PNG): {png_path}")

def plot_feature_boxplots_grid(dfs_dict, features, output_dir):
    """
    Generates a single grid figure of boxplots for all features
    and saves it in lossless SVG and high-res PNG formats.
    
    Applies log1p scale to sus features to handle extreme ranges / outliers.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    combined_list = []
    for name, df in dfs_dict.items():
        available_feats = [f for f in features if f in df.columns]
        if available_feats:
            df_sub = df[available_feats].copy()
            df_sub = df_sub.replace([np.inf, -np.inf], np.nan).dropna()
            df_sub['Dataset'] = name
            combined_list.append(df_sub)
            
    if not combined_list:
        print("[Warning] No overlapping features found to plot boxplots.")
        return
        
    combined_df = pd.concat(combined_list, ignore_index=True)
    
    sus_features = {
        'vertical_speed_variance', 'prediction_error', 'heading_speed_consistency',
        'motion_smoothness', 'path_curvature', 'ground_speed'
    }
    
    plot_df = combined_df.copy()
    for feat in features:
        if feat in sus_features and feat in plot_df.columns:
            plot_df[feat] = np.sign(plot_df[feat]) * np.log1p(np.abs(plot_df[feat]))
            
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, feat in enumerate(features):
        ax = axes[idx]
        if feat in plot_df.columns:
            sns.boxplot(data=plot_df, x='Dataset', y=feat, hue='Dataset', legend=False, ax=ax)
            is_sus = feat in sus_features
            title_suffix = " (log1p)" if is_sus else ""
            ax.set_title(f"{feat}{title_suffix}", fontsize=12, pad=10)
            ax.set_ylabel("Value (log1p)" if is_sus else "Value", fontsize=10)
            ax.set_xlabel("Dataset", fontsize=10)
        else:
            ax.axis('off')
            
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
        
    plt.suptitle("Comparative Feature Boxplots (Log-scale applied to sus features)", fontsize=16, y=0.99)
    plt.tight_layout()
    
    svg_path = output_dir / "boxplots_grid.svg"
    png_path = output_dir / "boxplots_grid.png"
    plt.savefig(svg_path, format='svg')
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✓ Saved boxplots grid to:")
    print(f"  - Vector (SVG): {svg_path}")
    print(f"  - High-res (PNG): {png_path}")

def plot_correlation_heatmaps_grid(dfs_dict, features, output_dir):
    """
    Generates a single figure showing correlation heatmaps for all datasets side-by-side
    and saves it in lossless SVG and high-res PNG formats.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_datasets = len(dfs_dict)
    if n_datasets == 0:
        return
        
    fig, axes = plt.subplots(1, n_datasets, figsize=(10 * n_datasets, 9))
    if n_datasets == 1:
        axes = [axes]
        
    for idx, (name, df) in enumerate(dfs_dict.items()):
        ax = axes[idx]
        df_clean = df[features].replace([np.inf, -np.inf], np.nan).dropna()
        if df_clean.empty:
            ax.axis('off')
            continue
            
        corr = df_clean.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True, fmt=".2f", square=True, 
                    linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title(f"Correlation Matrix: {name}", fontsize=14, pad=15)
        
    plt.suptitle("Dataset Correlation Heatmaps", fontsize=18, y=0.98)
    plt.tight_layout()
    
    svg_path = output_dir / "correlations_grid.svg"
    png_path = output_dir / "correlations_grid.png"
    plt.savefig(svg_path, format='svg')
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✓ Saved correlation heatmaps grid to:")
    print(f"  - Vector (SVG): {svg_path}")
    print(f"  - High-res (PNG): {png_path}")

# Retain old signature placeholders for backward compatibility
def plot_feature_distributions(dfs_dict, features, output_dir):
    plot_feature_distributions_grid(dfs_dict, features, output_dir)

def plot_feature_boxplots(dfs_dict, features, output_dir):
    plot_feature_boxplots_grid(dfs_dict, features, output_dir)

def plot_correlation_heatmap(df, features, title, output_path):
    pass

def compute_separability_metrics(dfs_dict, features):
    """
    Computes Kolmogorov-Smirnov (KS) test statistics to quantify the separability 
    of each feature between datasets.
    
    Returns a DataFrame with KS statistics and p-values for all dataset pairs.
    """
    keys = list(dfs_dict.keys())
    if len(keys) < 2:
        return pd.DataFrame()
        
    results = []
    
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            name1, name2 = keys[i], keys[j]
            df1, df2 = dfs_dict[name1], dfs_dict[name2]
            
            for feat in features:
                if feat in df1.columns and feat in df2.columns:
                    val1 = df1[feat].replace([np.inf, -np.inf], np.nan).dropna().values
                    val2 = df2[feat].replace([np.inf, -np.inf], np.nan).dropna().values
                    
                    if len(val1) > 0 and len(val2) > 0:
                        ks_stat, p_val = ks_2samp(val1, val2)
                        mean_diff = np.abs(np.mean(val1) - np.mean(val2))
                        results.append({
                            'Pair': f"{name1} vs {name2}",
                            'Feature': feat,
                            'KS Statistic': ks_stat,
                            'p-value': p_val,
                            'Absolute Mean Difference': mean_diff
                        })
                        
    return pd.DataFrame(results)

def run_exploratory_analysis(dfs_dict, features, output_dir):
    """
    Runs full exploratory data analysis on the provided datasets.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*50)
    print("=== STARTING EXPLORATORY DATA ANALYSIS ===")
    print(f"Datasets under comparison: {list(dfs_dict.keys())}")
    print(f"Number of features analyzed: {len(features)}")
    print("="*50)
    
    # 1. Summary Statistics
    stats_df = compute_summary_statistics(dfs_dict, features)
    stats_path = output_dir / 'summary_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Saved summary statistics to: {stats_path}")
    
    # 2. Plot Grids (Distributions and Boxplots)
    print("\nGenerating feature distribution overlay grid (SVG/PNG)...")
    plot_feature_distributions_grid(dfs_dict, features, plots_dir)
    
    print("\nGenerating feature boxplot grid (SVG/PNG)...")
    plot_feature_boxplots_grid(dfs_dict, features, plots_dir)
    
    # 3. Correlation heatmap grid
    print("\nGenerating correlation heatmaps grid (SVG/PNG)...")
    plot_correlation_heatmaps_grid(dfs_dict, features, plots_dir)
        
    # 4. Compute Statistical Separability (KS Test)
    separability_df = compute_separability_metrics(dfs_dict, features)
    if not separability_df.empty:
        sep_path = output_dir / 'feature_separability.csv'
        separability_df.to_csv(sep_path, index=False)
        print(f"\n✓ Saved statistical feature separability matrix to: {sep_path}")
        
        print("\nTop 5 Most Separable Features (Kolmogorov-Smirnov Test):")
        top_sep = separability_df.sort_values(by='KS Statistic', ascending=False).head(5)
        for _, row in top_sep.iterrows():
            print(f"  - [{row['Pair']}] Feature '{row['Feature']}' | KS Stat: {row['KS Statistic']:.4f} (p-value: {row['p-value']:.2e})")
            
    print("\n" + "="*50)
    print("=== EXPLORATORY DATA ANALYSIS COMPLETE ===")
    print("="*50 + "\n")
    
    return stats_df, separability_df

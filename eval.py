import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error # r2_score was not used
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import traceback

from matplotlib.backends.backend_pdf import PdfPages
import textwrap
import math

# --- Import necessary functions from HelperFuncs ---
try:
    from HelperFuncs import load_file
except ImportError:
    print("Error: Could not import from HelperFuncs. Ensure HelperFuncs.py is accessible.")
    def load_file(filepath, **kwargs): raise NotImplementedError("HelperFuncs not found")

# --- Configuration ---
METHOD_BEING_EVALUATED = 'loso_global_residual'
FINAL_PREDICTIONS_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segments_df\segments_final_predictions_v3_loso_global_residual.parquet" 
OUTPUT_DIR = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\evaluation_results"

TRIP_ID_COL = 'trip_id'
SEGMENT_ID_COL = 'segment_id'
GLOBAL_ERROR_COL = 'residual_error_soc_seg_delta'
PERSONAL_ERROR_COL = 'personal_final_error'
GT_COL = 'gt_soc_seg_delta'
GLOBAL_PRED_COL = 'global_pred_soc_seg_delta' # Optional, for context
PERSONAL_PRED_COL = 'personal_pred_soc_seg_delta' # Optional, for context

FEATURES_FOR_ERROR_ANALYSIS = [
    'distance_seg_actual_km', 'duration_seg_actual_s', 'soc_seg_start',
    'speed_seg_agg_mean_mps', 'stops_seg_count', 'accel_seg_agg_mean', 'decel_seg_agg_mean',
    'route_deviation_pct', 'percent_time_over_limit_seg_agg', 'match_confidence',
    'hour_seg_start', 'dayofweek_seg_start'
]
# CATEGORICAL_FEATURES_FOR_BREAKDOWN is used by make_error_feature_plots, keep it if that function is used.

PLOT_BINS = 30
PLOT_MAX_ERROR = 10
SAVE_PLOTS = True
SHOW_PLOTS = True
# --------------------------

# --- Metric Calculation Helper (No change needed) ---
def calculate_error_metrics(errors_series, model_name="Model"):
    # ... (function remains the same) ...
    metrics = {'MAE': np.nan, 'RMSE': np.nan, 'MBE': np.nan, 'StdDev': np.nan, 'Count': 0}
    if errors_series is None or errors_series.empty:
        return metrics
    if errors_series.isnull().all():
        return metrics
    errors_valid = errors_series.dropna()
    metrics['Count'] = len(errors_valid)
    if metrics['Count'] == 0:
         return metrics
    try:
        errors_numeric = pd.to_numeric(errors_valid, errors='coerce').dropna()
        if errors_numeric.empty:
             print(f"Warning: Error series for {model_name} has no valid numeric values after coercion/dropna.")
             return metrics
        metrics['Count'] = len(errors_numeric)
        target_zeros = np.zeros_like(errors_numeric)
        metrics['MAE'] = mean_absolute_error(target_zeros, errors_numeric)
        metrics['RMSE'] = np.sqrt(mean_squared_error(target_zeros, errors_numeric))
        metrics['MBE'] = errors_numeric.mean()
        metrics['StdDev'] = errors_numeric.std()
    except Exception as e:
        print(f"Error calculating metrics for {model_name}: {e}")
        metrics = {'MAE': np.nan, 'RMSE': np.nan, 'MBE': np.nan, 'StdDev': np.nan, 'Count': 0}
    return metrics

# --- Plotting Helpers (No change needed for these two) ---
# --- Plotting Helper (Modified) ---
def plot_distribution(data_series, title, filename, output_dir,
                      xlabel_override=None, show_zero_line=False, # New parameters
                      xlim=None, bins=PLOT_BINS): # Added bins parameter
    """Plots a histogram and KDE of a data series distribution."""
    if data_series is None or data_series.dropna().empty:
        print(f"Skipping plot '{title}': No valid data.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(data_series.dropna(), kde=True, bins=bins, stat='density', label='Distribution')
    plt.title(title)
    
    if xlabel_override:
        plt.xlabel(xlabel_override)
    else: # Default for error plots (if you still want to use it for that)
        plt.xlabel("Prediction Error (Actual - Prediction)")
        
    plt.ylabel("Density")

    if show_zero_line: # Only show if it's an error plot or meaningful
        plt.axvline(0, color='red', linestyle='--', label='Zero Reference')

    # Add lines for mean and +/- std dev if calculable
    mean_val = data_series.mean()
    std_val = data_series.std()
    if pd.notna(mean_val):
        plt.axvline(mean_val, color='black', linestyle=':', lw=1.5, label=f'Mean ({mean_val:.2f})')
        if pd.notna(std_val):
            plt.axvline(mean_val + std_val, color='grey', linestyle=':', lw=1, label=f'+1 Std Dev ({mean_val + std_val:.2f})')
            plt.axvline(mean_val - std_val, color='grey', linestyle=':', lw=1, label=f'-1 Std Dev ({mean_val - std_val:.2f})')
    elif pd.notna(std_val): # Plot std dev around 0 if mean is NaN (less likely for raw data)
         plt.axvline(std_val, color='grey', linestyle=':', lw=1, label=f'+1 Std Dev ({std_val:.2f})')
         plt.axvline(-std_val, color='grey', linestyle=':', lw=1, label=f'-1 Std Dev ({-std_val:.2f})')

    if xlim: plt.xlim(left=-xlim, right=xlim)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if SAVE_PLOTS:
        plot_path = os.path.join(output_dir, filename)
        try:
            plt.savefig(plot_path, dpi=150)
            print(f" - Saved plot: {plot_path}")
        except Exception as e: print(f"Warning: Failed to save plot {plot_path}. Error: {e}")
    if SHOW_PLOTS: plt.show()
    plt.close()

def make_error_feature_plots(df, error_col=PERSONAL_ERROR_COL, features=FEATURES_FOR_ERROR_ANALYSIS,
                             page_size=6, max_err=PLOT_MAX_ERROR, method_label=METHOD_BEING_EVALUATED.upper(),
                             out_dir=OUTPUT_DIR, fname_prefix="err_vs_feat"):
    # ... (function remains the same) ...
    sns.set_style("whitegrid"); sns.set_context("talk")
    cont_feats, disc_feats = [], []
    for f in features:
        if f not in df.columns: warnings.warn(f"Feature '{f}' not in dataframe – skipped."); continue
        if pd.api.types.is_numeric_dtype(df[f]) and df[f].nunique() > 15: cont_feats.append(f)
        else: disc_feats.append(f)
    def _plot_batch(feat_list, page_tag):
        if not feat_list: return
        n_pages = math.ceil(len(feat_list)/page_size)
        pdf_path = os.path.join(out_dir, f"{fname_prefix}_{page_tag}.pdf")
        with PdfPages(pdf_path) as pdf:
            for p in range(n_pages):
                batch = feat_list[p*page_size:(p+1)*page_size]
                cols = min(3, len(batch)); rows = math.ceil(len(batch)/cols)
                fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.5*rows), squeeze=False)
                axes = axes.flatten()
                for ax, feat in zip(axes, batch):
                    ser = df[feat]; samp = df[[feat, error_col]].sample(min(len(df), 4000), random_state=42)
                    if feat in cont_feats:
                        sns.regplot(x=feat, y=error_col, data=samp, ax=ax, scatter_kws=dict(alpha=.12, s=12), line_kws=dict(color='red', lw=1.3), lowess=True)
                    else:
                        uniq = ser.nunique()
                        if pd.api.types.is_numeric_dtype(ser) and 10 < uniq <= 30:
                            ser_binned = pd.cut(ser, 5, precision=0, include_lowest=True)
                            sns.boxplot(x=ser_binned, y=error_col, data=df, ax=ax, showfliers=False)
                            ax.set_xticklabels([t.get_text().replace(",", ",\n") for t in ax.get_xticklabels()], rotation=0)
                        else:
                            sns.boxplot(x=feat, y=error_col, data=df, ax=ax, showfliers=False)
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
                    ax.axhline(0, color="black", ls="--", lw=1); ax.set_ylim(-max_err, max_err)
                    title = textwrap.fill(f"{feat} ({method_label})", 25); ax.set_title(title, fontsize=11)
                    ax.set_xlabel(""); ax.set_ylabel(error_col if (axes.tolist().index(ax) % cols) == 0 else "")
                for ax in axes[len(batch):]: ax.axis("off")
                fig.tight_layout(); pdf.savefig(fig, dpi=200); plt.close(fig)
        print(f"✓ Saved {pdf_path}")
    _plot_batch(cont_feats, "continuous"); _plot_batch(disc_feats,  "categorical")

# Add this new function to your eval.py script

def make_absolute_error_feature_plots(df,
                                      error_col=PERSONAL_ERROR_COL, # Original signed error column
                                      features=FEATURES_FOR_ERROR_ANALYSIS,
                                      page_size=6,
                                      max_abs_err=PLOT_MAX_ERROR, # Max for y-axis, now non-negative
                                      method_label=METHOD_BEING_EVALUATED.upper(),
                                      out_dir=OUTPUT_DIR,
                                      fname_prefix="abs_err_vs_feat"): # New prefix
    """Create a multi-page PDF where each page has ≤page_size feature plots vs. ABSOLUTE error."""
    
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Create an absolute error column for plotting
    abs_error_col_name = f"abs_{error_col}"
    if error_col not in df.columns:
        warnings.warn(f"Original error column '{error_col}' not found in DataFrame. Skipping absolute error plots.")
        return
    
    df_plot = df.copy() # Work on a copy
    df_plot[abs_error_col_name] = df_plot[error_col].abs()

    cont_feats, disc_feats = [], []
    for f in features:
        if f not in df_plot.columns:
            warnings.warn(f"Feature '{f}' not in dataframe – skipped for absolute error plots.")
            continue
        if pd.api.types.is_numeric_dtype(df_plot[f]) and df_plot[f].nunique() > 15:
            cont_feats.append(f)
        else:
            disc_feats.append(f)

    def _plot_batch(feat_list, page_tag):
        if not feat_list:
            return
        n_pages = math.ceil(len(feat_list)/page_size)

        pdf_path = os.path.join(out_dir, f"{fname_prefix}_{page_tag}.pdf")
        with PdfPages(pdf_path) as pdf:
            for p in range(n_pages):
                batch = feat_list[p*page_size:(p+1)*page_size]
                cols = min(3, len(batch))
                rows = math.ceil(len(batch)/cols)

                fig, axes = plt.subplots(rows, cols,
                                         figsize=(6*cols, 4.5*rows),
                                         squeeze=False)
                axes = axes.flatten()

                for ax, feat in zip(axes, batch):
                    ser = df_plot[feat]
                    # Sample for plotting, now using the absolute error column
                    samp = df_plot[[feat, abs_error_col_name]].sample(
                        min(len(df_plot), 4000), random_state=42)
                    
                    if feat in cont_feats:
                        sns.regplot(x=feat, y=abs_error_col_name, data=samp, # Use absolute error
                                    ax=ax, scatter_kws=dict(alpha=.12, s=12),
                                    line_kws=dict(color='green', lw=1.3), # Different color for distinction
                                    lowess=True)
                    else:
                        uniq = ser.nunique()
                        if pd.api.types.is_numeric_dtype(ser) and 10 < uniq <= 30:
                            ser_binned = pd.cut(ser, 5, precision=0, include_lowest=True)
                            sns.boxplot(x=ser_binned, y=abs_error_col_name, # Use absolute error
                                        data=df_plot, ax=ax, showfliers=False,
                                        color="lightgreen") # Different color
                            ax.set_xticklabels(
                                [t.get_text().replace(",", ",\n") for t in ax.get_xticklabels()],
                                rotation=0)
                        else:
                            sns.boxplot(x=feat, y=abs_error_col_name, # Use absolute error
                                        data=df_plot, ax=ax, showfliers=False,
                                        color="lightgreen") # Different color
                            ax.set_xticklabels(ax.get_xticklabels(),
                                               rotation=30, ha='right')

                    # ax.axhline(0, color="black", ls="--", lw=1) # Baseline is 0 for absolute error
                    ax.set_ylim(0, max_abs_err) # Y-axis starts at 0 for absolute error
                    title_text = f"Abs Error vs {feat} ({method_label})"
                    title = textwrap.fill(title_text, 25)
                    ax.set_title(title, fontsize=11)
                    ax.set_xlabel("")
                    ax.set_ylabel(f"Absolute Error ({abs_error_col_name.split('_', 1)[1]})" if (axes.tolist().index(ax) % cols) == 0 else "")


                for ax_unused in axes[len(batch):]:
                    ax_unused.axis("off")

                fig.tight_layout()
                pdf.savefig(fig, dpi=200)
                plt.close(fig)
        print(f"✓ Saved Absolute Error PDF: {pdf_path}")

    _plot_batch(cont_feats, "continuous_abs_error")
    _plot_batch(disc_feats,  "categorical_abs_error")

# --- New Helper Functions for main() ---
def load_and_validate_data(filepath, required_cols_eval, optional_cols_eval, error_cols_numeric, gt_col_numeric):
    """Loads data and performs initial validation and type conversion."""
    print(f"\n--- Loading Final Predictions Data ---")
    print(f"Path: {filepath}")
    try:
        df = load_file(filepath)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Loaded data is not a DataFrame.")
        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns found: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}"); traceback.print_exc()
        return None

    print("\n--- Validating Required Columns & Data Types ---")
    cols_to_check = required_cols_eval + [col for col in optional_cols_eval if col in df.columns]
    missing_cols = [col for col in cols_to_check if col not in df.columns]
    if any(col in missing_cols for col in required_cols_eval):
        print(f"Error: DataFrame is missing REQUIRED columns: {[col for col in required_cols_eval if col in missing_cols]}")
        return None
    elif missing_cols:
        print(f"Warning: Optional columns missing: {missing_cols}")

    print(f" - Converting error columns to numeric...")
    try:
        for err_col in error_cols_numeric:
            if err_col in df.columns:
                df[err_col] = pd.to_numeric(df[err_col], errors='coerce')
            else:
                raise KeyError(f"Error column {err_col} not found for numeric conversion.")
        if gt_col_numeric and gt_col_numeric in df.columns:
            df[gt_col_numeric] = pd.to_numeric(df[gt_col_numeric], errors='coerce')
    except KeyError as e:
        print(f"Error: {e}."); return None
    except Exception as e:
        print(f"Error converting columns to numeric: {e}"); return None
    print("Required columns validated and converted.")
    return df

def filter_data_for_evaluation(df, global_err_col, personal_err_col):
    """Filters DataFrame to rows where both specified error columns are non-NaN."""
    print("\n--- Filtering Data for Evaluation ---")
    initial_rows = len(df)
    eval_df = df.dropna(subset=[global_err_col, personal_err_col]).copy()
    rows_dropped = initial_rows - len(eval_df)
    print(f" - Comparing on {len(eval_df)} segments where both global and final personal errors are non-NaN.")
    if rows_dropped > 0:
        print(f" - Excluded {rows_dropped} segments due to NaN errors.")
    if eval_df.empty:
        print("Error: No valid segments found for evaluation after filtering.")
        return None
    return eval_df

def calculate_and_display_overall_metrics(eval_df, global_err_col, personal_err_col, output_dir_path):
    """Calculates, displays, and saves overall performance metrics."""
    print("\n--- Overall Performance Metrics ---")
    overall_results_dict = {}
    evaluation_summary_list = []

    print(f"\nGlobal Model Baseline (Error: {global_err_col}):")
    global_m = calculate_error_metrics(eval_df[global_err_col], "Global")
    overall_results_dict['Global'] = global_m
    evaluation_summary_list.append({'Model': 'Global Baseline', **global_m})
    for metric, value in global_m.items(): print(f"  Overall {metric}: {value:.4f}")

    print(f"\nFinal Personal Model (Error: {personal_err_col}):")
    personal_m = calculate_error_metrics(eval_df[personal_err_col], "Personal")
    overall_results_dict['Personal'] = personal_m
    evaluation_summary_list.append({'Model': 'Final Personal', **personal_m})
    for metric, value in personal_m.items(): print(f"  Overall {metric}: {value:.4f}")

    print("\nComparison (Final Personal vs Global Baseline):")
    for metric in ['MAE', 'RMSE', 'MBE', 'StdDev']:
        if metric not in personal_m or metric not in global_m or \
           np.isnan(personal_m[metric]) or np.isnan(global_m[metric]):
            print(f"  Cannot compare {metric} due to missing/NaN values.")
            continue
        diff = personal_m[metric] - global_m[metric]
        change_pct = (diff / global_m[metric]) * 100 if abs(global_m[metric]) > 1e-9 else (np.inf if diff != 0 else 0)
        is_better = (metric != 'MBE' and diff < -1e-9) or \
                    (metric == 'MBE' and abs(personal_m[metric]) < abs(global_m[metric]) - 1e-9)
        is_worse = (metric != 'MBE' and diff > 1e-9) or \
                   (metric == 'MBE' and abs(personal_m[metric]) > abs(global_m[metric]) + 1e-9)
        comparison = "Better" if is_better else "Worse" if is_worse else "Same"
        print(f"  {metric:<7}: Personal={personal_m[metric]:<8.4f} Global={global_m[metric]:<8.4f} | Diff={diff:<+8.4f} ({change_pct:<+7.2f}%) -> Personal is {comparison}")

    try:
        summary_df = pd.DataFrame(evaluation_summary_list)
        summary_path = os.path.join(output_dir_path, 'overall_evaluation_summary.csv')
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"\n - Saved overall evaluation summary to: {summary_path}")
    except Exception as e: print(f"Warning: Could not save overall summary CSV. Error: {e}")
    return global_m, personal_m

def plot_overall_distributions_wrapper(eval_df, global_err_col, personal_err_col, global_m, personal_m, output_dir_path, max_err_plot):
    """Wrapper to plot overall error distributions for global and personal models."""
    print("\n--- Plotting Overall Error Distributions ---")
    plot_error_distribution(eval_df[global_err_col],
                            f'Overall Global Model Error Distribution\n(N={global_m["Count"]}, MAE={global_m["MAE"]:.2f}, RMSE={global_m["RMSE"]:.2f}, MBE={global_m["MBE"]:.2f})',
                            'overall_global_error_dist.png', output_dir_path, xlim=max_err_plot)
    plot_error_distribution(eval_df[personal_err_col],
                            f'Overall Final Personal Model Error Distribution\n(N={personal_m["Count"]}, MAE={personal_m["MAE"]:.2f}, RMSE={personal_m["RMSE"]:.2f}, MBE={personal_m["MBE"]:.2f})',
                            'overall_personal_error_dist.png', output_dir_path, xlim=max_err_plot)

def perform_per_driver_analysis(eval_df, trip_id_c, personal_err_c, global_err_c, gt_c, output_dir_path):
    """Performs per-driver (trip) analysis including metrics, improvements, and plots."""
    print("\n--- Per-Driver Performance Analysis ---")

    def get_driver_metrics_internal(group):
        metrics = {}
        metrics.update({f'personal_{k.lower()}': v for k, v in calculate_error_metrics(group[personal_err_c], "Personal").items()})
        metrics.update({f'global_{k.lower()}': v for k, v in calculate_error_metrics(group[global_err_c], "Global").items()})
        metrics['segment_count'] = len(group)
        if gt_c and gt_c in group.columns: metrics['avg_gt_soc_delta'] = group[gt_c].mean()
        return pd.Series(metrics)

    driver_metrics_df = eval_df.groupby(trip_id_c).apply(get_driver_metrics_internal)
    driver_metrics_df.dropna(subset=['personal_mae', 'global_mae'], how='any', inplace=True)
    print(f"Calculated metrics for {len(driver_metrics_df)} drivers with valid comparison data.")

    if driver_metrics_df.empty:
        print("Skipping per-driver analysis: No valid driver metrics.")
        return

    print("\nSample of Per-Driver Metrics:"); print(driver_metrics_df.head().round(4))
    driver_metrics_df['mae_improvement'] = driver_metrics_df['global_mae'] - driver_metrics_df['personal_mae']
    valid_improvements = driver_metrics_df['mae_improvement'].dropna()

    if not valid_improvements.empty:
        improved_drivers = (valid_improvements > 1e-6).sum()
        worsened_drivers = (valid_improvements < -1e-6).sum()
        total_drivers = len(valid_improvements)
        same_drivers = total_drivers - improved_drivers - worsened_drivers
        print(f"\nDriver MAE Comparison (Final Personal vs Global):")
        print(f"  Drivers with lower MAE (Improved): {improved_drivers} ({improved_drivers/total_drivers:.1%})")
        print(f"  Drivers with higher MAE (Worsened): {worsened_drivers} ({worsened_drivers/total_drivers:.1%})")
        print(f"  Drivers with approx. same MAE:      {same_drivers} ({same_drivers/total_drivers:.1%})")
        # ... (print other improvement stats) ...
        print(f"  Average MAE Improvement per driver: {valid_improvements.mean():.4f}")
        print(f"  Median MAE Improvement per driver: {valid_improvements.median():.4f}")

        try:
            driver_metrics_path = os.path.join(output_dir_path, 'per_driver_evaluation_metrics.csv')
            driver_metrics_df.round(4).to_csv(driver_metrics_path, index=True)
            print(f"\n - Saved per-driver evaluation metrics to: {driver_metrics_path}")
        except Exception as e: print(f"Warning: Could not save per-driver metrics CSV. Error: {e}")

        print("\n--- Plotting Per-Driver Results ---")
        try:
            plot_error_distribution(valid_improvements,
                                    f'Distribution of MAE Improvement per Driver (N={total_drivers})\n(Global MAE - Final Personal MAE)',
                                    'per_driver_mae_improvement_dist.png', output_dir_path, xlim=None)
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=driver_metrics_df[['personal_mae', 'global_mae']])
            plt.title('Distribution of MAE per Driver'); plt.ylabel('Mean Absolute Error (MAE)')
            plt.xticks(ticks=[0,1], labels=['Final Personal', 'Global Baseline']); plt.grid(True, axis='y', alpha=0.3); plt.tight_layout()
            if SAVE_PLOTS: plt.savefig(os.path.join(output_dir_path, 'per_driver_mae_boxplot.png'), dpi=150); print(" - Saved MAE boxplot.")
            if SHOW_PLOTS: plt.show()
            plt.close()

            plt.figure(figsize=(8, 8))
            max_lim = max(driver_metrics_df['personal_mae'].max(skipna=True), driver_metrics_df['global_mae'].max(skipna=True)) * 1.1
            min_lim = 0
            plt.scatter(driver_metrics_df['global_mae'], driver_metrics_df['personal_mae'], alpha=0.4, s=15, label='Driver')
            plt.plot([min_lim, max_lim], [min_lim, max_lim], color='red', linestyle='--', lw=1, label='y=x (No Change)')
            plt.xlabel('Global Model MAE per Driver'); plt.ylabel('Final Personal Model MAE per Driver')
            plt.title(f'Per-Driver MAE Comparison (N={total_drivers})'); plt.grid(True, alpha=0.3)
            plt.xlim(min_lim, max_lim); plt.ylim(min_lim, max_lim); plt.legend(); plt.gca().set_aspect('equal', adjustable='box'); plt.tight_layout()
            if SAVE_PLOTS: plt.savefig(os.path.join(output_dir_path, 'per_driver_mae_scatter.png'), dpi=150); print(" - Saved MAE scatter plot.")
            if SHOW_PLOTS: plt.show()
            plt.close()
        except Exception as e: warnings.warn(f"Could not generate per-driver plots. Error: {e}")
    else: print("Skipping MAE improvement plotting due to no valid improvement data.")

def plot_error_vs_features_combined(eval_df_in, features_list, error_col_name, method_label_str, output_dir_path):
    """Plots personal model error vs. a list of individual features in a combined PNG."""
    print("\n--- Plotting Error vs. Individual Segment Features (Combined PNG) ---")
    if not (PERSONAL_ERROR_COL in eval_df_in.columns):
        print(" - Skipping error vs features plot: Personal error column not available.")
        return

    features_to_analyze_info = []
    for f_name in features_list:
        if f_name in eval_df_in.columns:
            features_to_analyze_info.append({'name': f_name, 'dtype': eval_df_in[f_name].dtype})
        else:
            warnings.warn(f"Feature '{f_name}' for error analysis not found in eval_df. Skipping.")

    if not features_to_analyze_info:
        print(" - Skipping: None of the specified features for error analysis found in eval_df.")
        return

    num_plots = len(features_to_analyze_info)
    cols_subplot = 3
    rows_subplot = (num_plots + cols_subplot - 1) // cols_subplot
    fig = plt.figure(figsize=(6 * cols_subplot, 5 * rows_subplot))

    for i, feature_info in enumerate(features_to_analyze_info):
        feature = feature_info['name']
        dtype = feature_info['dtype']
        ax = fig.add_subplot(rows_subplot, cols_subplot, i + 1)
        sample_size = min(len(eval_df_in), 2000)
        sample_df = eval_df_in.sample(sample_size, random_state=42) if sample_size < len(eval_df_in) else eval_df_in
        is_numeric_continuous = pd.api.types.is_numeric_dtype(dtype) and sample_df[feature].nunique() > 15

        if is_numeric_continuous:
            try:
                sns.regplot(x=sample_df[feature], y=sample_df[error_col_name], ax=ax, scatter_kws={'alpha':0.1, 's':10, 'edgecolor':None}, line_kws={'color':'red', 'lw':1.5}, lowess=True)
            except Exception as e_reg:
                warnings.warn(f"Regplot failed for {feature}: {e_reg}. Falling back to scatter.")
                sns.scatterplot(x=sample_df[feature], y=sample_df[error_col_name], ax=ax, alpha=0.2, s=10, edgecolor=None)
        elif pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype) or eval_df_in[feature].nunique() <=15 :
            if pd.api.types.is_numeric_dtype(dtype) and 10 < eval_df_in[feature].nunique() <= 30:
                try:
                    min_val, max_val = eval_df_in[feature].min(), eval_df_in[feature].max()
                    if pd.notna(min_val) and pd.notna(max_val) and min_val != max_val:
                        bins = np.linspace(min_val, max_val, 6)
                        binned_feature_series = pd.cut(eval_df_in[feature], bins=bins, precision=0, include_lowest=True)
                        sns.boxplot(x=binned_feature_series, y=eval_df_in[error_col_name], ax=ax, showfliers=False)
                    else: sns.stripplot(x=eval_df_in[feature], y=eval_df_in[error_col_name], ax=ax, alpha=0.3, jitter=0.2)
                except Exception as e_binbox:
                    warnings.warn(f"Bin/boxplot failed for {feature}: {e_binbox}. Using stripplot.")
                    sns.stripplot(x=eval_df_in[feature], y=eval_df_in[error_col_name], ax=ax, alpha=0.3, jitter=0.2)
            else: sns.boxplot(x=eval_df_in[feature], y=eval_df_in[error_col_name], ax=ax, showfliers=False)
            ax.tick_params(axis='x', rotation=30, labelsize=8)
        else:
            ax.text(0.5, 0.5, f"'{feature}'\n(dtype: {dtype})\nPlot type TBD", ha='center', va='center', transform=ax.transAxes)
            warnings.warn(f"Feature '{feature}' (dtype {dtype}) not handled by scatter/box. Skipping detailed plot.")

        if ax.has_data(): # Check if any data was plotted
            ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.7)
            ax.set_title(f'Error vs {feature}\n({method_label_str.upper()})', fontsize=10)
            ax.set_xlabel(feature, fontsize=9); ax.set_ylabel(error_col_name, fontsize=9); ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=2.5)
    if SAVE_PLOTS:
        plot_path = os.path.join(output_dir_path, f'error_vs_all_features_{method_label_str}.png')
        try: fig.savefig(plot_path, dpi=150); print(f" - Saved combined error vs features plot: {plot_path}")
        except Exception as e_save: print(f"Warning: Failed to save combined plot {plot_path}. Error: {e_save}")
    if SHOW_PLOTS: plt.show()
    plt.close(fig)

# --- Main Orchestration Logic ---
def main():
    print("--- Starting Evaluation Script ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved in: {OUTPUT_DIR}")

    segments_df = load_and_validate_data(
        FINAL_PREDICTIONS_PATH,
        required_cols_eval=[TRIP_ID_COL, SEGMENT_ID_COL, GLOBAL_ERROR_COL, PERSONAL_ERROR_COL],
        optional_cols_eval=[GT_COL, GLOBAL_PRED_COL, PERSONAL_PRED_COL],
        error_cols_numeric=[GLOBAL_ERROR_COL, PERSONAL_ERROR_COL],
        gt_col_numeric=GT_COL
    )
    if segments_df is None: return

    eval_df = filter_data_for_evaluation(segments_df, GLOBAL_ERROR_COL, PERSONAL_ERROR_COL)
    if eval_df is None: return

    # --- Plot Delta SoC Distributions (NEW PLOTS) ---
    print("\n--- Plotting Delta SoC Distributions ---")
    if GT_COL in eval_df.columns:
        # 1. Plot of delta SoC across all segments
        plot_distribution(
            eval_df[GT_COL],
            title=f'Distribution of Actual Segment SoC Change (N={eval_df[GT_COL].notna().sum()})',
            filename='segment_gt_soc_delta_dist.png',
            output_dir=OUTPUT_DIR,
            xlabel_override='Actual SoC Change per Segment (%)',
            show_zero_line=False, # Not an error plot
            xlim=None # Let it auto-scale or set a reasonable xlim if needed
        )

        # 2. Plot of delta SoC across all trips
        if TRIP_ID_COL in eval_df.columns:
            trip_delta_soc = eval_df.groupby(TRIP_ID_COL)[GT_COL].sum()
            plot_distribution(
                trip_delta_soc,
                title=f'Distribution of Actual Trip SoC Change (N={len(trip_delta_soc)})',
                filename='trip_gt_soc_delta_dist.png',
                output_dir=OUTPUT_DIR,
                xlabel_override='Actual SoC Change per Trip (%)',
                show_zero_line=False,
                xlim=None
            )
        else:
            print(f"Warning: Trip ID column '{TRIP_ID_COL}' not found. Skipping per-trip delta SoC plot.")
    else:
        print(f"Warning: Ground Truth column '{GT_COL}' not found. Skipping delta SoC plots.")
    # --- END OF NEW PLOTS ---


    global_metrics, personal_metrics = calculate_and_display_overall_metrics(
        eval_df, GLOBAL_ERROR_COL, PERSONAL_ERROR_COL, OUTPUT_DIR
    )

    # Use the new plot_distribution for error plots as well
    print("\n--- Plotting Overall Error Distributions ---")
    plot_distribution(eval_df[GLOBAL_ERROR_COL],
                            f'Overall Global Model Error Distribution\n(N={global_metrics["Count"]}, MAE={global_metrics["MAE"]:.2f}, RMSE={global_metrics["RMSE"]:.2f}, MBE={global_metrics["MBE"]:.2f})',
                            'overall_global_error_dist.png', OUTPUT_DIR,
                            xlabel_override="Global Model Prediction Error (Actual - Prediction)", # Specific label
                            show_zero_line=True, # It's an error plot
                            xlim=PLOT_MAX_ERROR)
    plot_distribution(eval_df[PERSONAL_ERROR_COL],
                            f'Overall Final Personal Model Error Distribution\n(N={personal_metrics["Count"]}, MAE={personal_metrics["MAE"]:.2f}, RMSE={personal_metrics["RMSE"]:.2f}, MBE={personal_metrics["MBE"]:.2f})',
                            'overall_personal_error_dist.png', OUTPUT_DIR,
                            xlabel_override="Personal Model Prediction Error (Actual - Prediction)", # Specific label
                            show_zero_line=True, # It's an error plot
                            xlim=PLOT_MAX_ERROR)


    perform_per_driver_analysis(
        eval_df, TRIP_ID_COL, PERSONAL_ERROR_COL, GLOBAL_ERROR_COL, GT_COL, OUTPUT_DIR
    )

    plot_error_vs_features_combined(
        eval_df, FEATURES_FOR_ERROR_ANALYSIS, PERSONAL_ERROR_COL, METHOD_BEING_EVALUATED, OUTPUT_DIR
    )

    print("\n--- Generating Multi-Page PDF for Error vs Features ---")
    make_error_feature_plots(
        eval_df,
        error_col=PERSONAL_ERROR_COL,
        features=FEATURES_FOR_ERROR_ANALYSIS,
        max_err=PLOT_MAX_ERROR,
        method_label=METHOD_BEING_EVALUATED.upper(),
        out_dir=OUTPUT_DIR
    )

    # --- NEW: Call for Absolute Error Plots ---
    print("\n--- Generating Multi-Page PDF for Absolute Error vs Features ---")
    make_absolute_error_feature_plots(
        eval_df,
        error_col=PERSONAL_ERROR_COL, # Still pass the original signed error column
        features=FEATURES_FOR_ERROR_ANALYSIS,
        max_abs_err=PLOT_MAX_ERROR, # Max y-value for absolute error (can be same as signed for consistency or adjusted)
        method_label=METHOD_BEING_EVALUATED.upper(),
        out_dir=OUTPUT_DIR,
        fname_prefix="abs_err_vs_feat" # Distinct prefix
    )
    # --- END OF NEW CALL ---

    print("\n--- Evaluation Complete ---")

if __name__ == '__main__':
    main()
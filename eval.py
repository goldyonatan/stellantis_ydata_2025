import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    from HelperFuncs import load_file # Assuming this exists and handles parquet/pickle
except ImportError:
    print("Error: Could not import from HelperFuncs. Ensure HelperFuncs.py is accessible.")
    def load_file(filepath, **kwargs): raise NotImplementedError("HelperFuncs not found")

# --- Configuration ---
# *** Path to the file saved by the fit_predict script (contains final predictions) ***
METHOD_BEING_EVALUATED = 'loso' # CHANGE THIS TO 'walk_forward' or 'loso'
FINAL_PREDICTIONS_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segments_df\segments_final_predictions_v3_loso.parquet"
OUTPUT_DIR = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\evaluation_results"

# --- Define Column Names (MUST match output of fit_predict script) ---
TRIP_ID_COL = 'trip_id'
SEGMENT_ID_COL = 'segment_id'
# Error column for the standalone global model (Actual GT - Global Prediction)
# This name comes from the feature extraction script
GLOBAL_ERROR_COL = 'residual_error_soc_seg_delta' # Error = GT - GlobalPred
# Error column for the final combined personal model (Actual GT - Final Personal Prediction)
# This name comes from the fit_predict script
PERSONAL_ERROR_COL = 'personal_final_error' # Error = GT - (GlobalPred + PersonalResidualPred)
# Ground truth column (optional, but good for context/sanity checks)
GT_COL = 'gt_soc_seg_delta'
# Prediction columns (optional, for scatter plots etc.)
GLOBAL_PRED_COL = 'global_pred_soc_seg_delta'
PERSONAL_PRED_COL = 'personal_pred_soc_seg_delta'
# --------------------------------------------------------------------
# Add some key features from segments_df for error analysis
# These are ACTUAL segment properties, not historical ones used for training the personal model
FEATURES_FOR_ERROR_ANALYSIS = [
    'distance_seg_actual_km', 'duration_seg_actual_s', 'soc_seg_start',
    'speed_seg_agg_mean_mps', 'stops_seg_count', 'accel_seg_agg_mean', 'decel_seg_agg_mean',
    'route_deviation_pct', 'percent_time_over_limit_seg_agg', 'match_confidence',
    'hour_seg_start', 'dayofweek_seg_start'
]
CATEGORICAL_FEATURES_FOR_BREAKDOWN = [
    'hour_seg_start', # Will be binned
    'dayofweek_seg_start',
    'car_model_trip', # If present
    'match_status', # From feature_extraction
]
# --------------------------------------------------------------------

# --- Plotting Configuration ---
PLOT_BINS = 30 # Reduced for potentially smaller groups
PLOT_MAX_ERROR = 10
SAVE_PLOTS = True
SHOW_PLOTS = True
# --------------------------

# --- Helper Function for Metrics ---
def calculate_error_metrics(errors_series, model_name="Model"):
    """Calculates MAE, RMSE, MBE, StdDev, and Count for a series of errors."""
    metrics = {'MAE': np.nan, 'RMSE': np.nan, 'MBE': np.nan, 'StdDev': np.nan, 'Count': 0}
    if errors_series is None or errors_series.empty:
        # warnings.warn(f"Cannot calculate metrics for {model_name}: Input error series is empty.") # Less verbose
        return metrics
    if errors_series.isnull().all():
        # warnings.warn(f"Cannot calculate metrics for {model_name}: Input error series contains only NaN values.") # Less verbose
        return metrics

    # Drop NaNs before calculation
    errors_valid = errors_series.dropna()
    metrics['Count'] = len(errors_valid)
    if metrics['Count'] == 0:
         # warnings.warn(f"Cannot calculate metrics for {model_name}: No valid (non-NaN) errors found.") # Less verbose
         return metrics

    try:
        # Ensure errors are numeric
        errors_numeric = pd.to_numeric(errors_valid, errors='coerce').dropna()
        if errors_numeric.empty:
             print(f"Warning: Error series for {model_name} has no valid numeric values after coercion/dropna.")
             return metrics
        metrics['Count'] = len(errors_numeric) # Update count after potential coercion drop

        # Calculate metrics comparing errors to zero (ideal error)
        target_zeros = np.zeros_like(errors_numeric)
        metrics['MAE'] = mean_absolute_error(target_zeros, errors_numeric)
        metrics['RMSE'] = np.sqrt(mean_squared_error(target_zeros, errors_numeric))
        metrics['MBE'] = errors_numeric.mean() # Mean Bias Error (Prediction - Actual) -> (Error)
        metrics['StdDev'] = errors_numeric.std() # Standard Deviation of Error

    except Exception as e:
        print(f"Error calculating metrics for {model_name}: {e}")
        # Reset metrics to NaN on error
        metrics = {'MAE': np.nan, 'RMSE': np.nan, 'MBE': np.nan, 'StdDev': np.nan, 'Count': 0}

    return metrics

# --- Plotting Helper ---
def plot_error_distribution(errors_series, title, filename, output_dir, xlim=None):
    """Plots a histogram and KDE of the error distribution."""
    if errors_series is None or errors_series.dropna().empty:
        print(f"Skipping plot '{title}': No valid data.")
        return

    plt.figure(figsize=(10, 6))
    # Use dropna() within histplot to handle potential remaining NaNs gracefully
    sns.histplot(errors_series.dropna(), kde=True, bins=PLOT_BINS, stat='density', label='Error Distribution')
    plt.title(title)
    plt.xlabel("Prediction Error (Actual - Prediction)")
    plt.ylabel("Density")
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')

    # Add lines for mean and +/- std dev if calculable
    mean_err = errors_series.mean()
    std_err = errors_series.std()
    if pd.notna(mean_err):
        plt.axvline(mean_err, color='black', linestyle=':', lw=1.5, label=f'Mean Error ({mean_err:.2f})')
        if pd.notna(std_err):
            plt.axvline(mean_err + std_err, color='grey', linestyle=':', lw=1, label=f'+1 Std Dev ({mean_err + std_err:.2f})')
            plt.axvline(mean_err - std_err, color='grey', linestyle=':', lw=1, label=f'-1 Std Dev ({mean_err - std_err:.2f})')
    elif pd.notna(std_err): # Plot std dev around 0 if mean is NaN
         plt.axvline(std_err, color='grey', linestyle=':', lw=1, label=f'+1 Std Dev ({std_err:.2f})')
         plt.axvline(-std_err, color='grey', linestyle=':', lw=1, label=f'-1 Std Dev ({-std_err:.2f})')

    if xlim: plt.xlim(left=-xlim, right=xlim)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if SAVE_PLOTS:
        plot_path = os.path.join(output_dir, filename)
        try:
            plt.savefig(plot_path, dpi=150) # Increase dpi for better quality
            print(f" - Saved plot: {plot_path}")
        except Exception as e: print(f"Warning: Failed to save plot {plot_path}. Error: {e}")
    if SHOW_PLOTS: plt.show()
    plt.close() # Close plot to free memory


def make_error_feature_plots(df,  # <- eval_df in your script
                             error_col=PERSONAL_ERROR_COL,
                             features=FEATURES_FOR_ERROR_ANALYSIS,
                             page_size=6,            # how many sub-plots per figure
                             max_err=PLOT_MAX_ERROR,
                             method_label=METHOD_BEING_EVALUATED.upper(),
                             out_dir=OUTPUT_DIR,
                             fname_prefix="err_vs_feat"):
    """Create a multi-page PDF where each page has ≤page_size feature plots."""
    
    sns.set_style("whitegrid")
    sns.set_context("talk")     # bigger default fonts

    # --- Split into continuous / discrete for clearer pages -----------------
    cont_feats, disc_feats = [], []
    for f in features:
        if f not in df.columns:                 # skip missing
            warnings.warn(f"Feature '{f}' not in dataframe – skipped.")
            continue
        if pd.api.types.is_numeric_dtype(df[f]) and df[f].nunique() > 15:
            cont_feats.append(f)
        else:
            disc_feats.append(f)

    def _plot_batch(feat_list, page_tag):
        if not feat_list:          # nothing to draw
            return
        n_pages = math.ceil(len(feat_list)/page_size)

        pdf_path = os.path.join(out_dir, f"{fname_prefix}_{page_tag}.pdf")
        with PdfPages(pdf_path) as pdf:
            for p in range(n_pages):
                batch = feat_list[p*page_size:(p+1)*page_size]
                cols = min(3, len(batch))       # up to 3 columns
                rows = math.ceil(len(batch)/cols)

                fig, axes = plt.subplots(rows, cols,
                                         figsize=(6*cols, 4.5*rows),
                                         squeeze=False)
                axes = axes.flatten()

                for ax, feat in zip(axes, batch):
                    ser = df[feat]
                    # sample to avoid over-plotting
                    samp = df[[feat, error_col]].sample(
                        min(len(df), 4000), random_state=42)
                    
                    if feat in cont_feats:
                        sns.regplot(x=feat, y=error_col, data=samp,
                                    ax=ax, scatter_kws=dict(alpha=.12, s=12),
                                    line_kws=dict(color='red', lw=1.3),
                                    lowess=True)
                    else:
                        # small # of levels → boxplot; else → binned box
                        uniq = ser.nunique()
                        if pd.api.types.is_numeric_dtype(ser) and 10 < uniq <= 30:
                            ser_binned = pd.cut(ser, 5, precision=0,
                                                include_lowest=True)
                            sns.boxplot(x=ser_binned, y=error_col,
                                        data=df, ax=ax, showfliers=False)
                            ax.set_xticklabels(
                                [t.get_text().replace(",", ",\n") for t in ax.get_xticklabels()],
                                rotation=0)
                        else:
                            sns.boxplot(x=feat, y=error_col,
                                        data=df, ax=ax, showfliers=False)
                            ax.set_xticklabels(ax.get_xticklabels(),
                                               rotation=30, ha='right')

                    ax.axhline(0, color="black", ls="--", lw=1)
                    ax.set_ylim(-max_err, max_err)
                    title = textwrap.fill(f"{feat} ({method_label})", 25)  # wrap long names
                    ax.set_title(title, fontsize=11)
                    ax.set_xlabel("")
                    #ax.set_ylabel(error_col if ax.is_first_col() else "")
                    ax.set_ylabel(error_col if (axes.tolist().index(ax) % cols) == 0 else "")

                # turn off any unused axes
                for ax in axes[len(batch):]:
                    ax.axis("off")

                fig.tight_layout()
                pdf.savefig(fig, dpi=200)
                plt.close(fig)
        print(f"✓ Saved {pdf_path}")

    # -----------------------------------------------------------------------
    _plot_batch(cont_feats, "continuous")
    _plot_batch(disc_feats,  "categorical")

# --- Main Evaluation Logic ---
def main():
    print("--- Starting Evaluation Script ---")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved in: {OUTPUT_DIR}")

    # --- Load Data ---
    print(f"\n--- Loading Final Predictions Data ---")
    print(f"Path: {FINAL_PREDICTIONS_PATH}")
    try:
        segments_df = load_file(FINAL_PREDICTIONS_PATH)
        if not isinstance(segments_df, pd.DataFrame): raise TypeError("Loaded data is not a DataFrame.")
        print(f"Loaded data with shape: {segments_df.shape}")
        print(f"Columns found: {segments_df.columns.tolist()}")
    except FileNotFoundError: print(f"Error: Input file not found at {FINAL_PREDICTIONS_PATH}"); return
    except Exception as e: print(f"Error loading data: {e}"); traceback.print_exc(); return

    # --- Data Validation & Preparation ---
    print("\n--- Validating Required Columns & Data Types ---")
    # Use the globally defined column names
    required_cols = [TRIP_ID_COL, SEGMENT_ID_COL, GLOBAL_ERROR_COL, PERSONAL_ERROR_COL]
    optional_cols = [GT_COL, GLOBAL_PRED_COL, PERSONAL_PRED_COL]
    cols_to_check = required_cols + [col for col in optional_cols if col in segments_df.columns]

    missing_cols = [col for col in cols_to_check if col not in segments_df.columns]
    if any(col in missing_cols for col in required_cols):
        print(f"Error: DataFrame is missing REQUIRED columns for evaluation: {[col for col in required_cols if col in missing_cols]}")
        return
    elif missing_cols: print(f"Warning: Optional columns missing: {missing_cols}")

    # Ensure error columns are numeric
    print(f" - Converting error columns to numeric...")
    try:
        segments_df[GLOBAL_ERROR_COL] = pd.to_numeric(segments_df[GLOBAL_ERROR_COL], errors='coerce')
        segments_df[PERSONAL_ERROR_COL] = pd.to_numeric(segments_df[PERSONAL_ERROR_COL], errors='coerce')
        if GT_COL and GT_COL in segments_df.columns:
             segments_df[GT_COL] = pd.to_numeric(segments_df[GT_COL], errors='coerce')
    except KeyError as e: print(f"Error: Could not find error column {e}."); return
    except Exception as e: print(f"Error converting columns to numeric: {e}"); return

    print("Required columns validated and converted.")

    # --- Filter Data for Fair Comparison ---
    print("\n--- Filtering Data for Evaluation ---")
    initial_rows = len(segments_df)
    # Keep rows where BOTH error columns are non-NaN
    eval_df = segments_df.dropna(subset=[GLOBAL_ERROR_COL, PERSONAL_ERROR_COL]).copy()
    rows_dropped = initial_rows - len(eval_df)

    print(f" - Comparing on {len(eval_df)} segments where both global and final personal errors are non-NaN.")
    if rows_dropped > 0: print(f" - Excluded {rows_dropped} segments due to NaN errors.")

    if eval_df.empty: print("Error: No valid segments found for evaluation."); return

    # --- Overall Evaluation ---
    print("\n--- Overall Performance Metrics ---")
    overall_results = {}
    evaluation_summary = [] # Store results for saving

    print(f"\nGlobal Model Baseline (Error: {GLOBAL_ERROR_COL}):")
    global_metrics = calculate_error_metrics(eval_df[GLOBAL_ERROR_COL], "Global")
    overall_results['Global'] = global_metrics
    evaluation_summary.append({'Model': 'Global Baseline', **global_metrics})
    for metric, value in global_metrics.items(): print(f"  Overall {metric}: {value:.4f}")

    print(f"\nFinal Personal Model (Error: {PERSONAL_ERROR_COL}):")
    personal_metrics = calculate_error_metrics(eval_df[PERSONAL_ERROR_COL], "Personal")
    overall_results['Personal'] = personal_metrics
    evaluation_summary.append({'Model': 'Final Personal', **personal_metrics})
    for metric, value in personal_metrics.items(): print(f"  Overall {metric}: {value:.4f}")

    # Direct comparison
    print("\nComparison (Final Personal vs Global Baseline):")
    for metric in ['MAE', 'RMSE', 'MBE', 'StdDev']:
        if metric not in overall_results['Personal'] or metric not in overall_results['Global'] or \
           np.isnan(overall_results['Personal'][metric]) or np.isnan(overall_results['Global'][metric]):
            print(f"  Cannot compare {metric} due to missing/NaN values.")
            continue
        diff = overall_results['Personal'][metric] - overall_results['Global'][metric]
        if abs(overall_results['Global'][metric]) < 1e-9: change_pct = np.inf if diff != 0 else 0
        else: change_pct = (diff / overall_results['Global'][metric]) * 100
        # Comparison logic: Lower MAE/RMSE/StdDev is better, MBE closer to zero is better
        is_better = (metric != 'MBE' and diff < -1e-9) or \
                    (metric == 'MBE' and abs(overall_results['Personal'][metric]) < abs(overall_results['Global'][metric]) - 1e-9)
        is_worse = (metric != 'MBE' and diff > 1e-9) or \
                   (metric == 'MBE' and abs(overall_results['Personal'][metric]) > abs(overall_results['Global'][metric]) + 1e-9)
        comparison = "Better" if is_better else "Worse" if is_worse else "Same"
        print(f"  {metric:<7}: Personal={overall_results['Personal'][metric]:<8.4f} Global={overall_results['Global'][metric]:<8.4f} | Diff={diff:<+8.4f} ({change_pct:<+7.2f}%) -> Personal is {comparison}")

    # Save overall results to CSV
    try:
        summary_df = pd.DataFrame(evaluation_summary)
        summary_path = os.path.join(OUTPUT_DIR, 'overall_evaluation_summary.csv')
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"\n - Saved overall evaluation summary to: {summary_path}")
    except Exception as e: print(f"Warning: Could not save overall summary CSV. Error: {e}")


    # --- Plot Overall Error Distributions ---
    print("\n--- Plotting Overall Error Distributions ---")
    plot_error_distribution(eval_df[GLOBAL_ERROR_COL],
                            f'Overall Global Model Error Distribution\n(N={global_metrics["Count"]}, MAE={global_metrics["MAE"]:.2f}, RMSE={global_metrics["RMSE"]:.2f}, MBE={global_metrics["MBE"]:.2f})',
                            'overall_global_error_dist.png', OUTPUT_DIR, xlim=PLOT_MAX_ERROR)
    plot_error_distribution(eval_df[PERSONAL_ERROR_COL],
                            f'Overall Final Personal Model Error Distribution\n(N={personal_metrics["Count"]}, MAE={personal_metrics["MAE"]:.2f}, RMSE={personal_metrics["RMSE"]:.2f}, MBE={personal_metrics["MBE"]:.2f})',
                            'overall_personal_error_dist.png', OUTPUT_DIR, xlim=PLOT_MAX_ERROR)

    # --- Per-Driver Evaluation ---
    print("\n--- Per-Driver Performance Analysis ---")

    def get_driver_metrics(group):
        """Calculates metrics per driver group."""
        metrics = {}
        metrics.update({f'personal_{k.lower()}': v for k, v in calculate_error_metrics(group[PERSONAL_ERROR_COL], "Personal").items()})
        metrics.update({f'global_{k.lower()}': v for k, v in calculate_error_metrics(group[GLOBAL_ERROR_COL], "Global").items()})
        metrics['segment_count'] = len(group) # Use length of group passed to function
        if GT_COL and GT_COL in group.columns: metrics['avg_gt_soc_delta'] = group[GT_COL].mean()
        return pd.Series(metrics)

    # Group by TRIP_ID_COL and apply metric calculation
    driver_metrics_df = eval_df.groupby(TRIP_ID_COL, ).apply(get_driver_metrics)
    # Drop drivers where metrics couldn't be calculated
    driver_metrics_df.dropna(subset=['personal_mae', 'global_mae'], how='any', inplace=True) # Require both for comparison

    print(f"Calculated metrics for {len(driver_metrics_df)} drivers with valid comparison data.")

    if not driver_metrics_df.empty:
        print("\nSample of Per-Driver Metrics:")
        print(driver_metrics_df.head().round(4))

        # Analyze improvement per driver (MAE)
        driver_metrics_df['mae_improvement'] = driver_metrics_df['global_mae'] - driver_metrics_df['personal_mae']
        valid_improvements = driver_metrics_df['mae_improvement'].dropna() # Should be no NaNs due to earlier dropna

        if not valid_improvements.empty:
            improved_drivers = (valid_improvements > 1e-6).sum()
            worsened_drivers = (valid_improvements < -1e-6).sum()
            same_drivers = len(valid_improvements) - improved_drivers - worsened_drivers
            total_drivers = len(valid_improvements)

            print(f"\nDriver MAE Comparison (Final Personal vs Global):")
            print(f"  Drivers with lower MAE (Improved): {improved_drivers} ({improved_drivers/total_drivers:.1%})")
            print(f"  Drivers with higher MAE (Worsened): {worsened_drivers} ({worsened_drivers/total_drivers:.1%})")
            print(f"  Drivers with approx. same MAE:      {same_drivers} ({same_drivers/total_drivers:.1%})")
            print(f"  Average MAE Improvement per driver: {valid_improvements.mean():.4f}")
            print(f"  Median MAE Improvement per driver: {valid_improvements.median():.4f}")
            print(f"  Std Dev of MAE Improvement: {valid_improvements.std():.4f}")
            print(f"  Min Improvement (Max Worsening): {valid_improvements.min():.4f}")
            print(f"  Max Improvement: {valid_improvements.max():.4f}")

            # Save per-driver metrics to CSV
            try:
                driver_metrics_path = os.path.join(OUTPUT_DIR, 'per_driver_evaluation_metrics.csv')
                driver_metrics_df.round(4).to_csv(driver_metrics_path, index=True)
                print(f"\n - Saved per-driver evaluation metrics to: {driver_metrics_path}")
            except Exception as e: print(f"Warning: Could not save per-driver metrics CSV. Error: {e}")


            # --- Plotting Per-Driver Results ---
            print("\n--- Plotting Per-Driver Results ---")
            try:
                # MAE Improvement Distribution
                plot_error_distribution(valid_improvements,
                                        f'Distribution of MAE Improvement per Driver (N={total_drivers})\n(Global MAE - Final Personal MAE)',
                                        'per_driver_mae_improvement_dist.png', OUTPUT_DIR, xlim=None)

                # Box Plots of MAE per Driver
                plt.figure(figsize=(8, 6)) # Adjusted size
                sns.boxplot(data=driver_metrics_df[['personal_mae', 'global_mae']])
                plt.title('Distribution of MAE per Driver')
                plt.ylabel('Mean Absolute Error (MAE)')
                plt.xticks(ticks=[0,1], labels=['Final Personal', 'Global Baseline'])
                plt.grid(True, axis='y', alpha=0.3)
                plt.tight_layout()
                if SAVE_PLOTS: plt.savefig(os.path.join(OUTPUT_DIR, 'per_driver_mae_boxplot.png'), dpi=150); print(" - Saved MAE boxplot.")
                if SHOW_PLOTS: plt.show()
                plt.close()

                # Scatter plot of Personal MAE vs Global MAE
                plt.figure(figsize=(8, 8))
                # Calculate max limit dynamically, add some padding
                max_lim = max(driver_metrics_df['personal_mae'].max(), driver_metrics_df['global_mae'].max()) * 1.1
                min_lim = 0 # Assuming MAE is non-negative
                plt.scatter(driver_metrics_df['global_mae'], driver_metrics_df['personal_mae'], alpha=0.4, s=15, label='Driver')
                plt.plot([min_lim, max_lim], [min_lim, max_lim], color='red', linestyle='--', lw=1, label='y=x (No Change)')
                plt.xlabel('Global Model MAE per Driver')
                plt.ylabel('Final Personal Model MAE per Driver')
                plt.title(f'Per-Driver MAE Comparison (N={total_drivers})')
                plt.grid(True, alpha=0.3)
                plt.xlim(min_lim, max_lim); plt.ylim(min_lim, max_lim)
                plt.legend()
                plt.gca().set_aspect('equal', adjustable='box')
                plt.tight_layout()
                if SAVE_PLOTS: plt.savefig(os.path.join(OUTPUT_DIR, 'per_driver_mae_scatter.png'), dpi=150); print(" - Saved MAE scatter plot.")
                if SHOW_PLOTS: plt.show()
                plt.close()

            except Exception as e: warnings.warn(f"Could not generate per-driver plots. Error: {e}")
        else: print("Skipping MAE improvement plotting due to no valid improvement data.")
    else: print("Skipping MAE improvement analysis/plots: Required MAE columns not found.")

    # --- Enhanced Error Analysis vs. Segment Features ---
    print("\n--- Enhanced Error Analysis vs. Segment Features ---")
    plot_error_vs_features = True # Flag to enable/disable this section
    if plot_error_vs_features and PERSONAL_ERROR_COL in eval_df.columns:
        # Ensure all features in FEATURES_FOR_ERROR_ANALYSIS exist in eval_df
        # and get their data types
        features_to_analyze_info = []
        for f_name in FEATURES_FOR_ERROR_ANALYSIS: # Use the predefined list
            if f_name in eval_df.columns:
                features_to_analyze_info.append({'name': f_name, 'dtype': eval_df[f_name].dtype})
            else:
                warnings.warn(f"Feature '{f_name}' for error analysis not found in eval_df. Skipping.")

        if features_to_analyze_info:
            num_plots = len(features_to_analyze_info)
            # Determine grid size for subplots
            cols_subplot = 3 # Number of columns in subplot grid
            rows_subplot = (num_plots + cols_subplot - 1) // cols_subplot # Calculate rows needed

            fig = plt.figure(figsize=(6 * cols_subplot, 5 * rows_subplot)) # Create a single figure

            for i, feature_info in enumerate(features_to_analyze_info):
                feature = feature_info['name']
                dtype = feature_info['dtype']
                ax = fig.add_subplot(rows_subplot, cols_subplot, i + 1) # Add subplot to the figure

                # Use sample to avoid overplotting if many points, especially for scatter
                sample_size = min(len(eval_df), 2000)
                sample_df = eval_df.sample(sample_size, random_state=42) if sample_size < len(eval_df) else eval_df

                # Check if feature is numeric and has enough unique values for scatter/regplot
                # Heuristic: >15 unique values suggests continuous for regplot
                is_numeric_continuous = pd.api.types.is_numeric_dtype(dtype) and sample_df[feature].nunique() > 15

                if is_numeric_continuous:
                    try:
                        sns.regplot(x=sample_df[feature], y=sample_df[PERSONAL_ERROR_COL],
                                    ax=ax, # Plot on the current subplot
                                    scatter_kws={'alpha':0.1, 's':10, 'edgecolor':None},
                                    line_kws={'color':'red', 'lw':1.5},
                                    lowess=True) # Using LOESS smoother
                        ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.7)
                        ax.set_title(f'Error vs {feature}\n({METHOD_BEING_EVALUATED.upper()})', fontsize=10)
                        ax.set_xlabel(feature, fontsize=9)
                        ax.set_ylabel(PERSONAL_ERROR_COL, fontsize=9)
                        ax.grid(True, alpha=0.3)
                    except Exception as e_reg:
                        warnings.warn(f"Could not generate regplot for {feature}: {e_reg}. Falling back to scatter.")
                        sns.scatterplot(x=sample_df[feature], y=sample_df[PERSONAL_ERROR_COL], ax=ax, alpha=0.2, s=10, edgecolor=None)
                        ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.7)
                        ax.set_title(f'Error vs {feature} (Scatter)\n({METHOD_BEING_EVALUATED.upper()})', fontsize=10)
                        ax.set_xlabel(feature, fontsize=9); ax.set_ylabel(PERSONAL_ERROR_COL, fontsize=9); ax.grid(True, alpha=0.3)

                elif pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype) or eval_df[feature].nunique() <=15 :
                    # For discrete numeric with few unique values, or actual categoricals
                    # Bin if numeric discrete has too many unique values for good boxplot
                    if pd.api.types.is_numeric_dtype(dtype) and eval_df[feature].nunique() > 10 and eval_df[feature].nunique() <= 30:
                        try:
                            # Ensure bins cover the full range, handle potential NaN from cut
                            min_val, max_val = eval_df[feature].min(), eval_df[feature].max()
                            if pd.notna(min_val) and pd.notna(max_val) and min_val != max_val:
                                bins = np.linspace(min_val, max_val, 6) # 5 bins
                                binned_feature_series = pd.cut(eval_df[feature], bins=bins, precision=0, include_lowest=True)
                                sns.boxplot(x=binned_feature_series, y=eval_df[PERSONAL_ERROR_COL], ax=ax, showfliers=False)
                            else: # Fallback if binning is problematic
                                sns.stripplot(x=eval_df[feature], y=eval_df[PERSONAL_ERROR_COL], ax=ax, alpha=0.3, jitter=0.2)
                        except Exception as e_binbox:
                            warnings.warn(f"Could not bin/boxplot for {feature}: {e_binbox}. Using stripplot.")
                            sns.stripplot(x=eval_df[feature], y=eval_df[PERSONAL_ERROR_COL], ax=ax, alpha=0.3, jitter=0.2)
                    else: # Few unique numeric values or already categorical
                        sns.boxplot(x=eval_df[feature], y=eval_df[PERSONAL_ERROR_COL], ax=ax, showfliers=False)

                    ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.7)
                    ax.set_title(f'Error Distribution by {feature}\n({METHOD_BEING_EVALUATED.upper()})', fontsize=10)
                    ax.set_xlabel(feature, fontsize=9)
                    ax.set_ylabel(PERSONAL_ERROR_COL, fontsize=9)
                    ax.tick_params(axis='x', rotation=30, labelsize=8) # Rotate x-axis labels
                    ax.grid(True, axis='y', alpha=0.3)
                else:
                    warnings.warn(f"Feature '{feature}' has dtype {dtype} not handled well by scatter/box. Skipping detailed plot.")
                    ax.text(0.5, 0.5, f"'{feature}'\n(dtype: {dtype})\nPlot type TBD", ha='center', va='center', transform=ax.transAxes)

            fig.tight_layout(pad=2.5) # Adjust padding for the whole figure
            if SAVE_PLOTS:
                plot_path = os.path.join(OUTPUT_DIR, f'error_vs_all_features_{METHOD_BEING_EVALUATED}.png')
                try:
                    fig.savefig(plot_path, dpi=150)
                    print(f" - Saved combined error vs features plot: {plot_path}")
                except Exception as e_save:
                    print(f"Warning: Failed to save combined plot {plot_path}. Error: {e_save}")
            if SHOW_PLOTS: plt.show()
            plt.close(fig) # Close the figure object
        else:
            print(" - Skipping: None of the specified features for error analysis found in eval_df.")
    else:
        print(" - Skipping error vs features plot: Personal error column not available.")

    make_error_feature_plots(eval_df)
    print("\n--- Evaluation Complete ---")

if __name__ == '__main__':

    main()
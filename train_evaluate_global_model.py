import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
import traceback
import pickle

# --- Import necessary functions from HelperFuncs ---
try:
    from HelperFuncs import load_file # Assuming this exists and handles parquet/pickle
except ImportError:
    print("Error: Could not import from HelperFuncs. Ensure HelperFuncs.py is accessible.")
    def load_file(filepath, **kwargs): raise NotImplementedError("HelperFuncs not found")


# --- Configuration ---
# Input: Path to the features DataFrame created by the feature extraction script
# This file contains segment outcomes, start conditions, static features, AND predicted properties.
FEATURE_DATA_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segments_df\segments_features_with_routing.parquet" # Use the output WITH routing preds

# Output: Directory to save the trained global model
MODEL_OUTPUT_DIR = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\models"
MODEL_FILENAME = "global_baseline_model.joblib"

# Define the TARGET variable column name (ground truth segment outcome)
TARGET_COL = 'gt_soc_seg_delta' # This is what the global model predicts directly

PREPROCESSOR_FILENAME = "global_preprocessor.joblib" # Added filename for preprocessor

# --- Define the list of features to be used by the GLOBAL model ---
# This list uses PREDICTED segment properties + Start-of-Segment + Static features.
# It MUST match columns available in the input FEATURE_DATA_PATH.
ALL_CANDIDATE_GLOBAL_FEATURES = [
    # Start Conditions for Segment k
    'hour_seg_start',
    'dayofweek_seg_start',
    'month_seg_start',
    'lat_seg_start', # Use with caution or encode/bin
    'lon_seg_start',# Use with caution or encode/bin
    'temp_seg_start', # If available
    'battery_health_seg_start',
    'odo_seg_start',
    'soc_seg_start',
    #'start_dem_altitude', # If available
    #'start_dem_slope',    # If available
    #'start_dem_aspect',   # If available

    # Static Trip/Vehicle Features
    'car_model_trip',            # Needs encoding if not done before
    'manufacturer_trip',         # Needs encoding
    'weight_trip_kg',
    'battery_type_trip',         # Needs encoding

    # *** Predicted Segment k Properties (from Routing/DEM) ***
    'distance_seg_pred_m',
    'duration_seg_pred_s',
    'speed_seg_pred_kph',
    #'predicted_elevation_gain',  # If calculated
    #'predicted_elevation_loss',  # If calculated

    # DO NOT INCLUDE:
    # 'gt_soc_delta', 'distance_driven', 'duration_s', 'seg_speed_mean', etc. (Actual outcomes)
    # 'global_pred_soc_delta', 'global_prediction_error' (Outputs of this model/process)
    # Any 'k-1' or 'hist_' features (Belong to personal model)
]
# -----------------------------------------------------------------

# ['trip_id', 'segment_id', 'battery_health_seg_start', 'dayofweek_seg_start',
#   'hour_seg_start', 'lat_seg_start', 'lon_seg_start', 'month_seg_start',
#     'odo_seg_start', 'soc_seg_start', 'temp_seg_start', 'time_seg_start',
#       'battery_type_trip', 'car_model_trip', 'manufacturer_trip', 'odo_trip_start',
#         'soc_trip_start', 'weight_trip_kg', 'altitude_seg_agg_delta', 'altitude_seg_agg_mean',
#           'altitude_seg_agg_std', 'speed_seg_agg_max', 'speed_seg_agg_mean', 'speed_seg_agg_min',
#             'speed_seg_agg_std', 'temp_seg_agg_max', 'temp_seg_agg_mean', 'temp_seg_agg_min',
#               'temp_seg_agg_std', 'distance_seg_actual_km', 'duration_seg_actual_s', 'gt_soc_seg_delta',
#                 'odo_seg_end', 'soc_seg_end', 'time_seg_end', 'distance_seg_pred_m', 'duration_seg_pred_s',
#                   'global_pred_soc_seg_delta', 'residual_error_soc_seg_delta', 'speed_seg_pred_kph']


# Define the primary metric for selecting the best model
PRIMARY_METRIC = 'MAE' # Options: 'MAE', 'RMSE', 'R2'
RANDOM_STATE = 42
TEST_SIZE = 0.2 # Proportion of data to use for the test set

# Define candidate models to evaluate
MODELS_TO_EVALUATE = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, max_depth=10), # Added max_depth
    'CatBoost': CatBoostRegressor(
        iterations=200, # Number of trees (adjust as needed)
        learning_rate=0.1, # Adjust learning rate
        depth=6,           # Adjust tree depth
        l2_leaf_reg=3,     # L2 regularization
        loss_function='RMSE', # Common loss for regression
        eval_metric='MAE',    # Metric to watch during training (optional)
        random_seed=RANDOM_STATE,
        verbose=0          # Suppress verbose training output (set to 100 for updates every 100 iterations)
    )
    # Add other models like GradientBoostingRegressor etc.
}

# --- Helper Function for Metrics ---
def calculate_regression_metrics(y_true, y_pred):
    """Calculates MAE, RMSE, and R2 score."""
    # (Function definition remains the same)
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        warnings.warn("Invalid input for metric calculation.")
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_valid, y_pred_valid = y_true[valid_indices], y_pred[valid_indices]
    if len(y_true_valid) == 0:
         warnings.warn("No valid samples remaining after NaN removal for metric calculation.")
         return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    r2 = r2_score(y_true_valid, y_pred_valid)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# --- Modular Functions ---

def prepare_data_for_modeling(df, feature_cols, target_col):
    """Validates columns and handles NaNs for features and target."""
    print("\n--- Validating Data & Preparing Features/Target ---")
    if target_col not in df.columns: raise ValueError(f"Target column '{target_col}' not found.")
    feature_cols_present = [col for col in feature_cols if col in df.columns]
    missing_feature_cols = list(set(feature_cols) - set(feature_cols_present))
    if missing_feature_cols: warnings.warn(f"Missing requested feature columns: {missing_feature_cols}.")
    if not feature_cols_present: raise ValueError("No specified feature columns found.")
    if target_col in feature_cols_present: raise ValueError(f"Target column '{target_col}' in feature_cols.")

    required_cols = feature_cols_present + [target_col]
    print(f" - Using columns: {required_cols}")
    df_subset = df[required_cols].copy()
    initial_rows = len(df_subset)
    df_clean = df_subset.dropna(subset=[target_col])
    rows_dropped = initial_rows - len(df_clean)
    if rows_dropped > 0: print(f" - Dropped {rows_dropped} rows with NaN target.")
    if df_clean.empty: raise ValueError("No valid data remaining after handling NaNs in target.")
    X = df_clean[feature_cols_present]
    y = df_clean[target_col]
    print(f" - Prepared X shape: {X.shape}, y shape: {y.shape}")
    print("Data preparation complete.")
    return X, y

# --- MODIFIED create_preprocessor ---
def create_preprocessor(X_train):
    """
    Creates UN FITTED ColumnTransformers: general and CatBoost-specific.
    Identifies numerical and categorical columns based on X_train dtypes.
    """
    print("\n--- Defining Preprocessor Structures ---")
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f" - Identified Numerical Features ({len(numerical_features)}): {numerical_features}")
    print(f" - Identified Categorical Features ({len(categorical_features)}): {categorical_features}")

    # Define transformers
    # Add SimpleImputer(strategy='median') for numerical if needed
    # Add SimpleImputer(strategy='constant', fill_value='missing') for categorical if needed
    general_numerical_transformer = Pipeline([('scaler', StandardScaler())])
    general_categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor_general = ColumnTransformer(
        transformers=[
            ('num', general_numerical_transformer, numerical_features),
            ('cat', general_categorical_transformer, categorical_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False # Keep original names simpler
    )
    preprocessor_general.set_output(transform="pandas") # Output pandas DataFrame

    # --- Preprocessor for CatBoost ---
    catboost_numerical_transformer = Pipeline([('scaler', StandardScaler())])
    # Pass through categoricals - CatBoost handles them
    preprocessor_catboost = ColumnTransformer(
        transformers=[
            ('num', catboost_numerical_transformer, numerical_features),
            ('cat', 'passthrough', categorical_features) # Pass through
        ],
        remainder='drop',
        verbose_feature_names_out=False # Keep original names simpler
    )
    preprocessor_catboost.set_output(transform="pandas") # Output pandas DataFrame

    print("Preprocessor definitions created.")
    # Return identified lists as well, useful for CatBoost index finding later
    return preprocessor_general, preprocessor_catboost, numerical_features, categorical_features

# --- MODIFIED train_and_evaluate_models ---
def train_and_evaluate_models(X_train, y_train, X_test, y_test, models_dict,
                              fitted_preprocessor_general,
                              fitted_preprocessor_catboost,
                              catboost_categorical_features): # Pass original categorical feature names
    """Trains models using appropriate pipelines and evaluates."""
    print("\n--- Training and Evaluating Candidate Models ---")
    results = []
    trained_models = {} # Store trained PIPELINES

    # --- Preprocess Test Data Separately for CatBoost ---
    print(" - Preprocessing test data (General)...")
    try:
        X_test_processed_general = fitted_preprocessor_general.transform(X_test)
        print(f"   Test data transformed shape (General): {X_test_processed_general.shape}")
    except Exception as e:
        print(f"Error transforming test data (General): {e}. Cannot evaluate non-CatBoost models.")
        traceback.print_exc()
        X_test_processed_general = None # Flag failure

    print(" - Preprocessing test data (CatBoost)...")
    try:
        # CatBoost preprocessor just scales numericals and passes categoricals
        X_test_processed_catboost = fitted_preprocessor_catboost.transform(X_test)
        print(f"   Test data transformed shape (CatBoost): {X_test_processed_catboost.shape}")
    except Exception as e:
        print(f"Error transforming test data (CatBoost): {e}. Cannot evaluate CatBoost.")
        traceback.print_exc()
        X_test_processed_catboost = None # Flag failure
    # ----------------------------------------------------

    for model_name, model_instance in models_dict.items():
        print(f"\nTraining {model_name}...")
        is_catboost = isinstance(model_instance, CatBoostRegressor)
        current_preprocessor = fitted_preprocessor_catboost if is_catboost else fitted_preprocessor_general
        current_X_test_processed = X_test_processed_catboost if is_catboost else X_test_processed_general

        if current_X_test_processed is None and not is_catboost: # Skip non-catboost if general preproc failed
             print(f"Skipping {model_name} due to unavailable preprocessed test data.")
             results.append({'Model': model_name, 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}); trained_models[model_name] = None
             continue
        # Allow CatBoost to proceed even if its test transform failed, fit might still work

        try:
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessing', current_preprocessor),
                ('model', model_instance)
            ])

            fit_params = {}
            if is_catboost:
                # --- Find categorical feature indices AFTER CatBoost preprocessing ---
                # The CatBoost preprocessor passes categoricals through, keeping their original names.
                # We need the list of original categorical feature names passed to this function.
                # CatBoost's fit method can directly use these names if X is a DataFrame.
                # Ensure X_train IS a DataFrame before fitting the pipeline.
                if isinstance(X_train, pd.DataFrame):
                     # Check which categorical features actually exist in X_train
                     cat_features_present = [col for col in catboost_categorical_features if col in X_train.columns]
                     if cat_features_present:
                          fit_params['model__cat_features'] = cat_features_present
                          print(f"   Passing categorical feature names to CatBoost: {cat_features_present}")
                     else:
                          print("   No categorical features identified to pass explicitly to CatBoost.")
                else:
                     warnings.warn("X_train is not a DataFrame. Cannot pass categorical feature names to CatBoost. Relying on auto-detection.")
                # --------------------------------------------------------------------

            # Fit the pipeline
            pipeline.fit(X_train, y_train, **fit_params)
            print(" - Model fitting complete.")

            # Predict on the raw test data
            print(" - Predicting on test set...")
            # Need to handle case where test transform failed for this model type
            if current_X_test_processed is None and is_catboost:
                 print(f"   Skipping prediction for {model_name} as test data preprocessing failed.")
                 y_pred = np.full(len(y_test), np.nan) # Create NaN predictions
            else:
                 y_pred = pipeline.predict(X_test)


            # Calculate metrics
            print(" - Calculating performance metrics...")
            metrics = calculate_regression_metrics(y_test, y_pred) # Handles NaNs in y_pred
            results.append({'Model': model_name, **metrics})
            trained_models[model_name] = pipeline
            print(f"   Metrics for {model_name}: MAE={metrics.get('MAE', np.nan):.4f}, RMSE={metrics.get('RMSE', np.nan):.4f}, R2={metrics.get('R2', np.nan):.4f}")

        except Exception as e:
            print(f"Error training/evaluating {model_name}: {e}")
            traceback.print_exc()
            results.append({'Model': model_name, 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan})
            trained_models[model_name] = None

    results_df = pd.DataFrame(results).set_index('Model')
    return results_df, trained_models

def select_best_model(results_df, trained_models, primary_metric):
    """Selects the best model based on the primary evaluation metric."""
    # (Function definition remains the same)
    print("\n--- Selecting Best Model ---")
    if results_df.empty or results_df[primary_metric].isnull().all():
        raise ValueError("No models were successfully evaluated or primary metric is all NaN.")
    print("\nEvaluation Results Summary:")
    print(results_df.round(4)) # Print rounded results
    try:
        if primary_metric == 'R2':
            best_model_name = results_df[primary_metric].idxmax()
        elif primary_metric in ['MAE', 'RMSE']:
            best_model_name = results_df[primary_metric].idxmin()
        else: raise ValueError(f"Invalid PRIMARY_METRIC: {primary_metric}")
        best_metric_value = results_df.loc[best_model_name, primary_metric]
        print(f"\nSelected best model: '{best_model_name}' based on {primary_metric} = {best_metric_value:.4f}")
        best_model_object = trained_models[best_model_name]
        if best_model_object is None: raise ValueError(f"Selected best model '{best_model_name}' failed training.")
        return best_model_name, best_model_object
    except (KeyError, ValueError) as e: print(f"Error selecting best model: {e}"); raise
    except Exception as e: print(f"Unexpected error during model selection: {e}"); raise

def save_trained_model(model_object, output_dir, filename):
    """Saves the trained model object using joblib."""
    # (Function definition remains the same)
    print(f"\n--- Saving Selected Model ---")
    try:
        os.makedirs(output_dir, exist_ok=True)
        model_save_path = os.path.join(output_dir, filename)
        joblib.dump(model_object, model_save_path)
        print(f"Model saved successfully to: {model_save_path}")
    except Exception as e: print(f"Error saving the model: {e}"); raise

# --- Main Execution ---
def main():
    """Orchestrates the global baseline model training including CatBoost."""
    print("--- Starting Global Baseline Model Training & Evaluation (Dynamic Dtype Check w/ CatBoost) ---")
    print(f"\nInput features file: {FEATURE_DATA_PATH}")
    print(f"Model output directory: {MODEL_OUTPUT_DIR}")
    print(f"Target variable: {TARGET_COL}")
    print(f"Candidate features: {ALL_CANDIDATE_GLOBAL_FEATURES}") # Log candidate features

    try:
        # 1. Load Data
        print("\n--- Loading Data ---")
        # Load the DataFrame created by the feature extraction script
        # This DataFrame includes start features, static features, ACTUAL outcomes,
        # AND PREDICTED segment features (like predicted_distance_m).
        segments_df = load_file(FEATURE_DATA_PATH)
        if not isinstance(segments_df, pd.DataFrame): raise TypeError("Loaded data is not a DataFrame.")
        print(f"Loaded data with shape: {segments_df.shape}")
        print(f"Available columns: {segments_df.columns.tolist()}") # Log available columns

        # 2. Prepare Data for Modeling
        # This step selects ONLY the GLOBAL_FEATURES and TARGET_COL,
        # effectively excluding the actual segment outcomes for training input.
        X, y = prepare_data_for_modeling(segments_df, ALL_CANDIDATE_GLOBAL_FEATURES, TARGET_COL)

        # 3. Train/Test Split
        print("\n--- Splitting Data ---")
        # Consider GroupShuffleSplit if trips have multiple segments and leakage is a concern
        # For now, using simple random split on segments
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Test shape:  X={X_test.shape}, y={y_test.shape}")

        # 4. Define Preprocessor Structures
        # Gets BOTH definitions and the identified column lists
        preprocessor_general_def, preprocessor_catboost_def, numerical_features, categorical_features = create_preprocessor(X_train)

        # 5. Fit and Save Preprocessors
        print("\n--- Fitting and Saving Preprocessors ---")
        fitted_preprocessor_general = None
        fitted_preprocessor_catboost = None
        try:
            print(" - Fitting General Preprocessor...")
            preprocessor_general_def.fit(X_train)
            print("   General Preprocessor fitted.")
            save_trained_model(preprocessor_general_def, MODEL_OUTPUT_DIR, f"general_{PREPROCESSOR_FILENAME}")
            fitted_preprocessor_general = preprocessor_general_def
        except Exception as e: print(f"Error fitting/saving General Preprocessor: {e}"); traceback.print_exc()

        try:
            print(" - Fitting CatBoost Preprocessor...")
            preprocessor_catboost_def.fit(X_train)
            print("   CatBoost Preprocessor fitted.")
            save_trained_model(preprocessor_catboost_def, MODEL_OUTPUT_DIR, f"catboost_{PREPROCESSOR_FILENAME}")
            fitted_preprocessor_catboost = preprocessor_catboost_def
        except Exception as e: print(f"Error fitting/saving CatBoost Preprocessor: {e}"); traceback.print_exc()

        if fitted_preprocessor_general is None or fitted_preprocessor_catboost is None:
             raise ValueError("One or both preprocessors failed to fit. Cannot proceed.")

        # 6. Train and Evaluate Models
        # Pass the original categorical feature names list to help CatBoost pipeline
        results_df, trained_pipelines = train_and_evaluate_models(
            X_train, y_train, X_test, y_test, MODELS_TO_EVALUATE,
            fitted_preprocessor_general, fitted_preprocessor_catboost,
            categorical_features # Pass the list identified by create_preprocessor
        )

        # 7. Select Best Model
        best_model_name, best_pipeline_object = select_best_model(
            results_df, trained_pipelines, PRIMARY_METRIC
        )

        # 8. Save Best Model (Saves the entire selected PIPELINE)
        save_trained_model(best_pipeline_object, MODEL_OUTPUT_DIR, MODEL_FILENAME)

        print("\n--- Global Baseline Model Training & Evaluation Complete ---")

    except (FileNotFoundError, ValueError, TypeError, KeyError, Exception) as e:
        print(f"\n--- Pipeline Failed ---")
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
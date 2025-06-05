import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge # Example models
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor # Import CatBoost
from sklearn.metrics import mean_absolute_error, mean_squared_error # For evaluation
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # Keep for potentially saving the personal models if needed later
import warnings
import time
import traceback
from collections import defaultdict, OrderedDict # To ensure consistent feature order
from sklearn.feature_selection import VarianceThreshold # Added import


# --- Import necessary functions from HelperFuncs ---
try:
    from HelperFuncs import (
        load_file, save_file, split_trips_train_test,
        remove_highly_collinear_numerical_features, display_and_save_feature_importances
    )
except ImportError:
    print("Error: Could not import from HelperFuncs. Ensure HelperFuncs.py is accessible.")
    def load_file(filepath, **kwargs): raise NotImplementedError("HelperFuncs not found")
    def save_file(data, path, file_name, format='pickle', index=False): raise NotImplementedError("HelperFuncs not found")

# features from feature extraction script
# ['trip_id', 'segment_id', 'battery_health_seg_start', 'dayofweek_seg_start', 'hour_seg_start',
#  'lat_seg_start', 'lon_seg_start', 'month_seg_start', 'odo_seg_start', 'soc_seg_start', 
# 'temp_seg_start', 'time_seg_start', 'battery_type_trip', 'car_model_trip', 'manufacturer_trip', 
# 'odo_trip_start', 'soc_trip_start', 'weight_trip_kg', 'accel_seg_agg_max', 'accel_seg_agg_mean', 
# 'accel_seg_agg_std', 'altitude_seg_agg_delta', 'altitude_seg_agg_mean', 'altitude_seg_agg_std', 
# 'decel_seg_agg_max', 'decel_seg_agg_mean', 'decel_seg_agg_std', 'jerk_abs_seg_agg_max', 
# 'jerk_abs_seg_agg_mean', 'jerk_abs_seg_agg_std', 'speed_seg_agg_max_mps', 'speed_seg_agg_mean_mps', 
# 'speed_seg_agg_min_mps', 'speed_seg_agg_std_mps', 'stop_duration_seg_agg_s', 'stops_seg_count', 
# 'temp_seg_agg_max', 'temp_seg_agg_mean', 'temp_seg_agg_min', 'temp_seg_agg_std', 'distance_seg_actual_km', 
# 'duration_seg_actual_s', 'gt_soc_seg_delta', 'odo_seg_end', 'soc_seg_end', 'time_seg_end', 
# 'avg_speed_limit_kph_seg_agg', 'direct_route_distance_m', 'global_pred_soc_seg_delta', 'match_attempts', 
# 'match_confidence', 'match_status', 'matched_route_distance_m', 'matched_route_duration_s', 
# 'percent_time_over_limit_seg_agg', 'residual_error_soc_seg_delta', 'route_deviation_pct', 
# 'accel_high_event_count', 'decel_high_event_count', 'distance_seg_pred_m', 'duration_seg_pred_s', 
# 'final_match_radius_m', 'speed_seg_pred_kph']

# --- Configuration (Define Feature Names Consistently) ---
# Use the output from the feature extraction script that includes routing predictions AND new names
FEATURE_FILE_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segments_df\segments_features_v4_behavior.parquet" # Use output with driving behavior    
OUTPUT_FILE_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segments_df\segments_final_predictions_v3" # New output name
# --- NEW: Output path for ALL features ---
ALL_FEATURES_OUTPUT_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segments_df\segments_all_engineered_features_loso_global_residual.parquet"
# --- NEW: Output path for USED features list by the final model ---
USED_MODEL_FEATURES_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segments_df\used_model_features_loso_global_residual.txt"

OUTPUT_FORMAT = 'parquet'
MIN_TRAIN_SAMPLES = 3 # Minimum number of OTHER segments for LOSO, or history for WF
PRIMARY_METRIC = 'MAE' # Metric to choose best personal model
RANDOM_STATE = 42 # For models that use it
TRIP_SPLIT_TEST_SIZE_FOR_LGR = 0.25 # Example: 25% of trips for testing the global residual model
# *** CHOOSE EVALUATION METHOD ***
METHOD = 'loso_global_residual' # Options: 'walk_forward', 'loso', 'loso_global_residual'
NZV_THRESHOLD = 0.001 # Near zero variance THRESHOLD

# These MUST match the columns created by the updated feature extraction script
TRIP_ID_COL = 'trip_id'
SEGMENT_ID_COL = 'segment_id'
GT_TARGET_COL = 'gt_soc_seg_delta' # Ground truth outcome of the segment
GLOBAL_PRED_COL = 'global_pred_soc_seg_delta' # Global model's prediction for the segment
RESIDUAL_TARGET_COL = 'residual_error_soc_seg_delta' # Target for personal model (GT - GlobalPred)

# Features describing the segment's ACTUAL outcome (used for history calculation)
# Use NEW descriptive names from feature extraction output
SEGMENT_OUTCOME_FEATURES = [
    'gt_soc_seg_delta', 'distance_seg_actual_km', 'duration_seg_actual_s',
    # Altitude Aggregates
    'altitude_seg_agg_mean', 'altitude_seg_agg_delta', 'altitude_seg_agg_std',
    # Temperature Aggregates
    'temp_seg_agg_mean', 'temp_seg_agg_std', 'temp_seg_agg_max', 'temp_seg_agg_min',
    # Kinematics Aggregates (Names from aggregate_kinematics helper)
    'speed_seg_agg_mean_mps', 'speed_seg_agg_std_mps', 'speed_seg_agg_max_mps', 'speed_seg_agg_min_mps',
    'accel_seg_agg_mean', 'accel_seg_agg_std', 'accel_seg_agg_max', 'accel_high_event_count',
    'decel_seg_agg_mean', 'decel_seg_agg_std', 'decel_seg_agg_max', 'decel_high_event_count',
    'jerk_abs_seg_agg_mean', 'jerk_abs_seg_agg_std', 'jerk_abs_seg_agg_max',
    # Stop Aggregates (Names from calculate_stop_features helper)
    'stops_seg_count', 'stop_duration_seg_agg_s',
    # Map Matching / Routing Derived Actuals
    'matched_route_distance_m', 'matched_route_duration_s', 
    'direct_route_distance_m',
    'route_deviation_pct', 'percent_time_over_limit_seg_agg', 'avg_speed_limit_kph_seg_agg',
    'match_confidence',
    # Include the residual error itself as an outcome of the global model for that segment
    RESIDUAL_TARGET_COL # 'residual_error_soc_seg_delta'
    # Add other calculated ACTUAL segment outcome/aggregate metrics here if created
]

# Features PREDICTED for segment k by routing engine (subset of columns in input df)
# Use NEW descriptive names
PREDICTED_SEGMENT_FEATURES = [
    'distance_seg_pred_m',
    'duration_seg_pred_s',
    'speed_seg_pred_kph'
    # 'elevation_gain_seg_pred', # If available
    # 'elevation_loss_seg_pred', # If available
]

# Features known at the START of segment k (subset of columns in input df)
# Use NEW descriptive names
START_OF_SEGMENT_FEATURES = [
    'soc_seg_start', 'odo_seg_start', 'hour_seg_start', 'dayofweek_seg_start', 'month_seg_start',
    'lat_seg_start', 'lon_seg_start', 'temp_seg_start', 'battery_health_seg_start',
    #'dem_altitude_seg_start', 'dem_slope_seg_start', 'dem_aspect_seg_start', # If available
    # Add relevant static features (using _trip suffix)
    'weight_trip_kg','car_model_trip', 'manufacturer_trip', 'battery_type_trip',
    'soc_trip_start', 'odo_trip_start'
]

# --- USER-DEFINED COMPREHENSIVE FEATURE LIST ---
ALL_POSSIBLE_LOSO_FEATURES = [
    # --- Start of Segment k & Static Features ---
    'k_soc_seg_start', 'k_odo_seg_start', 'k_hour_seg_start', 'k_dayofweek_seg_start', 'k_month_seg_start',
    'k_lat_seg_start', 'k_lon_seg_start', 'k_temp_seg_start', 'k_battery_health_seg_start',
    # 'k_dem_altitude_seg_start', 'k_dem_slope_seg_start', 'k_dem_aspect_seg_start',
    'k_weight_trip_kg', 'k_car_model_trip', 'k_manufacturer_trip', 'k_battery_type_trip',
    'k_soc_trip_start', 'k_odo_trip_start',

    # --- Predicted Segment k Features ---
    'k_distance_seg_pred_m', 'k_duration_seg_pred_s', 'k_speed_seg_pred_kph',
    # 'k_elevation_gain_seg_pred', 'k_elevation_loss_seg_pred',

    # --- Global Prediction for Segment k ---
    'k_global_pred_soc_seg_delta',

    # --- Aggregates over OTHER Segments (j != k) ---
    'others_segment_count',
    'others_total_distance_seg_actual_km', 'others_total_duration_seg_actual_s',
    'others_total_stops_seg_count', 'others_total_stop_duration_seg_agg_s',
    'others_total_accel_high_event_count', 'others_total_decel_high_event_count',

    'others_avg_gt_soc_seg_delta', 'others_avg_distance_seg_actual_km', 'others_avg_duration_seg_actual_s',
    'others_avg_altitude_seg_agg_mean', 'others_avg_altitude_seg_agg_delta', 'others_avg_altitude_seg_agg_std',
    'others_avg_temp_seg_agg_mean', 'others_avg_temp_seg_agg_std', 'others_avg_temp_seg_agg_max', 'others_avg_temp_seg_agg_min',
    'others_avg_speed_seg_agg_mean_mps', 'others_avg_speed_seg_agg_std_mps', 'others_avg_speed_seg_agg_max_mps', 'others_avg_speed_seg_agg_min_mps',
    'others_avg_accel_seg_agg_mean', 'others_avg_accel_seg_agg_std', 'others_avg_accel_seg_agg_max', 'others_avg_accel_high_event_count',
    'others_avg_decel_seg_agg_mean', 'others_avg_decel_seg_agg_std', 'others_avg_decel_seg_agg_max', 'others_avg_decel_high_event_count',
    'others_avg_jerk_abs_seg_agg_mean', 'others_avg_jerk_abs_seg_agg_std', 'others_avg_jerk_abs_seg_agg_max',
    'others_avg_stops_seg_count', 'others_avg_stop_duration_seg_agg_s',
    'others_avg_direct_route_distance_m', 'others_avg_route_deviation_pct',
    'others_avg_percent_time_over_limit_seg_agg', 'others_avg_avg_speed_limit_kph_seg_agg',
    'others_avg_match_confidence',
    'others_avg_residual_error_soc_seg_delta',
    'others_abs_avg_residual_error_soc_seg_delta',

    'others_std_gt_soc_seg_delta', 'others_std_distance_seg_actual_km', 'others_std_duration_seg_actual_s',
    'others_std_altitude_seg_agg_mean', 'others_std_altitude_seg_agg_delta', 'others_std_altitude_seg_agg_std',
    'others_std_temp_seg_agg_mean', 'others_std_temp_seg_agg_std', 'others_std_temp_seg_agg_max', 'others_std_temp_seg_agg_min',
    'others_std_speed_seg_agg_mean_mps', 'others_std_speed_seg_agg_std_mps', 'others_std_speed_seg_agg_max_mps', 'others_std_speed_seg_agg_min_mps',
    'others_std_accel_seg_agg_mean', 'others_std_accel_seg_agg_std', 'others_std_accel_seg_agg_max', 'others_std_accel_high_event_count',
    'others_std_decel_seg_agg_mean', 'others_std_decel_seg_agg_std', 'others_std_decel_seg_agg_max', 'others_std_decel_high_event_count',
    'others_std_jerk_abs_seg_agg_mean', 'others_std_jerk_abs_seg_agg_std', 'others_std_jerk_abs_seg_agg_max',
    'others_std_stops_seg_count', 'others_std_stop_duration_seg_agg_s',
    'others_std_direct_route_distance_m', 'others_std_route_deviation_pct',
    'others_std_percent_time_over_limit_seg_agg', 'others_std_avg_speed_limit_kph_seg_agg',
    'others_std_match_confidence',
    'others_std_residual_error_soc_seg_delta',

    'others_rate_stops_per_km', 'others_rate_stops_per_minute',
    'others_rate_accel_high_events_per_km', 'others_rate_accel_high_events_per_minute',
    'others_rate_decel_high_events_per_km', 'others_rate_decel_high_events_per_minute',
    'others_ratio_stop_duration_to_total_duration'
]

# --- Define the FINAL feature vector order for the PERSONAL model ---
# This MUST be consistent between training and prediction. Use NEW names with prefixes.
FINAL_FEATURE_ORDER_WALK_FORWARD = [ # Needs verification!!
    # --- Start of Segment k Features ---
    'k_soc_seg_start',
    'k_odo_seg_start',
    'k_hour_seg_start',
    'k_dayofweek_seg_start',
    'k_month_seg_start',
    'k_temp_seg_start',
    'k_battery_health_seg_start',
    #'k_dem_altitude_seg_start',
    #'k_dem_slope_seg_start',
    # --- Static Features (relevant at start of k) ---
    'k_weight_trip_kg',
    #'k_soc_trip_start',
    # Add encoded categoricals if needed: 'k_car_model_trip_A', ... 'car_model_trip', 'manufacturer_trip', 'battery_type_trip'

    # --- Predicted Segment k Features ---
    'k_distance_seg_pred_m',
    'k_duration_seg_pred_s',
    'k_speed_seg_pred_kph',
    # 'k_elevation_gain_seg_pred',
    # 'k_elevation_loss_seg_pred',

    # --- Global Prediction for Segment k ---
    'k_global_pred_soc_seg_delta', # The baseline prediction itself as a feature

    # --- Lagged (k-1) Features (Actual Outcomes) ---
    'k-1_gt_soc_seg_delta',
    'k-1_distance_seg_actual_km',
    'k-1_duration_seg_actual_s',
    'k-1_speed_seg_agg_mean',
    'k-1_speed_seg_agg_std',
    'k-1_speed_seg_agg_min',
    'k-1_speed_seg_agg_max',
    'k-1_altitude_seg_agg_mean',
    'k-1_altitude_seg_agg_delta',
    'k-1_altitude_seg_agg_std',
    'k-1_temp_seg_agg_mean',
    'k-1_temp_seg_agg_std',
    'k-1_temp_seg_agg_max',
    'k-1_temp_seg_agg_min',

    'k-1_accel_seg_agg_mean', 'k-1_accel_high_event_count', # Example kinematics
    'k-1_decel_seg_agg_mean', 'k-1_decel_high_event_count', # Example kinematics
    'k-1_jerk_abs_seg_agg_mean', # Example kinematics
    'k-1_stops_seg_count', 'k-1_stop_duration_seg_agg_s', # Example stops

    # --- Historical (1 to k-1) Features (Aggregated Actual Outcomes) ---
    'hist_segment_count',
    'hist_avg_soc_delta_per_km', # Avg actual delta per actual km
    'hist_avg_speed_actual',     # Avg of actual segment mean speeds
    'hist_avg_temp_actual',      # Avg of actual segment mean temps
    'hist_cumulative_distance_actual',
    'hist_cumulative_duration_actual'
    'hist_avg_accel_mean', 'hist_avg_decel_mean', 'hist_avg_jerk_mean', # Example kinematics history
    'hist_avg_stops_per_km', 'hist_avg_stop_duration_per_stop', # Example stop history
    'hist_std_speed_actual' # Example: Std Dev of past segment mean speeds
    # Add more historical features here 
]

FINAL_FEATURE_ORDER_LOSO = [
    # Start of Segment k & Static
    'k_soc_seg_start', 'k_odo_seg_start', 'k_hour_seg_start', 'k_dayofweek_seg_start', 'k_month_seg_start', 
    'k_temp_seg_start', 'k_battery_health_seg_start',
    'k_weight_trip_kg', 'k_car_model_trip', 'k_manufacturer_trip', 'k_battery_type_trip', # Categorical

    #'k_soc_trip_start',
    #'k_dem_altitude_seg_start',
    #'k_dem_slope_seg_start',
   
    # Predicted Segment k
    'k_distance_seg_pred_m',
    'k_duration_seg_pred_s',
    'k_speed_seg_pred_kph',
    # 'k_elevation_gain_seg_pred',
    # 'k_elevation_loss_seg_pred',

    # Global Prediction for Segment k
    'k_global_pred_soc_seg_delta', # The baseline prediction itself as a feature
    
    # Aggregates over OTHER Segments (j != k)
    #'others_segment_count',
    #'others_total_distance_actual_km', 'others_total_duration_actual_s',
    # Averages
    'others_avg_gt_soc_seg_delta', #'others_avg_distance_seg_actual_km', 'others_avg_duration_seg_actual_s',
    'others_avg_speed_seg_agg_mean_mps', 'others_avg_speed_seg_agg_std_mps', 'others_avg_speed_seg_agg_max_mps',
    'others_avg_speed_seg_agg_min_mps',
    'others_avg_altitude_seg_agg_mean', 'others_avg_altitude_seg_agg_delta', #'others_avg_altitude_seg_agg_std',
    #'others_avg_temp_seg_agg_mean',
    #'others_avg_temp_seg_agg_std', 'others_avg_temp_seg_agg_max', 'others_avg_temp_seg_agg_min',
    'others_avg_accel_seg_agg_mean', 'others_avg_accel_seg_agg_std', 'others_avg_accel_seg_agg_max',
    'others_avg_decel_seg_agg_mean', 'others_avg_decel_seg_agg_std', 'others_avg_decel_seg_agg_max',
    'others_avg_jerk_abs_seg_agg_mean', 'others_avg_jerk_abs_seg_agg_std', 'others_avg_jerk_abs_seg_agg_max',
    'others_avg_stop_duration_seg_agg_s',
    'others_avg_direct_route_distance_m',
    'others_avg_route_deviation_pct', 'others_avg_percent_time_over_limit_seg_agg', 'others_avg_avg_speed_limit_kph_seg_agg',
    #'others_avg_match_confidence',
    'others_avg_residual_error_soc_seg_delta', 'others_abs_avg_residual_error_soc_seg_delta',
    # Standard Deviations
    'others_std_gt_soc_seg_delta', 'others_std_distance_seg_actual_km', #'others_std_duration_seg_actual_s',
    'others_std_speed_seg_agg_mean_mps', # Std of segment means
    #'others_std_altitude_seg_agg_mean', 'others_std_altitude_seg_agg_delta',
    #'others_std_temp_seg_agg_mean',
    'others_std_accel_seg_agg_mean', 'others_std_decel_seg_agg_mean', 'others_std_jerk_abs_seg_agg_mean',
    'others_std_stop_duration_seg_agg_s',
    'others_std_route_deviation_pct', 'others_std_percent_time_over_limit_seg_agg',
    #'others_std_match_confidence', 
    'others_std_avg_speed_limit_kph_seg_agg',
    'others_std_residual_error_soc_seg_delta',
    # --- Rates for "other" segments ---
    'others_rate_stops_per_km', # Total stops in others / Total distance in others
    'others_rate_stops_per_minute', # Total stops in others / Total duration in others
    'others_rate_accel_high_events_per_km',
    'others_rate_accel_high_events_per_minute',
    'others_rate_decel_high_events_per_km',
    'others_rate_decel_high_events_per_minute',
    'others_ratio_stop_duration_to_total_duration' # Total stop duration / Total segment duration
]
# -------------------------------------------------

# --- Feature Engineering Helpers ---
def build_walk_forward_feature_vector(current_segment_row, history_df): # Needs editing!
    """
    Constructs the feature vector for Walk-Forward predicting the current segment (k) based on
    start conditions (k), predicted properties (k), global prediction (k),
    and history of previous segments' ACTUAL outcomes (1 to k-1).
    Uses NEW descriptive column names.

    Args:
        current_segment_row (pd.Series): Row for segment k from segments_df.
        history_df (pd.DataFrame): DataFrame for segments 1 to k-1 from segments_df.

    Returns:
        pd.DataFrame: Single-row DataFrame with features, or None if invalid.
    """
    features = OrderedDict() # Use OrderedDict to help maintain order initially
    segment_id_debug = current_segment_row.get(SEGMENT_ID_COL, 'N/A') # For logging
    feature_order = FINAL_FEATURE_ORDER_WALK_FORWARD # Use WF order

    # --- 1. Start-of-Segment k & Static Features ---
    # Use 'k_' prefix and NEW descriptive names from current_segment_row
    features['k_soc_seg_start'] = current_segment_row.get('soc_seg_start', np.nan)
    features['k_odo_seg_start'] = current_segment_row.get('odo_seg_start', np.nan)
    features['k_hour_seg_start'] = current_segment_row.get('hour_seg_start', np.nan)
    features['k_dayofweek_seg_start'] = current_segment_row.get('dayofweek_seg_start', np.nan)
    features['k_month_seg_start'] = current_segment_row.get('month_seg_start', np.nan)
    features['k_temp_seg_start'] = current_segment_row.get('temp_seg_start', np.nan)
    features['k_battery_health_seg_start'] = current_segment_row.get('battery_health_seg_start', np.nan)
    #features['k_dem_altitude_seg_start'] = current_segment_row.get('dem_altitude_seg_start', np.nan)
    #features['k_dem_slope_seg_start'] = current_segment_row.get('dem_slope_seg_start', np.nan)
    features['k_weight_trip_kg'] = current_segment_row.get('weight_trip_kg', np.nan) # Static
    # Add other start/static features using .get() with NEW names
    # Add encoded categoricals if needed: 'k_car_model_trip_A', ... 'car_model_trip', 'manufacturer_trip', 'battery_type_trip'

    # --- 2. Predicted Segment k Features ---
    # Use 'k_' prefix and NEW descriptive names from current_segment_row
    features['k_distance_seg_pred_m'] = current_segment_row.get('distance_seg_pred_m', np.nan)
    features['k_duration_seg_pred_s'] = current_segment_row.get('duration_seg_pred_s', np.nan)
    features['k_speed_seg_pred_kph'] = current_segment_row.get('speed_seg_pred_kph', np.nan)
    #features['k_elevation_gain_seg_pred'] = current_segment_row.get('elevation_gain_seg_pred', np.nan)
    #features['k_elevation_loss_seg_pred'] = current_segment_row.get('elevation_loss_seg_pred', np.nan)
    # Add other predicted features using .get() with NEW names

    # --- 3. Global Prediction for Segment k ---
    # Use 'k_' prefix and NEW descriptive name from current_segment_row
    features['k_global_pred_soc_seg_delta'] = current_segment_row.get('global_pred_soc_seg_delta', np.nan)

    # # --- 4. Lagged Features (from Segment k-1 ACTUAL Outcomes) ---
    # # Use 'k-1_' prefix and NEW descriptive outcome names from history_df
    # # *** Use UPDATED SEGMENT_OUTCOME_FEATURES list ***
    # if not history_df.empty:
    #     last_segment_row = history_df.iloc[-1]
    #     # Use NEW outcome names from SEGMENT_OUTCOME_FEATURES list (updated below)
    #     features['k-1_gt_soc_seg_delta'] = last_segment_row.get('gt_soc_seg_delta', np.nan)
    #     features['k-1_distance_seg_actual_km'] = last_segment_row.get('distance_seg_actual_km', np.nan)
    #     features['k-1_duration_seg_actual_s'] = last_segment_row.get('duration_seg_actual_s', np.nan)
    #     features['k-1_speed_seg_agg_mean'] = last_segment_row.get('speed_seg_agg_mean', np.nan)
    #     features['k-1_speed_seg_agg_std'] = last_segment_row.get('speed_seg_agg_std', np.nan)
    #     features['k-1_speed_seg_agg_min'] = last_segment_row.get('speed_seg_agg_min', np.nan)
    #     features['k-1_speed_seg_agg_max'] = last_segment_row.get('speed_seg_agg_max', np.nan)
    #     features['k-1_altitude_seg_agg_delta'] = last_segment_row.get('altitude_seg_agg_delta', np.nan)
    #     features['k-1_altitude_seg_agg_mean'] = last_segment_row.get('altitude_seg_agg_mean', np.nan)
    #     features['k-1_altitude_seg_agg_std'] = last_segment_row.get('altitude_seg_agg_std', np.nan)
    #     features['k-1_temp_seg_agg_mean'] = last_segment_row.get('temp_seg_agg_mean', np.nan)
    #     features['k-1_temp_seg_agg_std'] = last_segment_row.get('temp_seg_agg_std', np.nan)
    #     features['k-1_temp_seg_agg_min'] = last_segment_row.get('temp_seg_agg_min', np.nan)
    #     features['k-1_temp_seg_agg_max'] = last_segment_row.get('temp_seg_agg_max', np.nan)
    #     # Add other k-1 features using .get() with NEW outcome names
    # else: # Handle k=2 case
    #     # Use NEW outcome names
    #     features['k-1_gt_soc_seg_delta'] = np.nan
    #     features['k-1_distance_seg_actual_km'] = np.nan
    #     features['k-1_duration_seg_actual_s'] = np.nan
    #     features['k-1_speed_seg_agg_mean'] = np.nan
    #     features['k-1_speed_seg_agg_std'] = np.nan
    #     features['k-1_speed_seg_agg_min'] = np.nan
    #     features['k-1_speed_seg_agg_max'] = np.nan
    #     features['k-1_altitude_seg_agg_delta'] = np.nan
    #     features['k-1_altitude_seg_agg_mean'] = np.nan
    #     features['k-1_altitude_seg_agg_std'] = np.nan
    #     features['k-1_temp_seg_agg_mean'] = np.nan
    #     features['k-1_temp_seg_agg_std'] = np.nan
    #     features['k-1_temp_seg_agg_min'] = np.nan
    #     features['k-1_temp_seg_agg_max'] = np.nan
    #     # Assign defaults (NaN) for other k-1 features

   # 4. Lagged (k-1) Features
    # *** Use UPDATED SEGMENT_OUTCOME_FEATURES list ***
    if not history_df.empty:
        last_segment_row = history_df.iloc[-1]
        for col in SEGMENT_OUTCOME_FEATURES:
             feature_name = f'k-1_{col}'
             # Also add previous residual error if needed
             if feature_name in feature_order: features[feature_name] = last_segment_row.get(col, np.nan)
        if 'k-1_soc_delta_residual_error' in feature_order: # Add previous error explicitly
             features['k-1_soc_delta_residual_error'] = last_segment_row.get(RESIDUAL_TARGET_COL, np.nan)
    else:
        for col in SEGMENT_OUTCOME_FEATURES:
             feature_name = f'k-1_{col}'
             if feature_name in feature_order: features[feature_name] = np.nan
        if 'k-1_soc_delta_residual_error' in feature_order: features['k-1_soc_delta_residual_error'] = np.nan


    # 5. Historical (1 to k-1) Features
    # *** Calculate NEW driving/routing behavior history features ***
    features['hist_segment_count'] = len(history_df)
    if not history_df.empty and len(history_df) > 0:
        # Basic history
        hist_dist = history_df['distance_seg_actual_km'].sum()
        hist_dur_s = history_df['duration_seg_actual_s'].sum()
        hist_soc_delta = history_df['gt_soc_seg_delta'].sum()
        hist_speed_mean_avg = history_df['speed_seg_agg_mean'].mean()
        hist_speed_mean_std = history_df['speed_seg_agg_mean'].std()
        hist_temp_mean_avg = history_df['temp_seg_agg_mean'].mean()
        # New driving behavior history
        hist_accel_mean_avg = history_df['accel_seg_agg_mean'].mean()
        hist_decel_mean_avg = history_df['decel_seg_agg_mean'].mean()
        hist_jerk_mean_avg = history_df['jerk_abs_seg_agg_mean'].mean()
        hist_stops_total = history_df['stops_seg_count'].sum()
        hist_stop_dur_total = history_df['stop_duration_seg_agg_s'].sum()
        hist_route_dev_avg = history_df['route_deviation_pct'].mean()
        hist_over_limit_avg = history_df['percent_time_over_limit_seg_agg'].mean()
        hist_match_conf_avg = history_df['match_confidence'].mean()
        # New Error History
        hist_global_error_mean = history_df[RESIDUAL_TARGET_COL].mean()
        hist_global_error_abs_mean = history_df[RESIDUAL_TARGET_COL].abs().mean()
        hist_global_error_std = history_df[RESIDUAL_TARGET_COL].std()

        # Assign features (check if needed by feature_order)
        if 'hist_cumulative_distance_actual' in feature_order: features['hist_cumulative_distance_actual'] = hist_dist
        if 'hist_cumulative_duration_actual' in feature_order: features['hist_cumulative_duration_actual'] = hist_dur_s
        if 'hist_avg_soc_delta_per_km' in feature_order: features['hist_avg_soc_delta_per_km'] = (hist_soc_delta / hist_dist) if hist_dist > 1e-3 else 0.0
        if 'hist_avg_speed_actual' in feature_order: features['hist_avg_speed_actual'] = hist_speed_mean_avg
        if 'hist_std_speed_actual' in feature_order: features['hist_std_speed_actual'] = hist_speed_mean_std
        if 'hist_avg_temp_actual' in feature_order: features['hist_avg_temp_actual'] = hist_temp_mean_avg
        if 'hist_avg_accel_mean' in feature_order: features['hist_avg_accel_mean'] = hist_accel_mean_avg
        if 'hist_avg_decel_mean' in feature_order: features['hist_avg_decel_mean'] = hist_decel_mean_avg
        if 'hist_avg_jerk_mean' in feature_order: features['hist_avg_jerk_mean'] = hist_jerk_mean_avg
        if 'hist_avg_stops_per_km' in feature_order: features['hist_avg_stops_per_km'] = (hist_stops_total / hist_dist) if hist_dist > 1e-3 else 0.0
        if 'hist_avg_stop_duration_per_stop' in feature_order: features['hist_avg_stop_duration_per_stop'] = (hist_stop_dur_total / hist_stops_total) if hist_stops_total > 0 else 0.0
        if 'hist_avg_route_deviation_pct' in feature_order: features['hist_avg_route_deviation_pct'] = hist_route_dev_avg
        if 'hist_avg_percent_time_over_limit' in feature_order: features['hist_avg_percent_time_over_limit'] = hist_over_limit_avg
        if 'hist_avg_match_confidence' in feature_order: features['hist_avg_match_confidence'] = hist_match_conf_avg
        if 'hist_avg_residual_error' in feature_order: features['hist_avg_residual_error'] = hist_global_error_mean
        if 'hist_abs_avg_residual_error' in feature_order: features['hist_abs_avg_residual_error'] = hist_global_error_abs_mean
        if 'hist_std_residual_error' in feature_order: features['hist_std_residual_error'] = hist_global_error_std

    else: # Handle k=2 case - set all hist features to default (NaN or 0)
        for key in feature_order:
            if key.startswith('hist_'):
                if 'count' in key or 'cumulative' in key or 'per_km' in key: features[key] = 0.0
                else: features[key] = np.nan

    # Ensure consistent order and create DataFrame
    final_features_dict = {key: features.get(key, np.nan) for key in feature_order}
    feature_vector_df = pd.DataFrame([final_features_dict], columns=feature_order)

    if feature_vector_df.isnull().values.any(): return None
    return feature_vector_df

def build_loso_feature_vector(current_segment_row, other_segments_df):
    """
    Constructs a COMPREHENSIVE feature vector for predicting segment k using LOSO method.
    Features include start/pred/static/global_pred for k, and aggregates over other_segments_df.
    The output DataFrame will have columns ordered by ALL_POSSIBLE_LOSO_FEATURES.
    """
    features = OrderedDict()
    segment_id_debug = current_segment_row.get(SEGMENT_ID_COL, 'N/A')
    #feature_order = FINAL_FEATURE_ORDER_LOSO # Use LOSO order
    feature_order_to_build = ALL_POSSIBLE_LOSO_FEATURES # MODIFIED

    # --- 1. Start-of-Segment k & Static Features --- (Same as walk-forward)
    # # Use 'k_' prefix and NEW descriptive names from current_segment_row
    # features['k_soc_seg_start'] = current_segment_row.get('soc_seg_start', np.nan)
    # features['k_odo_seg_start'] = current_segment_row.get('odo_seg_start', np.nan)
    # features['k_hour_seg_start'] = current_segment_row.get('hour_seg_start', np.nan)
    # features['k_dayofweek_seg_start'] = current_segment_row.get('dayofweek_seg_start', np.nan)
    # features['k_month_seg_start'] = current_segment_row.get('month_seg_start', np.nan)
    # features['k_temp_seg_start'] = current_segment_row.get('temp_seg_start', np.nan)
    # features['k_battery_health_seg_start'] = current_segment_row.get('battery_health_seg_start', np.nan)
    # #features['k_dem_altitude_seg_start'] = current_segment_row.get('dem_altitude_seg_start', np.nan)
    # #features['k_dem_slope_seg_start'] = current_segment_row.get('dem_slope_seg_start', np.nan)
    # features['k_weight_trip_kg'] = current_segment_row.get('weight_trip_kg', np.nan) # Static
    # # Add other start/static features using .get() with NEW names
    # # Add encoded categoricals if needed: 'k_car_model_trip_A', ... 'car_model_trip', 'manufacturer_trip', 'battery_type_trip'

    # --- 1. Start-of-Segment k & Static Features ---
    for col_base in START_OF_SEGMENT_FEATURES:
        k_col_name = f'k_{col_base}'
        if k_col_name in feature_order_to_build:
            features[k_col_name] = current_segment_row.get(col_base, np.nan)

    # --- 2. Predicted Segment k Features --- (Same as walk-forward)
    # # Use 'k_' prefix and NEW descriptive names from current_segment_row
    # features['k_distance_seg_pred_m'] = current_segment_row.get('distance_seg_pred_m', np.nan)
    # features['k_duration_seg_pred_s'] = current_segment_row.get('duration_seg_pred_s', np.nan)
    # features['k_speed_seg_pred_kph'] = current_segment_row.get('speed_seg_pred_kph', np.nan)
    # #features['k_elevation_gain_seg_pred'] = current_segment_row.get('elevation_gain_seg_pred', np.nan)
    # #features['k_elevation_loss_seg_pred'] = current_segment_row.get('elevation_loss_seg_pred', np.nan)
    # # Add other predicted features using .get() with NEW names

    # --- 2. Predicted Segment k Features ---
    for col_base in PREDICTED_SEGMENT_FEATURES:
        k_col_name = f'k_{col_base}'
        if k_col_name in feature_order_to_build:
            features[k_col_name] = current_segment_row.get(col_base, np.nan)

    # --- 3. Global Prediction for Segment k --- (Same as walk-forward)
    # Use 'k_' prefix and NEW descriptive name from current_segment_row
    if 'k_global_pred_soc_seg_delta' in feature_order_to_build:
        features['k_global_pred_soc_seg_delta'] = current_segment_row.get(GLOBAL_PRED_COL, np.nan)

    # --- 4. Aggregates over OTHER Segments (j != k) ---
    # *** Calculate NEW driving/routing behavior aggregates ***
    if 'others_segment_count' in feature_order_to_build:
        features['others_segment_count'] = len(other_segments_df)
    
    if not other_segments_df.empty:
        others_total_dist_km = other_segments_df['distance_seg_actual_km'].sum()
        others_total_dur_s = other_segments_df['duration_seg_actual_s'].sum()
        others_total_dur_min = others_total_dur_s / 60.0 if others_total_dur_s > 0 else 0

        if 'others_total_distance_actual_km' in feature_order_to_build: features['others_total_distance_actual_km'] = others_total_dist_km
        if 'others_total_duration_actual_s' in feature_order_to_build: features['others_total_duration_actual_s'] = others_total_dur_s

        # Calculate averages and stds for all outcome features
        for col_base in SEGMENT_OUTCOME_FEATURES:
            # Averages
            avg_feature_name = f'others_avg_{col_base.replace(RESIDUAL_TARGET_COL, "residual_error_soc_seg_delta")}'
            if avg_feature_name in feature_order_to_build:
                features[avg_feature_name] = other_segments_df[col_base].mean()

            # Standard Deviations
            std_feature_name = f'others_std_{col_base.replace(RESIDUAL_TARGET_COL, "residual_error_soc_seg_delta")}'
            if std_feature_name in feature_order_to_build:
                features[std_feature_name] = other_segments_df[col_base].std() 

            # Totals (for specific count-like features)
            total_feature_name = f'others_total_{col_base}'
            if col_base in ['stops_seg_count', 'accel_high_event_count', 'decel_high_event_count', 'stop_duration_seg_agg_s'] and total_feature_name in feature_order_to_build:
                features[total_feature_name] = other_segments_df[col_base].sum()  

        # Special MAE for residual error
        if 'others_abs_avg_residual_error_soc_seg_delta' in feature_order_to_build:
            features['others_abs_avg_residual_error_soc_seg_delta'] = other_segments_df[RESIDUAL_TARGET_COL].abs().mean() 

        # Calculate Rates
        others_total_stops = other_segments_df['stops_seg_count'].sum()
        others_total_accel_high = other_segments_df['accel_high_event_count'].sum()
        others_total_decel_high = other_segments_df['decel_high_event_count'].sum()
        others_total_stop_duration = other_segments_df['stop_duration_seg_agg_s'].sum()

        if 'others_rate_stops_per_km' in feature_order_to_build:
            features['others_rate_stops_per_km'] = (others_total_stops / others_total_dist_km) if others_total_dist_km > 1e-6 else 0.0
        if 'others_rate_stops_per_minute' in feature_order_to_build:
            features['others_rate_stops_per_minute'] = (others_total_stops / others_total_dur_min) if others_total_dur_min > 1e-6 else 0.0
        if 'others_rate_accel_high_events_per_km' in feature_order_to_build:
            features['others_rate_accel_high_events_per_km'] = (others_total_accel_high / others_total_dist_km) if others_total_dist_km > 1e-6 else 0.0
        if 'others_rate_accel_high_events_per_minute' in feature_order_to_build:
            features['others_rate_accel_high_events_per_minute'] = (others_total_accel_high / others_total_dur_min) if others_total_dur_min > 1e-6 else 0.0
        if 'others_rate_decel_high_events_per_km' in feature_order_to_build:
            features['others_rate_decel_high_events_per_km'] = (others_total_decel_high / others_total_dist_km) if others_total_dist_km > 1e-6 else 0.0
        if 'others_rate_decel_high_events_per_minute' in feature_order_to_build:
            features['others_rate_decel_high_events_per_minute'] = (others_total_decel_high / others_total_dur_min) if others_total_dur_min > 1e-6 else 0.0
        if 'others_ratio_stop_duration_to_total_duration' in feature_order_to_build:
            features['others_ratio_stop_duration_to_total_duration'] = (others_total_stop_duration / others_total_dur_s) if others_total_dur_s > 1e-6 else 0.0  
    
        # Use NEW outcome names
        # features['others_avg_altitude_seg_agg_delta'] = other_segments_df['altitude_seg_agg_delta'].mean()
        # Add new driving behavior aggregates
        # Add other aggregates if defined in feature_order

        # if 'others_avg_gt_soc_seg_delta' in feature_order: features['others_avg_gt_soc_seg_delta'] = other_segments_df['gt_soc_seg_delta'].mean()
        # if 'others_avg_distance_seg_actual_km' in feature_order: features['others_avg_distance_seg_actual_km'] = other_segments_df['distance_seg_actual_km'].mean()
        # if 'others_avg_duration_seg_actual_s' in feature_order: features['others_avg_duration_seg_actual_s'] = other_segments_df['duration_seg_actual_s'].mean()
        # if 'others_avg_speed_seg_agg_mean_mps' in feature_order: features['others_avg_speed_seg_agg_mean_mps'] = other_segments_df['speed_seg_agg_mean_mps'].mean()
        # if 'others_std_speed_seg_agg_mean_mps' in feature_order: features['others_std_speed_seg_agg_mean_mps'] = other_segments_df['speed_seg_agg_mean_mps'].std()
        # if 'others_avg_temp_seg_agg_mean' in feature_order: features['others_avg_temp_seg_agg_mean'] = other_segments_df['temp_seg_agg_mean'].mean()
        # if 'others_avg_accel_seg_agg_mean' in feature_order: features['others_avg_accel_seg_agg_mean'] = other_segments_df['accel_seg_agg_mean'].mean()
        # if 'others_avg_decel_seg_agg_mean' in feature_order: features['others_avg_decel_seg_agg_mean'] = other_segments_df['decel_seg_agg_mean'].mean()
        # if 'others_avg_jerk_abs_seg_agg_mean' in feature_order: features['others_avg_jerk_abs_seg_agg_mean'] = other_segments_df['jerk_abs_seg_agg_mean'].mean()
        # if 'others_total_stops_seg_count' in feature_order: features['others_total_stops_seg_count'] = other_segments_df['stops_seg_count'].sum()
        # if 'others_avg_stop_duration_seg_agg_s' in feature_order: features['others_avg_stop_duration_seg_agg_s'] = other_segments_df['stop_duration_seg_agg_s'].mean()
        # if 'others_avg_route_deviation_pct' in feature_order: features['others_avg_route_deviation_pct'] = other_segments_df['route_deviation_pct'].mean()
        # if 'others_avg_percent_time_over_limit' in feature_order: features['others_avg_percent_time_over_limit'] = other_segments_df['percent_time_over_limit_seg_agg'].mean()
        # if 'others_avg_match_confidence' in feature_order: features['others_avg_match_confidence'] = other_segments_df['match_confidence'].mean()
        # if 'others_avg_residual_error' in feature_order: features['others_avg_residual_error'] = other_segments_df[RESIDUAL_TARGET_COL].mean()
        # if 'others_abs_avg_residual_error' in feature_order: features['others_abs_avg_residual_error'] = other_segments_df[RESIDUAL_TARGET_COL].abs().mean()
        # if 'others_std_residual_error' in feature_order: features['others_std_residual_error'] = other_segments_df[RESIDUAL_TARGET_COL].std()
        # # if 'others_total_distance_actual_km' in feature_order: features['others_total_distance_actual_km'] = other_segments_df['distance_seg_actual_km'].sum()
        # # if 'others_total_duration_actual_s' in feature_order: features['others_total_duration_actual_s'] = other_segments_df['duration_seg_actual_s'].

    else:
        # Assign defaults for all 'other_' features defined in LOSO_FEATURE_ORDER
        for key in feature_order_to_build:
            if key.startswith('others_'):
                if 'count' in key or 'total' in key: features[key] = 0.0
                else: features[key] = np.nan  
    
    # --- Ensure consistent order and create DataFrame ---
    # Use the globally defined LOSO_FEATURE_ORDER list
    final_features_dict = {key: features.get(key, np.nan) for key in feature_order_to_build}
    feature_vector_df = pd.DataFrame([final_features_dict], columns=feature_order_to_build)
    
    # Do not check for NaNs here, let the calling function handle it after subsetting
    # if feature_vector_df.isnull().values.any():
    #      # nan_cols = feature_vector_df.columns[feature_vector_df.isnull().any()].tolist()
    #      # warnings.warn(f"DEBUG LOSO: NaNs found in feature vector for segment {segment_id_debug}. NaN columns: {nan_cols}")
    #      return None
    return feature_vector_df

# --- Apply Functions for Walk-Forward and LOSO ---

def apply_walk_forward_evaluation(cycle_df, models_dict, min_train_samples=2): # Needs editing!
    """
    Applies walk-forward training and prediction for MULTIPLE residual models.
    Returns predictions for all models for evaluation.

    Args:
        cycle_df (pd.DataFrame): Segments for a single trip_id, sorted.
        models_dict (dict): Dictionary of model instances {'name': model_obj}.
        min_train_samples (int): Minimum history segments needed for training.

    Returns:
        pd.DataFrame: DataFrame with predicted residuals for each model, indexed like cycle_df.
    """
    cycle_df = cycle_df.sort_values(SEGMENT_ID_COL)
    # Initialize DataFrame to store predictions for all models
    predictions_df = pd.DataFrame(index=cycle_df.index)

    for model_name in models_dict.keys():
        predictions_df[f'pred_{model_name}'] = np.nan # Initialize columns

    trip_id = cycle_df[TRIP_ID_COL].iloc[0]
    #print(f"\nDEBUG: Processing Trip {trip_id}, Segments: {cycle_df[SEGMENT_ID_COL].tolist()}") # Print segments in trip

    # Iterate through segments k (index i) starting from where we have enough history
    # to potentially form a training set of size min_train_samples.
    # The first segment we can *predict* is segment min_train_samples + 1 (index i = min_train_samples)
    # Example: min_train_samples=2 -> first prediction for k=3 (i=2)
    for i in range(min_train_samples, len(cycle_df)):
        current_segment_k = cycle_df[SEGMENT_ID_COL].iloc[i]
        current_segment_row = cycle_df.iloc[i]
        current_segment_index = cycle_df.index[i]
        history_df = cycle_df.iloc[:i] # History is segments 1 to k-1 (indices 0 to i-1)

        #print(f"  DEBUG: Predicting for Segment k={current_segment_k} (index i={i})")

        # --- Build Feature Vector for Prediction (Segment k) ---
        X_pred_df = build_walk_forward_feature_vector(current_segment_row, history_df)
        if X_pred_df is None:
            print(f"    DEBUG: Failed to build prediction feature vector X_pred_df for seg {current_segment_k}. Skipping.")
            warnings.warn(f"Could not build valid feature vector for prediction for {trip_id}, segment {current_segment_k}. Skipping.")
            continue # Skip prediction if features invalid

        # --- Build ALL available Training Data from history (segments 1 to k-1) ---
        X_train_list = []
        y_train_list = []
        # We can potentially train using targets from segment 2 up to segment k-1
        # The features for target j are built using history 1 to j-1
        # The first target we can use is from segment 2 (index j=1), using history segment 1 (index 0)
        for j in range(1, i): # Loop through indices 1 to i-1 (segments 2 to k-1)
            train_target_segment_row = cycle_df.iloc[j] # Segment j+1 provides target y
            train_target_segment_id = train_target_segment_row.get(SEGMENT_ID_COL, 'N/A')
            train_history_df = cycle_df.iloc[:j] # History is segments 1 to j
            target_y = train_target_segment_row.get(RESIDUAL_TARGET_COL, np.nan)

            #print(f"      DEBUG: Trying training sample: Target from seg {train_target_segment_id} (index j={j}), History 1 to {j}")

            if pd.isna(target_y): # Skip if target is NaN for this training row
                print(f"      DEBUG: Target y ({RESIDUAL_TARGET_COL}) for training seg {train_target_segment_id} is NaN. Skipping sample.")
                continue

            # Features are built based on start conditions of segment j+1 and history 1 to j
            X_train_j_df = build_walk_forward_feature_vector(train_target_segment_row, train_history_df)

            if X_train_j_df is not None:
                if not X_train_j_df.isnull().values.any():
                    X_train_list.append(X_train_j_df.iloc[0].to_dict())
                    y_train_list.append(target_y)
                    #print(f"        DEBUG: Added valid training sample (Features built successfully, Target={target_y:.4f}).")
                else:
                    nan_cols_train = X_train_j_df.columns[X_train_j_df.isnull().any()].tolist()
                    print(f"        DEBUG: Built training feature vector, but found NaNs: {nan_cols_train}. Skipping sample.")
            else:
                 print(f"        DEBUG: Failed to build training feature vector X_train_j_df. Skipping sample.")

        # --- Check if ANY valid training data was generated ---
        # We need at least one valid (X, y) pair to attempt fitting.
        # Change the check from '< min_train_samples' to '< 1' or allow model fit to handle it.
        # Let's require at least 1 sample explicitly.
        if len(X_train_list) < 1: # *** CHANGED THIS LINE ***
            warnings.warn(f"Insufficient valid training samples ({len(X_train_list)} < 1) generated for {trip_id}, segment {current_segment_k}. Skipping prediction.")
            continue

        # Convert training data lists to DataFrame/Series
        X_train = pd.DataFrame(X_train_list, columns=FINAL_FEATURE_ORDER_WALK_FORWARD)
        y_train = pd.Series(y_train_list)
        
        # # Final check before fitting
        # if X_train.isnull().values.any() or y_train.isnull().values.any():
        #      warnings.warn(f"NaNs detected in final training data for {trip_id}, segment {current_segment_k}. Skipping.")
        #      continue

        # --- Fit and Predict for EACH model ---
        for model_name, model_instance in models_dict.items():
            try:
                # Create a fresh instance for each fit to avoid state issues
                # (Important for stateful models, good practice otherwise)
                current_model = model_instance.__class__(**model_instance.get_params())
                current_model.fit(X_train, y_train)
                residual_prediction = current_model.predict(X_pred_df)
                predictions_df.loc[current_segment_index, f'pred_{model_name}'] = residual_prediction[0]
            except ValueError as ve: # Catch fit errors (e.g., insufficient samples)
                 warnings.warn(f"ValueError fit/predict {model_name} for {trip_id}-{current_segment_k}: {ve}.")
            except Exception as e:
                warnings.warn(f"Error fit/predict {model_name} for {trip_id}-{current_segment_k}: {e}.")

    return predictions_df # Return DataFrame with predictions from all models

def apply_loso_evaluation(cycle_df, models_dict, min_train_samples=1):
    """
    Applies Leave-One-Segment-Out evaluation for multiple models.
    Trains on all segments j != k to predict segment k.

    Args:
        cycle_df (pd.DataFrame): Segments for a single trip_id, sorted.
        models_dict (dict): Dictionary of model instances {'name': model_obj}.
        min_train_samples (int): Minimum number of OTHER segments required for training.

    Returns:
        pd.DataFrame: DataFrame with predicted residuals for each model, indexed like cycle_df.
    """
    cycle_df = cycle_df.sort_values(SEGMENT_ID_COL)
    predictions_df = pd.DataFrame(index=cycle_df.index)
    trip_id = cycle_df[TRIP_ID_COL].iloc[0]
    for model_name in models_dict.keys(): predictions_df[f'pred_{model_name}'] = np.nan
    
    fold_feature_stats_for_trip = [] # NEW: To store (considered_list, removed_list) for each fold

    # Iterate through each segment k (index i) to predict
    for i in range(len(cycle_df)):
        current_segment_row = cycle_df.iloc[i]
        current_segment_index = cycle_df.index[i] 
        current_segment_k = cycle_df[SEGMENT_ID_COL].iloc[i]

        # Define training data: all segments EXCEPT k
        other_segments_df = cycle_df.drop(index=current_segment_index)

        # Check if enough OTHER segments exist for training
        if len(other_segments_df) < min_train_samples:
            # warnings.warn(f"LOSO: Not enough other segments ({len(other_segments_df)} < {min_train_samples}) to train for {trip_id}, segment {current_segment_k}. Skipping.")
            continue

        # --- Build Feature Vector for Prediction (Segment k) ---
        # Features are based on k's start/pred values and aggregates over OTHER segments
        X_pred_df = build_loso_feature_vector(current_segment_row, other_segments_df)
        if X_pred_df is None:
            # warnings.warn(f"LOSO: Could not build prediction feature vector for {trip_id}, segment {current_segment_k}. Skipping.")
            continue

        # --- Build Training Data (using other_segments_df) ---
        # Target is the residual from the 'other' segments
        #y_train = other_segments_df[RESIDUAL_TARGET_COL].dropna()

        # Features for training set: For each row j in 'other', build features
        # based on j's start/pred values and aggregates over {all} - {j}.
        # This is computationally heavy if done perfectly.
        # Pragmatic approach: Use the same feature structure as prediction,
        # calculating aggregates over the 'other' set defined for each training row.
        # Simplest approach: Use the features calculated for each row j when it was the 'current' row.
        # Let's try the pragmatic approach first: build features for each training row based on *its* context.
        y_train_aligned_list = []
        temp_X_train_list_for_df = []
        #valid_y_train_indices = y_train.index # Indices where target is not NaN
        valid_y_train_indices = other_segments_df[RESIDUAL_TARGET_COL].dropna().index
        if len(valid_y_train_indices) < min_train_samples: # Check after dropping NaN targets
             # warnings.warn(f"LOSO: Not enough valid targets in other segments for {trip_id}, segment {current_segment_k}. Skipping.")
             continue

        for train_idx in valid_y_train_indices:
            train_segment_row = cycle_df.loc[train_idx] # Get the row from original df
            train_other_segments_df = cycle_df.drop(index=train_idx) # Define 'other' for this training row
            # Ensure enough 'other' segments for this training point's feature calculation
            if len(train_other_segments_df) >= min_train_samples:
                 X_train_j_df = build_loso_feature_vector(train_segment_row, train_other_segments_df)
                 if X_train_j_df is not None and not X_train_j_df.isnull().values.any():
                      temp_X_train_list_for_df.append(X_train_j_df.iloc[0].to_dict())
                      y_train_aligned_list.append(cycle_df.loc[train_idx, RESIDUAL_TARGET_COL])
                 # else: warnings.warn(f"LOSO: Skipping training sample {train_idx} for {trip_id}-{current_segment_k} due to invalid features.")
            # else: warnings.warn(f"LOSO: Skipping training sample {train_idx} for {trip_id}-{current_segment_k} due to insufficient 'other' segments.")

        # Check if enough valid training feature vectors were generated
        if len(temp_X_train_list_for_df) < min_train_samples:
            # warnings.warn(f"LOSO: Not enough valid training feature vectors ({len(X_train_list)}) generated for {trip_id}, segment {current_segment_k}. Skipping.")
            continue

        X_train = pd.DataFrame(temp_X_train_list_for_df, columns=FINAL_FEATURE_ORDER_LOSO)
        # Align y_train with the successfully generated X_train rows
        y_train = pd.Series(y_train_aligned_list, index=X_train.index if len(y_train_aligned_list) == len(X_train) else None)

        # Default to original features
        X_train_for_fitting = X_train
        X_pred_for_predicting = X_pred_df

        # --- Apply Near-Zero Variance Feature Removal ---
        if not X_train.empty and X_train.shape[1] > 0:
            considered_features_this_fold = list(X_train.columns)
            removed_features_this_fold = [] # Default: no features removed by NZV

            if X_train.shape[0] > 1: # Need >1 samples for variance calculation
                try:
                    selector = VarianceThreshold(threshold=NZV_THRESHOLD)
                    selector.fit(X_train)
                    selected_features_mask = selector.get_support()

                    if np.any(selected_features_mask):
                        X_train_for_fitting = X_train.loc[:, selected_features_mask]
                        X_pred_for_predicting = X_pred_df.loc[:, selected_features_mask]
                        removed_features_this_fold = X_train.columns[~selected_features_mask].tolist()
                    else: # All features had variance below threshold
                        warnings.warn(f"LOSO NZV: All features removed by VarianceThreshold (var < {NZV_THRESHOLD}) for {trip_id}, segment {current_segment_k}. No features to train on.")
                        X_train_for_fitting = pd.DataFrame(columns=X_train.columns)
                        X_pred_for_predicting = pd.DataFrame(columns=X_pred_df.columns)
                        removed_features_this_fold = list(X_train.columns) # All considered were removed
                
                except Exception as e_nzv:
                    warnings.warn(f"LOSO NZV: Error during variance thresholding for {trip_id}, segment {current_segment_k}: {e_nzv}. Using original features.")
                    # removed_features_this_fold remains empty, X_train_for_fitting is original X_train
            
            else: # Not enough samples for variance calculation, NZV skipped
                 warnings.warn(f"LOSO NZV: Not enough samples ({X_train.shape[0]}) in X_train for variance thresholding for {trip_id}, segment {current_segment_k}. Using original features.")
                 # removed_features_this_fold remains empty
            
            fold_feature_stats_for_trip.append(
                (considered_features_this_fold, removed_features_this_fold)
            )
        # If X_train was empty or had no columns, no stats are recorded for this fold.

        # --- Fit and Predict for EACH model ---
        for model_name, model_instance in models_dict.items():
            try:
                if X_train_for_fitting.empty or X_train_for_fitting.shape[1] == 0:
                    reason = "initial X_train was empty or had no features"
                    if X_train.shape[1] > 0 and X_train_for_fitting.shape[1] == 0 : # Original had features, NZV removed all
                        reason = f"all features removed by NZV (threshold {NZV_THRESHOLD})"
                    warnings.warn(f"LOSO: Skipping model {model_name} for {trip_id}-{current_segment_k} as there are {reason}.")
                    continue

                current_model = model_instance.__class__(**model_instance.get_params())
                current_model.fit(X_train_for_fitting, y_train)
                residual_prediction = current_model.predict(X_pred_for_predicting)
                predictions_df.loc[current_segment_index, f'pred_{model_name}'] = residual_prediction[0]
            except ValueError as ve: warnings.warn(f"LOSO ValueError fit/predict {model_name} for {trip_id}-{current_segment_k} (Features: {X_train_for_fitting.shape[1]}): {ve}.")
            except Exception as e: warnings.warn(f"LOSO Error fit/predict {model_name} for {trip_id}-{current_segment_k} (Features: {X_train_for_fitting.shape[1]}): {e}.")

    return predictions_df, fold_feature_stats_for_trip

# --- MODIFIED Main Orchestration Function ---
def evaluate_and_predict_residuals(segments_df, models_dict, method='loso_global_residual', primary_metric='MAE',
                                   min_train_samples=2,
                                   trip_level_split_test_size=0.2,
                                   random_state_split=RANDOM_STATE,
                                   collinearity_threshold=0.90,
                                   NZV_THRESHOLD=0.001): # Added NZV_THRESHOLD parameter
    start_time = time.time()
    print(f"\n--- Starting {method.upper()} Evaluation & Prediction ---")

    required_cols = [TRIP_ID_COL, SEGMENT_ID_COL, GT_TARGET_COL, GLOBAL_PRED_COL, RESIDUAL_TARGET_COL]
    required_cols.extend(SEGMENT_OUTCOME_FEATURES); required_cols.extend(PREDICTED_SEGMENT_FEATURES)
    required_cols.extend(START_OF_SEGMENT_FEATURES); required_cols = sorted(list(set(required_cols)))
    missing_cols = [col for col in required_cols if col not in segments_df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns for dynamic feature building: {missing_cols}")

    num_total_segments = len(segments_df)
    num_total_trips = segments_df[TRIP_ID_COL].nunique()
    print(f"Processing {num_total_segments} segments from {num_total_trips} unique trips using {method} method...")

    evaluation_set_predictions_df = pd.DataFrame()
    # --- NEW: For storing all engineered features ---
    all_engineered_features_for_output_list = []
    all_engineered_features_indices = []

    # feature_removal_counts_loso_per_trip = defaultdict(int)
    # feature_considered_counts_loso_per_trip = defaultdict(int)
    
    # # For LOSO-Global-Residual feature removal stats
    # global_loso_train_considered_features = []
    # global_loso_train_removed_features_nZV = []
    # global_loso_train_removed_features_collinear = []

    if method.lower() == 'walk_forward':
        print(f" - Applying Walk-Forward (min_train_samples={min_train_samples})...")
        wf_results_list = []
        for _, group_df in segments_df.groupby(TRIP_ID_COL, sort=False):
            wf_results_list.append(
                apply_walk_forward_evaluation(group_df, models_dict, min_train_samples)
            )
        if wf_results_list: evaluation_set_predictions_df = pd.concat(wf_results_list)
        else:
            pred_cols = [f'pred_{name}' for name in models_dict.keys()]
            evaluation_set_predictions_df = pd.DataFrame(columns=pred_cols, index=segments_df.index)

    elif method.lower() == 'loso':
        print(f" - Applying Leave-One-Segment-Out (Per-Trip Models, min_train_samples={min_train_samples})...")
        loso_results_list = []
        for _, group_df in segments_df.groupby(TRIP_ID_COL, sort=False):
            preds_df_for_trip, fold_stats_list_for_trip = apply_loso_evaluation(
                group_df, models_dict, min_train_samples
            )
            if not preds_df_for_trip.empty:
                loso_results_list.append(preds_df_for_trip)
            for considered_in_fold, removed_in_fold in fold_stats_list_for_trip:
                for feature_name in considered_in_fold: feature_considered_counts_loso_per_trip[feature_name] += 1
                for feature_name in removed_in_fold: feature_removal_counts_loso_per_trip[feature_name] += 1
        
        if loso_results_list:
            evaluation_set_predictions_df = pd.concat(loso_results_list) # Corrected: populate evaluation_set_predictions_df
        else:
            pred_cols_for_empty_df = [f'pred_{model_name}' for model_name in models_dict.keys()]
            evaluation_set_predictions_df = pd.DataFrame(columns=pred_cols_for_empty_df, index=segments_df.index)

    elif method.lower() == 'loso_global_residual':
        print(f" - Applying LOSO-Style Features for a Global Residual Model with Trip-Level Split...")
        train_trip_ids, test_trip_ids = split_trips_train_test(
            segments_df, TRIP_ID_COL, trip_level_split_test_size, random_state_split
        )
        if (len(train_trip_ids) == 0 or len(test_trip_ids) == 0) and num_total_trips > 1:
             warnings.warn(f"Trip split resulted in empty train ({len(train_trip_ids)}) or test ({len(test_trip_ids)}) set. Using all trips for train/test.")
             if num_total_trips > 0:
                 train_trip_ids = segments_df[TRIP_ID_COL].unique()
                 test_trip_ids = segments_df[TRIP_ID_COL].unique()
             else: raise ValueError("No trips in segments_df for loso_global_residual.")

        print("   Generating ALL POSSIBLE LOSO-style features for TRAINING trips...")
        train_segments_df_filtered = segments_df[segments_df[TRIP_ID_COL].isin(train_trip_ids)]
        all_X_train_loso_wide_list, all_y_train_loso_list = [], []
        for _, cycle_df_orig in train_segments_df_filtered.groupby(TRIP_ID_COL, sort=False):
            cycle_df = cycle_df_orig.sort_values(SEGMENT_ID_COL)
            if len(cycle_df) <= min_train_samples: continue
            for i in range(len(cycle_df)):
                current_segment_row = cycle_df.iloc[i]
                other_segments_df = cycle_df.drop(index=cycle_df.index[i])
                if len(other_segments_df) < min_train_samples: continue
                # Build the WIDE set of features
                X_seg_df_wide = build_loso_feature_vector(current_segment_row, other_segments_df)
                y_seg = current_segment_row.get(RESIDUAL_TARGET_COL)
                if X_seg_df_wide is not None and not pd.isna(y_seg): # NaNs in features will be handled after subsetting
                    all_X_train_loso_wide_list.append(X_seg_df_wide.iloc[0].to_dict())
                    all_y_train_loso_list.append(y_seg)

        if not all_X_train_loso_wide_list:
            raise ValueError("No LOSO-style features (wide set) generated for TRAINING global residual model.")

        # Convert WIDE features to DataFrame
        X_train_all_loso_wide = pd.DataFrame(all_X_train_loso_wide_list, columns=ALL_POSSIBLE_LOSO_FEATURES)
        y_train_all_loso = pd.Series(all_y_train_loso_list)

        # --- Select SUBSET of features for modeling using FINAL_FEATURE_ORDER_LOSO ---
        print(f"   Selecting subset of {len(FINAL_FEATURE_ORDER_LOSO)} features for model training from {X_train_all_loso_wide.shape[1]} engineered features.")
        # Ensure all columns in FINAL_FEATURE_ORDER_LOSO exist in the wide set, warn if not
        missing_in_wide = [col for col in FINAL_FEATURE_ORDER_LOSO if col not in X_train_all_loso_wide.columns]
        if missing_in_wide:
            warnings.warn(f"Warning: The following features from FINAL_FEATURE_ORDER_LOSO were not found in the ALL_POSSIBLE_LOSO_FEATURES output: {missing_in_wide}. They will be missing from the model input.")
        
        # Select only the features intended for the model, handle missing ones by not selecting them
        features_for_model_training = [col for col in FINAL_FEATURE_ORDER_LOSO if col in X_train_all_loso_wide.columns]
        X_train_all_loso_subset = X_train_all_loso_wide[features_for_model_training].copy()
        
        # --- Handle NaNs in the SUBSET that goes into the model ---
        # Option 1: Drop rows with any NaNs in the selected features
        nan_rows_mask_train = X_train_all_loso_subset.isnull().any(axis=1)
        if nan_rows_mask_train.any():
            print(f"   Dropping {nan_rows_mask_train.sum()} training rows due to NaNs in selected model features.")
            X_train_all_loso_subset = X_train_all_loso_subset[~nan_rows_mask_train]
            y_train_all_loso = y_train_all_loso[~nan_rows_mask_train]

        if X_train_all_loso_subset.empty:
            raise ValueError("No training data remaining after NaN handling in selected model features.")
        print(f"   Generated {X_train_all_loso_subset.shape[0]} training feature vectors for the model (shape: {X_train_all_loso_subset.shape}).")


        # --- Preprocessing for TRAINING data (on the SUBSET) ---
        X_train_processed = X_train_all_loso_subset # This is now the subset
        # ... (rest of NZV, collinearity, preprocessor fitting on X_train_processed)
        global_loso_train_considered_features = list(X_train_processed.columns)
        num_cols_train = X_train_processed.select_dtypes(include=np.number).columns.tolist()
        cat_cols_train = X_train_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"   Applying Near-Zero Variance filter (threshold={NZV_THRESHOLD}) to numerical features in TRAINING set...")
        X_train_num_nZV = X_train_processed[num_cols_train].copy()
        nZV_selector = None
        global_loso_train_removed_features_nZV = [] # Initialize here
        if num_cols_train and X_train_processed[num_cols_train].shape[0] > 1:
            nZV_selector = VarianceThreshold(threshold=NZV_THRESHOLD)
            try:
                nZV_selector.fit(X_train_processed[num_cols_train])
                selected_num_mask_nZV = nZV_selector.get_support()
                X_train_num_nZV = X_train_processed[num_cols_train].loc[:, selected_num_mask_nZV].copy()
                removed_nZV = X_train_processed[num_cols_train].columns[~selected_num_mask_nZV].tolist()
                global_loso_train_removed_features_nZV.extend(removed_nZV)
                print(f"     NZV removed {len(removed_nZV)} numerical features.")
            except Exception as e_nzv:
                warnings.warn(f"Error during training NZV on numerical features: {e_nzv}. Using all numerical features for NZV step.")
                nZV_selector = None
        
        print("   Applying Collinearity filter to NZV'd numerical features in TRAINING set...")
        X_train_num_final = X_train_num_nZV.copy()
        global_loso_train_removed_features_collinear = [] # Initialize here
        if not X_train_num_nZV.empty:
            X_train_num_final, removed_coll = remove_highly_collinear_numerical_features(
                X_train_num_nZV, threshold=collinearity_threshold
            )
            global_loso_train_removed_features_collinear.extend(removed_coll)
        else: print("     Skipping collinearity: no numerical features after NZV.")
        
        # Reconstruct X_train_processed with the actual features after NZV and collinearity
        X_train_processed = pd.concat([X_train_num_final, X_train_processed[cat_cols_train].copy()], axis=1)
        print(f"   Shape of training features after all preprocessing: {X_train_processed.shape}")


        print("   Generating ALL POSSIBLE LOSO-style features for TESTING trips...")
        test_segments_df_filtered = segments_df[segments_df[TRIP_ID_COL].isin(test_trip_ids)]
        all_X_test_loso_wide_list, all_y_test_actuals_list, all_test_original_indices = [], [], []
        for _, cycle_df_orig in test_segments_df_filtered.groupby(TRIP_ID_COL, sort=False):
            cycle_df = cycle_df_orig.sort_values(SEGMENT_ID_COL)
            if len(cycle_df) <= min_train_samples: continue
            for i in range(len(cycle_df)):
                current_segment_row = cycle_df.iloc[i]
                other_segments_df = cycle_df.drop(index=cycle_df.index[i])
                if len(other_segments_df) < min_train_samples: continue
                X_seg_df_wide = build_loso_feature_vector(current_segment_row, other_segments_df)
                y_seg = current_segment_row.get(RESIDUAL_TARGET_COL)
                if X_seg_df_wide is not None and not pd.isna(y_seg):
                    all_X_test_loso_wide_list.append(X_seg_df_wide.iloc[0].to_dict())
                    all_y_test_actuals_list.append(y_seg)
                    all_test_original_indices.append(cycle_df.index[i])
        
        X_test_processed_subset = pd.DataFrame(columns=X_train_processed.columns) # For model prediction
        y_test_all_actuals_series = pd.Series(dtype=float)

        if all_X_test_loso_wide_list:
            X_test_all_loso_wide = pd.DataFrame(all_X_test_loso_wide_list, columns=ALL_POSSIBLE_LOSO_FEATURES)
            y_test_all_actuals_series = pd.Series(all_y_test_actuals_list, index=pd.Index(all_test_original_indices))
            print(f"   Generated {X_test_all_loso_wide.shape[0]} test feature vectors (wide set).")

            # Select SUBSET for model prediction, align with X_train_processed columns
            X_test_subset_for_model = X_test_all_loso_wide[features_for_model_training].copy()
            
            # --- Handle NaNs in the test SUBSET ---
            nan_rows_mask_test = X_test_subset_for_model.isnull().any(axis=1)
            if nan_rows_mask_test.any():
                # For test data, we usually can't drop rows if we need to make predictions for them.
                # Imputation is preferred. The preprocessor (StandardScaler, OneHotEncoder) handles this.
                # If not using a preprocessor that imputes, you'd need to impute here.
                # For now, we assume the preprocessor handles it or the model can handle NaNs.
                # If dropping, you'd need to align y_test_all_actuals_series.
                warnings.warn(f"   {nan_rows_mask_test.sum()} test rows have NaNs in selected model features. Preprocessor should handle this.")
                # Example if you had to drop:
                # X_test_subset_for_model = X_test_subset_for_model[~nan_rows_mask_test]
                # y_test_all_actuals_series = y_test_all_actuals_series[~nan_rows_mask_test]


            # Apply the same NZV and collinearity transformations as training
            X_test_num_raw = X_test_subset_for_model[num_cols_train].copy() # Use num_cols_train from training
            X_test_cat_raw = X_test_subset_for_model[cat_cols_train].copy() # Use cat_cols_train from training
            
            X_test_num_nZV = X_test_num_raw
            if nZV_selector is not None and not X_test_num_raw.empty:
                try:
                    X_test_num_nZV_np = nZV_selector.transform(X_test_num_raw)
                    X_test_num_nZV = pd.DataFrame(X_test_num_nZV_np, columns=X_train_num_nZV.columns, index=X_test_num_raw.index)
                except Exception as e_nzv_test:
                    warnings.warn(f"Error applying training NZV to test numerical: {e_nzv_test}.")
            
            X_test_num_final = X_test_num_nZV.copy()
            if not X_test_num_nZV.empty:
                cols_to_keep_after_collinearity = X_train_num_final.columns # From training
                valid_cols_in_test_num = [col for col in cols_to_keep_after_collinearity if col in X_test_num_nZV.columns]
                X_test_num_final = X_test_num_nZV[valid_cols_in_test_num]
            
            X_test_processed_subset = pd.concat([X_test_num_final, X_test_cat_raw], axis=1)
            # Final alignment to ensure exact same columns as X_train_processed
            X_test_processed_subset = X_test_processed_subset.reindex(columns=X_train_processed.columns, fill_value=np.nan) # Fill with NaN if a training col is missing after test processing
        else:
            warnings.warn("No valid LOSO-style features (wide set) for TESTING.")
        print(f"   Shape of test features for model after all preprocessing: {X_test_processed_subset.shape}")


        # --- Model Training and Prediction (using X_train_processed and X_test_processed_subset) ---
        # ... (preprocessor_general definition and model fitting loop remains largely the same,
        #      but it operates on X_train_processed and X_test_processed_subset)
        final_num_cols_for_pipe = X_train_processed.select_dtypes(include=np.number).columns.tolist()
        final_cat_cols_for_pipe = X_train_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        transformers_general = []
        if final_num_cols_for_pipe: transformers_general.append(('num', StandardScaler(), final_num_cols_for_pipe))
        if final_cat_cols_for_pipe: transformers_general.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), final_cat_cols_for_pipe))
        
        preprocessor_general = ColumnTransformer(transformers=transformers_general, remainder='passthrough' if not transformers_general else 'drop')
        if not transformers_general: warnings.warn("No num/cat features for general preprocessing pipeline.")

        temp_eval_preds_storage = pd.DataFrame(index=y_test_all_actuals_series.index) # Store predictions on test set
        
        if X_train_processed.empty or X_train_processed.shape[1] == 0:
            warnings.warn("No training features after preprocessing for LOSO-Global-Residual.")
        elif X_test_processed_subset.empty and not y_test_all_actuals_series.empty: # Check if test features are empty but we have a target
            warnings.warn("No test features for LOSO-Global-Residual, cannot make predictions for evaluation.")
        else:
            for model_name, model_instance in models_dict.items():
                print(f"   Training {model_name} on processed training trips' LOSO features...")
                try:
                    current_model_for_fitting = model_instance.__class__(**model_instance.get_params())
                    if isinstance(current_model_for_fitting, (LinearRegression, Ridge, RandomForestRegressor)):
                        pipeline = Pipeline([('preprocessing', preprocessor_general), ('model', current_model_for_fitting)])
                        pipeline.fit(X_train_processed, y_train_all_loso)
                        if not X_test_processed_subset.empty:
                             predictions_on_test = pipeline.predict(X_test_processed_subset)
                        else: # No test data to predict on
                             predictions_on_test = np.array([]) # Empty array
                    elif isinstance(current_model_for_fitting, CatBoostRegressor):
                        X_train_cb = X_train_processed.copy()
                        X_test_cb = X_test_processed_subset.copy() # Use the subset for CatBoost as well
                        cb_cat_features_for_fit = [col for col in final_cat_cols_for_pipe if col in X_train_cb.columns]
                        for col in cb_cat_features_for_fit:
                            X_train_cb[col] = X_train_cb[col].astype(str)
                            if col in X_test_cb.columns: X_test_cb[col] = X_test_cb[col].astype(str)
                        current_model_for_fitting.fit(X_train_cb, y_train_all_loso, cat_features=cb_cat_features_for_fit if cb_cat_features_for_fit else None)
                        if not X_test_cb.empty:
                            predictions_on_test = current_model_for_fitting.predict(X_test_cb)
                        else:
                            predictions_on_test = np.array([])
                    else:
                        current_model_for_fitting.fit(X_train_processed, y_train_all_loso)
                        if not X_test_processed_subset.empty:
                            predictions_on_test = current_model_for_fitting.predict(X_test_processed_subset)
                        else:
                            predictions_on_test = np.array([])
                    
                    # Ensure predictions_on_test aligns with temp_eval_preds_storage.index
                    if len(predictions_on_test) == len(temp_eval_preds_storage.index):
                        temp_eval_preds_storage[f'pred_{model_name}'] = predictions_on_test
                    elif len(predictions_on_test) == 0 and len(temp_eval_preds_storage.index) > 0:
                         temp_eval_preds_storage[f'pred_{model_name}'] = np.nan # Fill with NaN if no predictions made
                    elif len(predictions_on_test) > 0 : # Mismatch
                         warnings.warn(f"Prediction length mismatch for {model_name}. Got {len(predictions_on_test)}, expected {len(temp_eval_preds_storage.index)}. Filling with NaN.")
                         temp_eval_preds_storage[f'pred_{model_name}'] = np.nan


                except Exception as e:
                    warnings.warn(f"Error training/predicting {model_name} for LOSO-Global-Residual (split): {e}")
                    traceback.print_exc()
                    temp_eval_preds_storage[f'pred_{model_name}'] = np.nan
        
        evaluation_set_predictions_df = temp_eval_preds_storage
        if RESIDUAL_TARGET_COL not in evaluation_set_predictions_df.columns and not y_test_all_actuals_series.empty:
            evaluation_set_predictions_df[RESIDUAL_TARGET_COL] = y_test_all_actuals_series
    else: raise ValueError(f"Unknown method: {method}.")

    print("\n--- Evaluating Model Performance on Residuals (on appropriate evaluation set) ---")
    evaluation_results = []
    eval_metrics_df = evaluation_set_predictions_df # This now contains predictions on the test set for LGR
    if eval_metrics_df.empty and method.lower() == 'loso_global_residual' and not y_test_all_actuals_series.empty :
        warnings.warn("Evaluation set predictions DataFrame is empty but test actuals exist. Metrics will be NaN.")
        # Create columns for pred_model_name with NaNs if they don't exist
        for model_name_key in models_dict.keys():
            if f'pred_{model_name_key}' not in eval_metrics_df.columns:
                eval_metrics_df[f'pred_{model_name_key}'] = np.nan
        if RESIDUAL_TARGET_COL not in eval_metrics_df.columns:
             eval_metrics_df[RESIDUAL_TARGET_COL] = y_test_all_actuals_series

    for model_name in models_dict.keys():
        pred_col = f'pred_{model_name}'
        if pred_col in eval_metrics_df.columns and RESIDUAL_TARGET_COL in eval_metrics_df.columns:
            # Ensure y_true and y_pred are aligned by index before comparison
            comparison_df = eval_metrics_df[[RESIDUAL_TARGET_COL, pred_col]].dropna()
            if not comparison_df.empty:
                 mae = mean_absolute_error(comparison_df[RESIDUAL_TARGET_COL], comparison_df[pred_col])
                 rmse = np.sqrt(mean_squared_error(comparison_df[RESIDUAL_TARGET_COL], comparison_df[pred_col]))
                 mbe = (comparison_df[pred_col] - comparison_df[RESIDUAL_TARGET_COL]).mean()
                 metrics = {'MAE': mae, 'RMSE': rmse, 'MBE': mbe, 'Predictions': len(comparison_df)}
                 print(f" - {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, MBE={mbe:.4f} (on {metrics['Predictions']} evaluation segments)")
                 evaluation_results.append({'Model': model_name, **metrics})
            else: evaluation_results.append({'Model': model_name, 'MAE': np.nan, 'RMSE': np.nan, 'MBE': np.nan, 'Predictions': 0})
        else: evaluation_results.append({'Model': model_name, 'MAE': np.nan, 'RMSE': np.nan, 'MBE': np.nan, 'Predictions': 0})
    eval_results_df = pd.DataFrame(evaluation_results).set_index('Model')

    # if method.lower() == 'loso' and feature_considered_counts_loso_per_trip:
    #     print("\n--- Feature Removal Statistics (LOSO Per-Trip NZV) ---")
    #     removal_ratios_data = []
    #     all_loso_features_pt = sorted(list(feature_considered_counts_loso_per_trip.keys()))
    #     for feature in all_loso_features_pt:
    #         considered_count = feature_considered_counts_loso_per_trip.get(feature, 0)
    #         removed_count = feature_removal_counts_loso_per_trip.get(feature, 0)
    #         ratio = (removed_count / considered_count) if considered_count > 0 else 0
    #         removal_ratios_data.append({'feature': feature, 'removed_count': removed_count, 'considered_count': considered_count, 'removal_ratio': ratio})
    #     removal_stats_df_pt = pd.DataFrame(removal_ratios_data)
    #     if not removal_stats_df_pt.empty: print(removal_stats_df_pt.sort_values(by='removal_ratio', ascending=False))
    #     else: print("No per-trip LOSO NZV stats generated.")

    if method.lower() == 'loso_global_residual' and global_loso_train_considered_features:
        print("\n--- Feature Removal Statistics (LOSO-Global-Residual on Training Data) ---")
        print(f" - Features initially considered for model (from FINAL_FEATURE_ORDER_LOSO): {len(global_loso_train_considered_features)}")
        print(f" - Numerical features removed by NZV (threshold={NZV_THRESHOLD}): {len(global_loso_train_removed_features_nZV)}")
        if global_loso_train_removed_features_nZV: print(f"   NZV Removed list : {global_loso_train_removed_features_nZV}")
        print(f" - Numerical features removed by Collinearity filter (threshold={collinearity_threshold}): {len(global_loso_train_removed_features_collinear)}")
        if global_loso_train_removed_features_collinear: print(f"   Collinearity Removed list: {global_loso_train_removed_features_collinear}")
    
    best_model_name = None
    if not eval_results_df.empty and primary_metric in eval_results_df.columns:
        valid_results = eval_results_df.dropna(subset=[primary_metric])
        if not valid_results.empty:
             if primary_metric in ['MAE', 'RMSE']: best_model_name = valid_results[primary_metric].idxmin()
             if best_model_name: print(f"\nSelected best model type: '{best_model_name}' (based on {primary_metric} on evaluation set)")
             else: print("\nWarning: Could not determine best model type.")
        else: print("\nWarning: No valid evaluation results to select best model type.")
    else: print("\nWarning: Cannot select best model type - evaluation failed or metric missing.")

    segments_df_out = segments_df.copy()
    predicted_residual_col = 'personal_model_pred_residual'
    final_pred_col = 'personal_pred_soc_seg_delta'
    segments_df_out[predicted_residual_col] = np.nan

    # --- NEW: For storing ALL features dataframe ---
    all_features_df_for_saving = pd.DataFrame()
    # --- NEW: For storing list of features used by the final model ---
    final_model_used_features_list = []

    # # --- Determine output directory for plots ---
    # # This assumes OUTPUT_FILE_PATH is defined globally in the script or passed appropriately
    # # For simplicity, let's derive it here if OUTPUT_FILE_PATH is the full path to the output data file
    # try:
    #     current_output_dir = os.path.dirname(OUTPUT_FILE_PATH) # Assuming OUTPUT_FILE_PATH is defined
    #     os.makedirs(current_output_dir, exist_ok=True)
    # except NameError:
    #     warnings.warn("OUTPUT_FILE_PATH not defined, feature importance plot might not save correctly.")
    #     current_output_dir = "." # Default to current directory

    if best_model_name and method.lower() == 'loso_global_residual':
        print(f"\n--- Generating final predictions for ALL segments using best model type: {best_model_name} (LOSO-Global-Residual Method) ---")
        best_model_archetype = models_dict[best_model_name]
        
        print("   Retraining best model type on PROCESSED LOSO features from ALL trips for final output...")
        # 1. Generate WIDE features for ALL trips
        all_X_loso_full_wide_list, all_y_loso_full_list, all_indices_full = [], [], []
        for trip_idx, cycle_df_orig in segments_df.groupby(TRIP_ID_COL, sort=False): # Keep track of trip_id for raw features
            cycle_df = cycle_df_orig.sort_values(SEGMENT_ID_COL)
            if len(cycle_df) <= min_train_samples: continue
            for i in range(len(cycle_df)):
                current_segment_row = cycle_df.iloc[i]
                other_segments_df = cycle_df.drop(index=cycle_df.index[i])
                if len(other_segments_df) < min_train_samples: continue
                
                X_seg_df_wide = build_loso_feature_vector(current_segment_row, other_segments_df)
                y_seg = current_segment_row.get(RESIDUAL_TARGET_COL)

                if X_seg_df_wide is not None and not pd.isna(y_seg):
                    # Store the WIDE engineered features
                    engineered_features_dict = X_seg_df_wide.iloc[0].to_dict()
                    all_X_loso_full_wide_list.append(engineered_features_dict)
                    all_y_loso_full_list.append(y_seg)
                    all_indices_full.append(cycle_df.index[i]) # Original index from segments_df

        if not all_X_loso_full_wide_list:
            warnings.warn("Could not generate WIDE LOSO features from any trip for final model retraining and ALL_FEATURES output.")
        else:
            X_all_loso_full_wide = pd.DataFrame(all_X_loso_full_wide_list, index=pd.Index(all_indices_full), columns=ALL_POSSIBLE_LOSO_FEATURES)
            y_all_loso_full = pd.Series(all_y_loso_full_list, index=pd.Index(all_indices_full))

            # --- Prepare data for "ALL FEATURES" output ---
            # Select raw features from the original segments_df for the rows where LOSO features were generated
            raw_features_for_output = segments_df.loc[X_all_loso_full_wide.index].copy()
            # Combine raw features with ALL engineered LOSO features
            all_features_df_for_saving = pd.concat([raw_features_for_output, X_all_loso_full_wide], axis=1)
            # Add target columns (gt_soc_seg_delta, global_pred_soc_seg_delta, residual_error_soc_seg_delta)
            # These are already in raw_features_for_output if they were in segments_df
            # We will add personal_model_pred_residual and personal_final_error later

            # 2. Select SUBSET for final model retraining
            print(f"   Selecting subset of {len(FINAL_FEATURE_ORDER_LOSO)} features for FINAL model training from {X_all_loso_full_wide.shape[1]} engineered features.")
            final_model_features_subset = [col for col in FINAL_FEATURE_ORDER_LOSO if col in X_all_loso_full_wide.columns]
            X_all_loso_full_subset = X_all_loso_full_wide[final_model_features_subset].copy()

            # Handle NaNs in the SUBSET for final retraining
            nan_rows_mask_final_train = X_all_loso_full_subset.isnull().any(axis=1)
            if nan_rows_mask_final_train.any():
                print(f"   Dropping {nan_rows_mask_final_train.sum()} rows for FINAL model training due to NaNs in selected features.")
                X_all_loso_full_subset = X_all_loso_full_subset[~nan_rows_mask_final_train]
                y_all_loso_full_for_model = y_all_loso_full[~nan_rows_mask_final_train] # Align target
            else:
                y_all_loso_full_for_model = y_all_loso_full.copy()
            
            if X_all_loso_full_subset.empty:
                 warnings.warn("No data remaining for FINAL model training after NaN handling in subset.")
            else:
                # 3. Apply Preprocessing (NZV and Collinearity, fitted on THIS full subset)
                X_all_loso_full_processed = X_all_loso_full_subset # Start with the subset
                num_cols_full_retrain = X_all_loso_full_processed.select_dtypes(include=np.number).columns.tolist()
                cat_cols_full_retrain = X_all_loso_full_processed.select_dtypes(include=['object', 'category']).columns.tolist()
                
                X_full_num_nZV_retrain = X_all_loso_full_processed[num_cols_full_retrain].copy()
                nZV_selector_final_retrain = None
                if num_cols_full_retrain and X_all_loso_full_processed[num_cols_full_retrain].shape[0] > 1:
                    nZV_selector_final_retrain = VarianceThreshold(threshold=NZV_THRESHOLD)
                    try:
                        nZV_selector_final_retrain.fit(X_all_loso_full_processed[num_cols_full_retrain])
                        selected_mask_final_nZV_retrain = nZV_selector_final_retrain.get_support()
                        X_full_num_nZV_retrain = X_all_loso_full_processed[num_cols_full_retrain].loc[:, selected_mask_final_nZV_retrain].copy()
                        removed_final_nZV = X_all_loso_full_processed[num_cols_full_retrain].columns[~selected_mask_final_nZV_retrain].tolist()
                        print(f"     For FINAL model retraining, NZV removed: {removed_final_nZV}")
                    except Exception as e_final_nzv: warnings.warn(f"Error during NZV for final model retraining: {e_final_nzv}.")
                
                X_full_num_final_retrain = X_full_num_nZV_retrain.copy()
                if not X_full_num_nZV_retrain.empty:
                    X_full_num_final_retrain, removed_coll_final = remove_highly_collinear_numerical_features(
                        X_full_num_nZV_retrain, threshold=collinearity_threshold
                    )
                    print(f"     For FINAL model retraining, collinearity removed: {removed_coll_final}")
                
                X_all_loso_full_processed = pd.concat([X_full_num_final_retrain, X_all_loso_full_processed[cat_cols_full_retrain].copy()], axis=1)
                
                # --- Store the features ACTUALLY USED by the model ---
                final_model_used_features_list = X_all_loso_full_processed.columns.tolist()
                print(f"   Final model will be trained on {len(final_model_used_features_list)} features: {final_model_used_features_list}")


                if X_all_loso_full_processed.empty or X_all_loso_full_processed.shape[1] == 0:
                    warnings.warn("No features remaining for retraining final LOSO-Global model.")
                else:
                    final_model_retrained = best_model_archetype.__class__(**best_model_archetype.get_params())
                    
                    fitted_model_for_importance = None
                    feature_names_for_importance_plot = None
                    
                    final_num_cols_for_final_pipe = X_all_loso_full_processed.select_dtypes(include=np.number).columns.tolist()
                    final_cat_cols_for_final_pipe = X_all_loso_full_processed.select_dtypes(include=['object', 'category']).columns.tolist()

                    if isinstance(final_model_retrained, (LinearRegression, Ridge, RandomForestRegressor)):
                        # ... (pipeline fitting logic as before, using X_all_loso_full_processed and y_all_loso_full_for_model)
                        transformers_final_pipe = []
                        if final_num_cols_for_final_pipe: transformers_final_pipe.append(('num', StandardScaler(), final_num_cols_for_final_pipe))
                        if final_cat_cols_for_final_pipe: transformers_final_pipe.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), final_cat_cols_for_final_pipe))
                        preprocessor_final_pipe = ColumnTransformer(transformers=transformers_final_pipe, remainder='passthrough' if not transformers_final_pipe else 'drop')
                        pipeline_final_retrained = Pipeline([('preprocessing', preprocessor_final_pipe), ('model', final_model_retrained)])
                        pipeline_final_retrained.fit(X_all_loso_full_processed, y_all_loso_full_for_model)
                        final_preds_array = pipeline_final_retrained.predict(X_all_loso_full_processed) # Predict on the same data it was trained on
                        
                        fitted_model_for_importance = pipeline_final_retrained.named_steps['model']
                        try: feature_names_for_importance_plot = pipeline_final_retrained.named_steps['preprocessing'].get_feature_names_out()
                        except: feature_names_for_importance_plot = X_all_loso_full_processed.columns.tolist()

                    elif isinstance(final_model_retrained, CatBoostRegressor):
                        # ... (CatBoost fitting logic as before)
                        X_full_cb_retrain = X_all_loso_full_processed.copy()
                        cb_cat_features_final_retrain = [col for col in final_cat_cols_for_final_pipe if col in X_full_cb_retrain.columns]
                        for col in cb_cat_features_final_retrain: X_full_cb_retrain[col] = X_full_cb_retrain[col].astype(str)
                        final_model_retrained.fit(X_full_cb_retrain, y_all_loso_full_for_model, cat_features=cb_cat_features_final_retrain if cb_cat_features_final_retrain else None)
                        final_preds_array = final_model_retrained.predict(X_full_cb_retrain)
                        fitted_model_for_importance = final_model_retrained
                        feature_names_for_importance_plot = X_full_cb_retrain.columns.tolist()
                    else: # Fallback
                        final_model_retrained.fit(X_all_loso_full_processed, y_all_loso_full_for_model)
                        final_preds_array = final_model_retrained.predict(X_all_loso_full_processed)
                        fitted_model_for_importance = final_model_retrained
                        feature_names_for_importance_plot = X_all_loso_full_processed.columns.tolist()
                    
                    # Assign predictions to the correct rows in segments_df_out
                    # The index of y_all_loso_full_for_model matches the rows used for training the final model
                    segments_df_out.loc[y_all_loso_full_for_model.index, predicted_residual_col] = final_preds_array
    
                    if fitted_model_for_importance and feature_names_for_importance_plot is not None:
                        current_output_dir = os.path.dirname(OUTPUT_FILE_PATH)
                        os.makedirs(current_output_dir, exist_ok=True)
                        display_and_save_feature_importances(
                            model=fitted_model_for_importance,
                            feature_names=feature_names_for_importance_plot,
                            model_name_label=f"{best_model_name}_LOSO_Global_Residual_Final",
                            output_dir=current_output_dir
                        )
    elif best_model_name and method.lower() in ['walk_forward', 'loso']: # For other methods
        if f'pred_{best_model_name}' in evaluation_set_predictions_df.columns:
            segments_df_out.loc[evaluation_set_predictions_df.index, predicted_residual_col] = evaluation_set_predictions_df[f'pred_{best_model_name}']
            # For these methods, 'all_features_df_for_saving' would need to be constructed differently
            # or this feature might be specific to loso_global_residual as requested.
            # For now, all_features_df_for_saving will only be populated for loso_global_residual.
            warnings.warn(f"Saving of ALL engineered features is currently implemented only for 'loso_global_residual'. For '{method}', this output will be empty.")
            # Similarly for final_model_used_features_list
            warnings.warn(f"Saving of USED model features is currently most meaningful for 'loso_global_residual'. For '{method}', this output will be empty.")


    else:
        print(" - Warning: Best model type not selected or method not 'loso_global_residual'. Final prediction columns will have NaNs. ALL FEATURES output will be empty.")

    segments_df_out[final_pred_col] = segments_df_out[GLOBAL_PRED_COL] + segments_df_out[predicted_residual_col]
    print(f"Predicted residual column created: '{predicted_residual_col}'")
    print(f"Final combined prediction column created: '{final_pred_col}'")

    # --- Add final error to the ALL FEATURES dataframe before saving it ---
    if not all_features_df_for_saving.empty:
        # Merge the final predictions into it, if they exist for those rows
        if predicted_residual_col in segments_df_out.columns:
            all_features_df_for_saving = all_features_df_for_saving.join(segments_df_out[[predicted_residual_col, final_pred_col]], how='left')
            # Calculate personal_final_error for this dataframe
            if GT_TARGET_COL in all_features_df_for_saving.columns and final_pred_col in all_features_df_for_saving.columns:
                 all_features_df_for_saving['personal_final_error'] = all_features_df_for_saving[GT_TARGET_COL] - all_features_df_for_saving[final_pred_col]
            else:
                 all_features_df_for_saving['personal_final_error'] = np.nan
        else:
            all_features_df_for_saving[predicted_residual_col] = np.nan
            all_features_df_for_saving[final_pred_col] = np.nan
            all_features_df_for_saving['personal_final_error'] = np.nan


    end_time = time.time()
    print(f"Evaluation and prediction complete. Time taken: {end_time - start_time:.2f} seconds.")
    # Return the main output df, eval results, the new all_features_df, and the used_features_list
    return segments_df_out, eval_results_df, all_features_df_for_saving, final_model_used_features_list

# --- Final Error Calculation Function (Vectorized) ---
def add_final_prediction_error(segments_df):
    """Calculates the error of the final combined personal prediction."""
    final_pred_col = 'personal_pred_soc_seg_delta'
    final_error_col = 'personal_final_error'
    if GT_TARGET_COL not in segments_df.columns: raise ValueError(f"Ground truth column '{GT_TARGET_COL}' not found.")
    if final_pred_col not in segments_df.columns:
        warnings.warn(f"Final prediction column '{final_pred_col}' not found. Final error will be NaN.")
        segments_df[final_error_col] = np.nan
        return segments_df
    segments_df[final_error_col] = segments_df[GT_TARGET_COL] - segments_df[final_pred_col]
    print(f"\n--- Calculating Final Prediction Errors ---")
    print(f"Calculated final personal error column: '{final_error_col}'")
    return segments_df


def main():
    """Loads features, evaluates multiple residual models via specified method, saves best predictions."""

    # --- Define Candidate Models for Residual Prediction ---
    MODELS_TO_EVALUATE = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'RandomForest': RandomForestRegressor(n_estimators=50, # Reduced estimators for speed
                                              random_state=RANDOM_STATE,
                                              n_jobs=-1,
                                              max_depth=8, # Reduced depth
                                              min_samples_leaf=5), # Added regularization
        'CatBoost': CatBoostRegressor(iterations=100, # Reduced iterations
                                      learning_rate=0.1,
                                      depth=6,
                                      l2_leaf_reg=3,
                                      loss_function='RMSE',
                                      eval_metric='MAE',
                                      random_seed=RANDOM_STATE,
                                      verbose=False)
    }
    # ----------------------------------------------------

    print(f"--- Starting Personal Residual Model Evaluation & Prediction ---")
    print(f"Method: {METHOD.upper()}")
    print(f"Minimum Training Samples Required: {MIN_TRAIN_SAMPLES}")
    print(f"Loading features data from: {FEATURE_FILE_PATH}")

    try:
        # 1. Load Data (Output of feature extraction with NEW names)
        segments_df = load_file(FEATURE_FILE_PATH)
        if not isinstance(segments_df, pd.DataFrame): raise TypeError("Loaded data is not a DataFrame.")
        print(f"Loaded data with {segments_df.shape[0]} rows and {segments_df.shape[1]} columns.")
        print(f"Columns found: {segments_df.columns.tolist()}") # Optional

        # Convert potential categorical columns in the loaded segments_df to 'category' dtype.
        # These are the base names as they appear in the Parquet file.
        # Prefixed versions (e.g., 'k_car_model_trip') are created later during feature vector construction.
        print("\n--- Ensuring correct dtype for base categorical columns ---")
        base_categorical_columns_to_convert = ['car_model_trip', 'manufacturer_trip', 'battery_type_trip']
        for col_name in base_categorical_columns_to_convert:
            if col_name in segments_df.columns:
                segments_df[col_name] = segments_df[col_name].astype('category')
                print(f"  Converted column '{col_name}' to category dtype.")
            else:
                warnings.warn(f"  Base categorical column '{col_name}' not found in loaded segments_df.")

        # 2. Data Checks (Verify columns with NEW names needed for this script exist)
        print("\n--- Performing Initial Data Checks ---")
        # Check essential columns needed for the process
        required_load_cols = [TRIP_ID_COL, SEGMENT_ID_COL, GT_TARGET_COL, GLOBAL_PRED_COL, RESIDUAL_TARGET_COL]
        required_load_cols.extend(SEGMENT_OUTCOME_FEATURES) # Includes driving behavior now
        required_load_cols.extend(PREDICTED_SEGMENT_FEATURES)
        required_load_cols.extend(START_OF_SEGMENT_FEATURES)
        required_load_cols = sorted(list(set(required_load_cols)))

        missing_cols = [col for col in required_load_cols if col not in segments_df.columns]
        if missing_cols:
             # Provide more context on missing columns
             print("\n!!! ERROR: Input DataFrame is missing required columns !!!")
             print(f"Missing: {missing_cols}")
             print("\nExpected columns based on configuration:")
             print(f"  Identifiers: {TRIP_ID_COL}, {SEGMENT_ID_COL}")
             print(f"  Targets/Preds: {GT_TARGET_COL}, {GLOBAL_PRED_COL}, {RESIDUAL_TARGET_COL}")
             print(f"  Segment Outcomes: {SEGMENT_OUTCOME_FEATURES}")
             print(f"  Predicted Segment: {PREDICTED_SEGMENT_FEATURES}")
             print(f"  Start of Segment/Static: {START_OF_SEGMENT_FEATURES}")
             raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")
        print("Initial data checks passed.")

        # 3. Evaluate Models and Generate Predictions from Best
        segments_df_with_preds, eval_results_df, all_features_output_df, used_features_list = evaluate_and_predict_residuals(
            segments_df,
            models_dict=MODELS_TO_EVALUATE,
            method=METHOD,
            primary_metric=PRIMARY_METRIC,
            min_train_samples=MIN_TRAIN_SAMPLES,
            trip_level_split_test_size=TRIP_SPLIT_TEST_SIZE_FOR_LGR,
            random_state_split=RANDOM_STATE,
            collinearity_threshold=0.90,
            NZV_THRESHOLD=NZV_THRESHOLD
        )
        print(f"\n--- Evaluation Summary (on Residuals using {METHOD}) ---")
        print(eval_results_df)

        # --- Report Predictions (using segments_df_with_preds) ---
        final_pred_col_name = 'personal_pred_soc_seg_delta'
        predicted_residual_col = 'personal_model_pred_residual'

        if predicted_residual_col in segments_df_with_preds.columns and final_pred_col_name in segments_df_with_preds.columns:
            non_nan_residuals = segments_df_with_preds[predicted_residual_col].notna().sum()
            non_nan_final = segments_df_with_preds[final_pred_col_name].notna().sum()
            
            # Calculate total_possible based on the original segments_df and method constraints
            if METHOD.lower() == 'loso' or METHOD.lower() == 'loso_global_residual':
                trip_segment_counts = segments_df.groupby(TRIP_ID_COL)[SEGMENT_ID_COL].transform('count')
                possible_pred_mask = (trip_segment_counts - 1) >= MIN_TRAIN_SAMPLES
                total_possible = possible_pred_mask.sum()
            elif METHOD.lower() == 'walk_forward':
                possible_pred_mask = segments_df[SEGMENT_ID_COL] > MIN_TRAIN_SAMPLES
                total_possible = possible_pred_mask.sum()
            else:
                warnings.warn(f"Unknown METHOD '{METHOD}' for calculating total_possible.")
                total_possible = len(segments_df)

            print(f"\n--- Prediction Summary (Using Best Model for method: {METHOD.upper()}) ---")
            print(f"Total number of residual predictions generated: {non_nan_residuals} / {total_possible} possible.")
            print(f"Total number of final combined predictions generated: {non_nan_final} / {total_possible} possible.")
        else:
            print(f"\nFinal prediction columns ('{predicted_residual_col}' or '{final_pred_col_name}') not found in output.")

        # --- Calculate Final Error (using segments_df_with_preds) ---
        segments_df_final_output = add_final_prediction_error(segments_df_with_preds)

           # --- Save Main Results ---
        print(f"\n--- Saving Final Predictions DataFrame ---")
        output_dir = os.path.dirname(OUTPUT_FILE_PATH)
        main_base_file_name = f"{os.path.splitext(os.path.basename(OUTPUT_FILE_PATH))[0]}_{METHOD}"
        save_file(data=segments_df_final_output, path=output_dir, file_name=main_base_file_name, format=OUTPUT_FORMAT, index=False)

        eval_file_name = f"personal_model_evaluation_{METHOD}.csv"
        eval_results_df.to_csv(os.path.join(output_dir, eval_file_name))
        print(f"Saved model evaluation summary to: {os.path.join(output_dir, eval_file_name)}")

        # --- NEW: Save ALL Engineered Features ---
        if not all_features_output_df.empty and METHOD.lower() == 'loso_global_residual':
            print(f"\n--- Saving ALL Engineered Features DataFrame ---")
            all_feat_output_dir = os.path.dirname(ALL_FEATURES_OUTPUT_PATH)
            all_feat_base_name = os.path.splitext(os.path.basename(ALL_FEATURES_OUTPUT_PATH))[0]
            # Ensure the directory exists
            os.makedirs(all_feat_output_dir, exist_ok=True)
            save_file(data=all_features_output_df, path=all_feat_output_dir, file_name=all_feat_base_name, format='parquet', index=True) # Save index as it's from original df
        elif METHOD.lower() == 'loso_global_residual':
            print("\nNote: ALL Engineered Features DataFrame was empty, not saved.")


        # --- NEW: Save USED Model Features List ---
        if used_features_list and METHOD.lower() == 'loso_global_residual':
            print(f"\n--- Saving List of Features Used by the Final Model ---")
            used_feat_output_dir = os.path.dirname(USED_MODEL_FEATURES_PATH)
            os.makedirs(used_feat_output_dir, exist_ok=True)
            try:
                with open(USED_MODEL_FEATURES_PATH, 'w') as f:
                    for feature_name in used_features_list:
                        f.write(f"{feature_name}\n")
                print(f"Saved used model features list to: {USED_MODEL_FEATURES_PATH}")
            except Exception as e:
                warnings.warn(f"Could not save used model features list: {e}")
        elif METHOD.lower() == 'loso_global_residual':
             print("\nNote: List of used model features was empty, not saved.")


        print(f"\n--- Pipeline Complete ---")

    except (FileNotFoundError, TypeError, ValueError, KeyError, ImportError, Exception) as e:
        print(f"\n--- Pipeline Failed ---")
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
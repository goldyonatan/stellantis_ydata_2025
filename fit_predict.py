import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge # Example models
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor # Import CatBoost
from sklearn.metrics import mean_absolute_error, mean_squared_error # For evaluation
from sklearn.exceptions import NotFittedError
import joblib # Keep for potentially saving the personal models if needed later
import warnings
import time
import traceback
from collections import OrderedDict # To ensure consistent feature order

# --- Import necessary functions from HelperFuncs ---
try:
    from HelperFuncs import load_file, save_file # Assuming these exist
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
    #'matched_route_distance_m', 'matched_route_duration_s', 
    'direct_route_distance_m',
    'route_deviation_pct', 'percent_time_over_limit_seg_agg', 'avg_speed_limit_kph_seg_agg',
    #'match_confidence',
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
    'weight_trip_kg','car_model_trip', 'manufacturer_trip', 'battery_type_trip'
    #'soc_trip_start', 'odo_trip_start'
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
    'k_soc_seg_start', 'k_odo_seg_start', 'k_hour_seg_start', #'k_dayofweek_seg_start', 'k_month_seg_start', 
    'k_temp_seg_start', 'k_battery_health_seg_start',
    # 'k_weight_trip_kg', 'k_car_model_trip', 'k_manufacturer_trip', 'k_battery_type_trip', # Categorical

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
    #'others_avg_stop_duration_seg_agg_s',
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
    Constructs the feature vector for predicting segment k using LOSO method.
    Features include start/pred/static/global_pred for k, and aggregates over other segments.
    """
    features = OrderedDict()
    segment_id_debug = current_segment_row.get(SEGMENT_ID_COL, 'N/A')
    feature_order = FINAL_FEATURE_ORDER_LOSO # Use LOSO order

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
        if k_col_name in feature_order:
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
        if k_col_name in feature_order:
            features[k_col_name] = current_segment_row.get(col_base, np.nan)

    # --- 3. Global Prediction for Segment k --- (Same as walk-forward)
    # Use 'k_' prefix and NEW descriptive name from current_segment_row
    if 'k_global_pred_soc_seg_delta' in feature_order:
        features['k_global_pred_soc_seg_delta'] = current_segment_row.get('global_pred_soc_seg_delta', np.nan)

    # --- 4. Aggregates over OTHER Segments (j != k) ---
    # *** Calculate NEW driving/routing behavior aggregates ***
    features['others_segment_count'] = len(other_segments_df)
    if not other_segments_df.empty:

        others_total_dist_km = other_segments_df['distance_seg_actual_km'].sum()
        others_total_dur_s = other_segments_df['duration_seg_actual_s'].sum()
        others_total_dur_min = others_total_dur_s / 60.0

        if 'others_total_distance_actual_km' in feature_order: features['others_total_distance_actual_km'] = others_total_dist_km
        if 'others_total_duration_actual_s' in feature_order: features['others_total_duration_actual_s'] = others_total_dur_s

        # Calculate averages and stds for all outcome features
        for col_base in SEGMENT_OUTCOME_FEATURES:
            # Averages
            avg_feature_name = f'others_avg_{col_base.replace(RESIDUAL_TARGET_COL, "residual_error_soc_seg_delta")}'
            if avg_feature_name in feature_order:
                features[avg_feature_name] = other_segments_df[col_base].mean()

            # Standard Deviations
            std_feature_name = f'others_std_{col_base.replace(RESIDUAL_TARGET_COL, "residual_error_soc_seg_delta")}'
            if std_feature_name in feature_order:
                features[std_feature_name] = other_segments_df[col_base].std() 

            # Totals (for specific count-like features)
            total_feature_name = f'others_total_{col_base}'
            if col_base in ['stops_seg_count', 'accel_high_event_count', 'decel_high_event_count', 'stop_duration_seg_agg_s'] and total_feature_name in feature_order:
                features[total_feature_name] = other_segments_df[col_base].sum()  

        # Special MAE for residual error
        if 'others_abs_avg_residual_error_soc_seg_delta' in feature_order:
            features['others_abs_avg_residual_error_soc_seg_delta'] = other_segments_df[RESIDUAL_TARGET_COL].abs().mean() 

        # Calculate Rates
        others_total_stops = other_segments_df['stops_seg_count'].sum()
        others_total_accel_high = other_segments_df['accel_high_event_count'].sum()
        others_total_decel_high = other_segments_df['decel_high_event_count'].sum()
        others_total_stop_duration = other_segments_df['stop_duration_seg_agg_s'].sum()

        if 'others_rate_stops_per_km' in feature_order:
            features['others_rate_stops_per_km'] = (others_total_stops / others_total_dist_km) if others_total_dist_km > 1e-3 else 0.0
        if 'others_rate_stops_per_minute' in feature_order:
            features['others_rate_stops_per_minute'] = (others_total_stops / others_total_dur_min) if others_total_dur_min > 1e-3 else 0.0
        if 'others_rate_accel_high_events_per_km' in feature_order:
            features['others_rate_accel_high_events_per_km'] = (others_total_accel_high / others_total_dist_km) if others_total_dist_km > 1e-3 else 0.0
        if 'others_rate_accel_high_events_per_minute' in feature_order:
            features['others_rate_accel_high_events_per_minute'] = (others_total_accel_high / others_total_dur_min) if others_total_dur_min > 1e-3 else 0.0
        if 'others_rate_decel_high_events_per_km' in feature_order:
            features['others_rate_decel_high_events_per_km'] = (others_total_decel_high / others_total_dist_km) if others_total_dist_km > 1e-3 else 0.0
        if 'others_rate_decel_high_events_per_minute' in feature_order:
            features['others_rate_decel_high_events_per_minute'] = (others_total_decel_high / others_total_dur_min) if others_total_dur_min > 1e-3 else 0.0
        if 'others_ratio_stop_duration_to_total_duration' in feature_order:
            features['others_ratio_stop_duration_to_total_duration'] = (others_total_stop_duration / others_total_dur_s) if others_total_dur_s > 1e-3 else 0.0        
    
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
        for key in feature_order:
            if key.startswith('others_'):
                if 'count' in key or 'total' in key: features[key] = 0.0
                else: features[key] = np.nan  
    
    # --- Ensure consistent order and create DataFrame ---
    # Use the globally defined LOSO_FEATURE_ORDER list
    final_features_dict = {key: features.get(key, np.nan) for key in feature_order}
    feature_vector_df = pd.DataFrame([final_features_dict], columns=feature_order)

    if feature_vector_df.isnull().values.any():
         # nan_cols = feature_vector_df.columns[feature_vector_df.isnull().any()].tolist()
         # warnings.warn(f"DEBUG LOSO: NaNs found in feature vector for segment {segment_id_debug}. NaN columns: {nan_cols}")
         return None
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
        X_train_list = []
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
        y_train = pd.Series(y_train_aligned_list) # y_train is now aligned with X_train

        # # Rebuild y_train based on successful X_train generation
        # y_train_final_list = []
        # successful_train_indices = X_train.index # Assuming DataFrame kept index from list of dicts
        # for idx in successful_train_indices:
        #      y_train_final_list.append(cycle_df.loc[idx, RESIDUAL_TARGET_COL])
        # y_train = pd.Series(y_train_final_list, index=successful_train_indices)

        # --- Fit and Predict for EACH model ---
        for model_name, model_instance in models_dict.items():
            try:
                current_model = model_instance.__class__(**model_instance.get_params())
                current_model.fit(X_train, y_train)
                residual_prediction = current_model.predict(X_pred_df)
                predictions_df.loc[current_segment_index, f'pred_{model_name}'] = residual_prediction[0]
            except ValueError as ve: warnings.warn(f"LOSO ValueError fit/predict {model_name} for {trip_id}-{current_segment_k}: {ve}.")
            except Exception as e: warnings.warn(f"LOSO Error fit/predict {model_name} for {trip_id}-{current_segment_k}: {e}.")

    return predictions_df

# --- MODIFIED Main Orchestration Function ---
def evaluate_and_predict_residuals(segments_df, models_dict, method='walk_forward', primary_metric='MAE', min_train_samples=2):
    """
    Orchestrates walk-forward OR LOSO evaluation of multiple residual models,
    selects the best, and calculates final predictions using the best model.

    Args:
        segments_df (pd.DataFrame): Input DataFrame from feature extraction.
        models_dict (dict): Dictionary of model instances {'name': model_obj}.
        method (str): Evaluation method ('walk_forward' or 'loso').
        primary_metric (str): Metric ('MAE', 'RMSE') to select best model.
        min_train_samples (int): Minimum history/other segments needed for training.

    Returns:
        pd.DataFrame: DataFrame with predictions/errors from the best model.
        pd.DataFrame: DataFrame with evaluation metrics for all tested models.
    """
    start_time = time.time()
    print(f"\n--- Starting {method.upper()} Evaluation & Prediction ---")

    # Verify necessary input columns exist (using NEW names)
    #required_cols = [TRIP_ID_COL, SEGMENT_ID_COL, GT_TARGET_COL, GLOBAL_PRED_COL, RESIDUAL_TARGET_COL] # Start features needed by build_feature_vector
    # Add outcome features needed by build_feature_vector for history
    # required_cols.extend([
    #     'distance_seg_actual_km', 'duration_seg_actual_s', 'speed_seg_agg_mean',
    #     'speed_seg_agg_std', 'speed_seg_agg_max', 'speed_seg_agg_min', 'altitude_seg_agg_mean',
    #     'altitude_seg_agg_delta', 'altitude_seg_agg_std', 'temp_seg_agg_mean', 'temp_seg_agg_std',
    #     'temp_seg_agg_max', 'temp_seg_agg_min'
    # ])
    # # Add predicted features needed by build_feature_vector
    # required_cols.extend([
    #     'distance_seg_pred_m', 'duration_seg_pred_s', 'speed_seg_pred_kph',
    #     # 'elevation_gain_seg_pred', 'elevation_loss_seg_pred'
    # ])
    # # Add static features needed by build_feature_vector
    # required_cols.extend(['weight_trip_kg']) # Examples
    required_cols = [TRIP_ID_COL, SEGMENT_ID_COL, GT_TARGET_COL, GLOBAL_PRED_COL, RESIDUAL_TARGET_COL]
    required_cols.extend(SEGMENT_OUTCOME_FEATURES); required_cols.extend(PREDICTED_SEGMENT_FEATURES)
    required_cols.extend(START_OF_SEGMENT_FEATURES); required_cols = sorted(list(set(required_cols)))
    required_cols = sorted(list(set(required_cols))) # Unique and sorted

    missing_cols = [col for col in required_cols if col not in segments_df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns for dynamic feature building: {missing_cols}")

    num_cycles = segments_df[TRIP_ID_COL].nunique()
    print(f"Processing {num_cycles} unique trips using {method} method...")

        # --- Get Predictions for ALL Models using Apply ---
    if method.lower() == 'walk_forward':
        apply_func = apply_walk_forward_evaluation
        print(f" - Applying Walk-Forward (min_train_samples={min_train_samples})...")
    elif method.lower() == 'loso':
        apply_func = apply_loso_evaluation
        print(f" - Applying Leave-One-Segment-Out (min_train_samples={min_train_samples})...")
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'walk_forward' or 'loso'.")

    # --- Get Predictions for ALL Models using Apply ---
    all_predictions_df = segments_df.groupby(TRIP_ID_COL, group_keys=False).apply(
        apply_func,
        models_dict=models_dict,
        min_train_samples=min_train_samples
    )

    # --- Evaluate Models ---
    print("\n--- Evaluating Model Performance on Residuals ---")
    evaluation_results = []
    # Merge predictions back to original df to get the actual residual target
    eval_temp_df = segments_df[[RESIDUAL_TARGET_COL]].merge(
        all_predictions_df, left_index=True, right_index=True, how='inner'
    )

    for model_name in models_dict.keys():
        pred_col = f'pred_{model_name}'
        if pred_col in eval_temp_df.columns:
            # Calculate metrics comparing predicted residual to actual residual
            # Need to align and drop NaNs for fair comparison
            comparison_df = eval_temp_df[[RESIDUAL_TARGET_COL, pred_col]].dropna()
            if not comparison_df.empty:
                 # Metrics calculated on the error of the residual prediction itself
                 # MAE(actual_residual, predicted_residual)
                 mae = mean_absolute_error(comparison_df[RESIDUAL_TARGET_COL], comparison_df[pred_col])
                 rmse = np.sqrt(mean_squared_error(comparison_df[RESIDUAL_TARGET_COL], comparison_df[pred_col]))
                 # MBE (bias in residual prediction)
                 mbe = (comparison_df[pred_col] - comparison_df[RESIDUAL_TARGET_COL]).mean()
                 metrics = {'MAE': mae, 'RMSE': rmse, 'MBE': mbe, 'Predictions': comparison_df[pred_col].count()}
                 print(f" - {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, MBE={mbe:.4f} (on {metrics['Predictions']} segments)")
                 evaluation_results.append({'Model': model_name, **metrics})
            else:
                 print(f" - {model_name}: No non-NaN predictions available for evaluation.")
                 evaluation_results.append({'Model': model_name, 'MAE': np.nan, 'RMSE': np.nan, 'MBE': np.nan, 'Predictions': 0})
        else:
             print(f" - {model_name}: Prediction column not found.")
             evaluation_results.append({'Model': model_name, 'MAE': np.nan, 'RMSE': np.nan, 'MBE': np.nan, 'Predictions': 0})

    eval_results_df = pd.DataFrame(evaluation_results).set_index('Model')

    # --- Select Best Model ---
    best_model_name = None
    if not eval_results_df.empty and primary_metric in eval_results_df.columns:
        valid_results = eval_results_df.dropna(subset=[primary_metric])
        if not valid_results.empty:
             if primary_metric in ['MAE', 'RMSE']:
                 best_model_name = valid_results[primary_metric].idxmin()
             # Add R2 if needed, but MAE/RMSE more common for error eval
             # elif primary_metric == 'R2': best_model_name = valid_results[primary_metric].idxmax()
             if best_model_name:
                  print(f"\nSelected best personal model: '{best_model_name}' (based on lowest {primary_metric} on residuals)")
             else: print("\nWarning: Could not determine best model based on primary metric.")
        else: print("\nWarning: No valid evaluation results to select best model.")
    else: print("\nWarning: Cannot select best model - evaluation failed or metric missing.")

    # --- Add Predictions from BEST Model Only ---
    predicted_residual_col = 'personal_model_pred_residual'
    final_pred_col = 'personal_pred_soc_seg_delta' # Use NEW base name

    if best_model_name and f'pred_{best_model_name}' in all_predictions_df.columns:
        print(f" - Using predictions from '{best_model_name}' for final output.")
        segments_df[predicted_residual_col] = all_predictions_df[f'pred_{best_model_name}']
    else:
        print(" - Warning: Best model not selected or predictions missing. Final prediction columns will have NaNs.")
        segments_df[predicted_residual_col] = np.nan

    segments_df[final_pred_col] = segments_df[GLOBAL_PRED_COL] + segments_df[predicted_residual_col]
    print(f"Predicted residual column created: '{predicted_residual_col}'")
    print(f"Final combined prediction column created: '{final_pred_col}'")

    end_time = time.time()
    print(f"Evaluation and prediction complete. Time taken: {end_time - start_time:.2f} seconds.")
    return segments_df, eval_results_df # Return both results

# --- Final Error Calculation Function (Vectorized) ---
def add_final_prediction_error(segments_df):
    """Calculates the error of the final combined personal prediction."""
    # Use NEW descriptive names
    final_pred_col = 'personal_pred_soc_seg_delta'
    final_error_col = 'personal_final_error'
    if GT_TARGET_COL not in segments_df.columns: raise ValueError(f"Ground truth column '{GT_TARGET_COL}' not found.")
    if final_pred_col not in segments_df.columns: raise ValueError(f"Final prediction column '{final_pred_col}' not found.")
    segments_df[final_error_col] = segments_df[GT_TARGET_COL] - segments_df[final_pred_col]
    print(f"\n--- Calculating Final Prediction Errors ---")
    print(f"Calculated final personal error column: '{final_error_col}'")
    return segments_df


def main():
    """Loads features, evaluates multiple residual models via specified method, saves best predictions."""
    # --- Configuration ---
    # Use the output from the feature extraction script that includes routing predictions AND new names
    FEATURE_FILE_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segments_df\segments_features_v4_behavior.parquet" # Use output with driving behavior    
    OUTPUT_FILE_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segments_df\segments_final_predictions_v3" # New output name
    OUTPUT_FORMAT = 'parquet'
    MIN_TRAIN_SAMPLES = 3 # Minimum number of OTHER segments for LOSO, or history for WF
    PRIMARY_METRIC = 'MAE' # Metric to choose best personal model
    RANDOM_STATE = 42 # For models that use it
    # *** CHOOSE EVALUATION METHOD ***
    METHOD = 'loso' # Options: 'walk_forward', 'loso'
    # --------------------

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
        segments_df, eval_results_df = evaluate_and_predict_residuals(
            segments_df,
            models_dict=MODELS_TO_EVALUATE,
            method=METHOD, # Pass the chosen method
            primary_metric=PRIMARY_METRIC,
            min_train_samples=MIN_TRAIN_SAMPLES
        )
        print(f"\n--- Walk-Forward/LOSO Evaluation Summary (on Residuals using {METHOD}) ---")
        print(eval_results_df)

        # 4. Report Predictions (based on best model)
        final_pred_col_name = 'personal_pred_soc_seg_delta'
        predicted_residual_col = 'personal_model_pred_residual'
        if final_pred_col_name in segments_df.columns:
            non_nan_residuals = segments_df[predicted_residual_col].notna().sum()
            non_nan_final = segments_df[final_pred_col_name].notna().sum()
            if METHOD.lower() == 'loso':
                trip_segment_counts = segments_df.groupby(TRIP_ID_COL)[SEGMENT_ID_COL].transform('count')
                # A segment k is predictable if its trip has enough other segments for training
                possible_pred_mask = (trip_segment_counts - 1) >= MIN_TRAIN_SAMPLES
                total_possible = possible_pred_mask.sum()
            else: # For walk_forward
                possible_pred_mask = segments_df[SEGMENT_ID_COL] > MIN_TRAIN_SAMPLES
                total_possible = possible_pred_mask.sum()
            print(f"\n--- Prediction Summary (Using Best Model) ---")
            print(f"Total number of residual predictions generated: {non_nan_residuals} / {total_possible} possible.")
            print(f"Total number of final combined predictions generated: {non_nan_final} / {total_possible} possible.")
        else: print("\nFinal prediction column was not generated.")

        # 5. Calculate Final Error (based on best model's predictions)
        segments_df = add_final_prediction_error(segments_df)

        # 6. Save Results
        print(f"\n--- Saving Final Predictions DataFrame ---")
        output_dir = os.path.dirname(OUTPUT_FILE_PATH)
        base_file_name = f"{os.path.basename(OUTPUT_FILE_PATH)}_{METHOD}" # Add method to filename
        save_file(data=segments_df, path=output_dir, file_name=base_file_name, format=OUTPUT_FORMAT, index=False)

        # Save evaluation results
        eval_file_name = f"personal_model_evaluation_{METHOD}.csv"
        eval_results_df.to_csv(os.path.join(output_dir, eval_file_name))
        print(f"Saved model evaluation summary to: {os.path.join(output_dir, eval_file_name)}")

        print(f"\n--- Pipeline Complete ---")

    except (FileNotFoundError, TypeError, ValueError, KeyError, ImportError, Exception) as e:
        print(f"\n--- Pipeline Failed ---")
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
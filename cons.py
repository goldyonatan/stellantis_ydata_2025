# contants and definitions


# Preprocess.py
# --- Configuration Block (Moved outside main for better visibility) ---
INPUT_FILE_PATH = r'.\df_sample\df_sample.parquet'
CLEANED_DATA_PATH = r'.\clean_df\clean_df.parquet'
# *** Path to your land shapefile ***
#LAND_SHAPEFILE_PATH = r".\ne_10m_land.shp"

# --- Preprocessing Thresholds ---
MIN_TRIP_KM = 0.5 # Increased slightly from 0 to avoid noise trips
MAX_TRIP_KM = 500.0 # max length
MAX_GPS_JUMP_KM = 5.0 # Max distance between consecutive GPS points
MAX_CONSECUTIVE_SPEED_DIFF = 100 # Max diff between mean speed of consecutive records (kph)
MAX_CONSECUTIVE_ALT_DIFF = 500 # Max altitude diff between consecutive records (meters)
MAX_CONSECUTIVE_SOC_DIFF = 10 # Max SoC % diff between consecutive records
MAX_CONSECUTIVE_TEMP_DIFF = 5 # Max Temp C diff between consecutive records
MAX_CONSECUTIVE_TIME_GAP_SECONDS = 110 # Max seconds between consecutive records (110 sec)
MIN_VALID_SOC = 0
MAX_VALID_SOC = 100
MIN_VALID_TEMP = -20
MAX_VALID_TEMP = 50
MIN_VALID_ALTITUDE = -500 # Allow slightly below sea level
MAX_VALID_ALTITUDE = 10000 # Reasonable max altitude
MIN_VALID_BATT_HEALTH = 50 # SOH %
MAX_VALID_BATT_HEALTH = 130 # SOH %
MAX_VALID_ODO = 1000000 # Max odometer reading (km)
MAX_VALID_WEIGHT = 5000 # Max empty weight (kg)
MIN_VALID_WEIGHT = 500 # Min empty weight (kg)
SPEED_DIST_TOLERANCE_FACTOR = 3.5 # Allow distance covered to be X times speed*time (for accel/decel)
SOC_INCREASE_THRESHOLD = 0.5 # Allow small SoC increases (regen, noise)

# --- Preprocessing Parameters ---
FILTER_ZERO_COORDINATES = True # Filter individual (0,0) points?
FILTER_MOSTLY_ZERO_TRIPS = True # Filter trips with >75% (0,0) points?
ZERO_TRIP_THRESHOLD = 0.75 # Threshold for filtering trips

MAX_TEMP_DIFFERENCE_C = 5 # Max allowed diff between vehicle temp and weather temp
FETCH_WEATHER_DATA = False # Set to False to skip weather fetching/comparison
PROCESS_SUBSET = None # Set to an integer (e.g., 5000) to process only the first N rows for testing

# MAX_WEATHER_STATION_DIST_KM = 35 # Max distance to nearest station for weather data to be considered reliable

COLUMN_RENAME_MAP = {
    'FAMILY_LABEL': 'car_model','COMMERCIAL_BRAND_COMPONENT_TYP': 'manufacturer',
    'TYPE_OF_TRACTION_CHAIN': 'traction_type','EMPTY_WEIGHT_KG': 'empty_weight_kg',
    'DESTINATION_COUNTRY_CODE': 'destination_country_code','BATTERY_TYPE': 'battery_type',
    'LEV_BATT_CAPC_HEAL': 'battery_health','CYCLE_ID': 'trip_id',
    'MESS_ID_START': 'message_session_id','DATETIME_START': 'cycle_datetime_start',
    'DATETIME_END': 'cycle_datetime_end','dt': 'cycle_date',
    'SOC_START': 'cycle_soc_start','SOC_END': 'cycle_soc_end',
    'ODO_START': 'cycle_odo_start','ODO_END': 'cycle_odo_end',
    'geoindex_10_start': 'cycle_location_start','geoindex_10_end': 'cycle_location_end',
    'HEAD_COLL_TIMS': 'timestamp','PERD_VEHC_LIFT_MILG': 'current_odo',
    'LEV_BATT_LOAD_LEVL': 'current_soc','geoindex_10': 'current_location',# This is the H3 index column
    'latitude': 'latitude', # Include even if not present in input, will be created
    'longitude': 'longitude', # Include even if not present in input, will be created
    'PERD_VEHC_OUTS_TEMP': 'outside_temp','PERD_ALTT': 'altitude_array',
    'PERD_VEHC_LONG_SPD': 'speed_array',
}

# --- Standard Column Names (used internally after renaming) ---
TRIP_ID_COL = 'trip_id'
TIME_COL = 'timestamp'
ODO_COL = 'current_odo'
SOC_COL = 'current_soc'
H3_COL = 'current_location' # Standard name for the H3 index column
LAT_COL = 'latitude'       # Standard name for the latitude column TO BE CREATED
LON_COL = 'longitude'      # Standard name for the longitude column TO BE CREATED
SPEED_ARRAY_COL = 'speed_array'
ALT_ARRAY_COL = 'altitude_array'
VEHICLE_TEMP_COL = 'outside_temp' # Standard name for vehicle temp
BATT_HEALTH_COL = 'battery_health'
WEIGHT_COL = 'empty_weight_kg'
MSG_SESSION_ID_COL = 'message_session_id'

# constant columns for trip data
STATIC_TRIP_COLS = [
    'car_model', 'manufacturer', 'traction_type', 'empty_weight_kg',
    'destination_country_code', 'battery_type',
    'message_session_id', 'cycle_datetime_start', 'cycle_datetime_end',
    'cycle_date', 'cycle_soc_start', 'cycle_soc_end', 'cycle_odo_start',
    'cycle_odo_end', 'cycle_location_start', 'cycle_location_end'
]

# Columns for duplicate row check
DUPLICATE_CHECK_COLS_STD = [TRIP_ID_COL, TIME_COL]

# Helper functions:
# ------------------
GEO_FILE_PATH = r'110m_cultural\ne_110m_admin_0_countries.shp'

# Trip segmentation:
 # --- Configuration ---

SEGMENTATION_DICT_DIR = r".\segmentation_dict"
OUTPUT_FORMAT_FOR_SEGMENTATION = 'pickle'
SEGMENTATION_DICT_FILENAME = 'cycles_seg_dict_v2' + '.' + OUTPUT_FORMAT_FOR_SEGMENTATION


# Parameters for stop finding and segmentation
STOP_SPEED_KPH = 1.5 # Speed threshold for stops
STOP_DIST_KM = 0.05 # Odo distance threshold for stops
MIN_SEGMENT_POINTS = 10 # Minimum number of non-stop data points for a valid segment
CONSECUTIVE_STOPS_FOR_BREAK = 3 # Number of consecutive stop points to end a driving block
# --------------------

# Feature extraction
# --- Configuration Block ---
GLOBAL_MODEL_PATH = r".\global_baseline_model.joblib"
OUTPUT_FEATURE_PATH = r".\segments_df\segments_features_v4_behavior.parquet" # Updated name
OSRM_URL = "http://router.project-osrm.org" # Public instance - use responsibly! Or set up local instance.

# --- Feature Extraction Parameters ---
MIN_SEGMENTS_PER_TRIP = 2
OUTPUT_FORMAT_FOR_FEAT_EXTRACTION = 'parquet'

FETCH_ROUTING_FEATURES = True # For predicted route (start->end)
#FETCH_MAP_MATCHING = True # For matched route comparison
#PROCESS_SUBSET = None # To process only x samples
# *** New Flag: Set to True only AFTER global model is trained and saved ***
CALCULATE_GLOBAL_PREDS = True # Default to False for initial run
# --------------------

# --- Driving Behavior & Matching Thresholds ---
ACCEL_THRESHOLD_MPS2 = 1.5; DECEL_THRESHOLD_MPS2 = -1.5; #JERK_THRESHOLD_MPS3 = 2.0

# Map Matching Config
INITIAL_MATCH_RADIUS = 25; MAX_MATCH_RADIUS = 100; MATCH_RETRY_ATTEMPTS = 3
MATCH_DISTANCE_TOLERANCE_PCT = 10.0; GAPS_THRESHOLD_S = 300
#MAX_POINTS_PER_OSRM_REQUEST = 500 # Limit points sent to OSRM match
#API_DELAY_S = 0.05 # Small delay between OSRM calls to be nice to public server
#MATCH_RADIUS_MULTIPLIER = 1.5
# ---------------------------------------------

# More column ames:
TEMP_COL = 'outside_temp'
MEAN_SPEED_KPH_COL = 'mean_speed_kph'
ACCEL_MPS2_COL = 'accel_mps2' 
JERK_MPS3_COL = 'jerk_mps3'
TIME_DIFF_S_COL = 'time_diff_s'




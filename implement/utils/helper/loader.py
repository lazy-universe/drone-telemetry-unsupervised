import pandas as pd

from implement.utils.dataset_processing.esp32_prep import preprocess_esp32_points
from implement.utils.helper.features import INTERSECTING_FEATURES, compute_geographic_bearing, engineer_features_for_df
from implement.utils.helper.paths import (
    get_esp32_raw_file,
    get_esp32_final_file,
    get_simulated_engineered_dir,
)


def engineer_consistent_telemetry_features(df_consistent: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the standardized engineering pipeline to the direct schema:
    timestamp, latitude, longitude, altitude, ground_speed, vertical_speed, course.
    Legacy OSD-style inputs are accepted for backward compatibility.
    """

    def pick_column(*names, default=None):
        for name in names:
            if name in df_consistent.columns:
                return df_consistent[name]
        if default is not None:
            return default
        raise KeyError(f"Missing required columns. Expected one of: {names}")

    df_prepared = pd.DataFrame()
    df_prepared['timestamp'] = pick_column('timestamp', 'OSD.flyTime', 'OSD.flyTime [s]')
    df_prepared['latitude'] = pick_column('latitude', 'OSD.latitude')
    df_prepared['longitude'] = pick_column('longitude', 'OSD.longitude')
    df_prepared['altitude'] = pick_column('altitude', 'height', 'OSD.height', 'OSD.altitude')
    df_prepared['ground_speed'] = pick_column('ground_speed', 'speed_horizontal', 'OSD.hSpeed')
    df_prepared['vertical_speed'] = pick_column('vertical_speed', 'speed_vertical', 'OSD.vSpeed', 'OSD.zSpeed')

    course = pick_column('course', 'direction', default=None)
    if course is None:
        lat_series = df_prepared['latitude']
        lon_series = df_prepared['longitude']
        next_lat = lat_series.shift(-1)
        next_lon = lon_series.shift(-1)
        course = pd.Series(
            compute_geographic_bearing(lat_series.values, lon_series.values, next_lat.values, next_lon.values),
            index=df_consistent.index,
            dtype=float,
        )
    else:
        next_lat = df_prepared['latitude'].shift(-1)
        next_lon = df_prepared['longitude'].shift(-1)
        bearing = pd.Series(
            compute_geographic_bearing(df_prepared['latitude'].values, df_prepared['longitude'].values, next_lat.values, next_lon.values),
            index=df_consistent.index,
            dtype=float,
        )
        course = pd.Series(course, index=df_consistent.index, dtype=float)
        course = course.where(course.notna(), bearing)
    df_prepared['course'] = course

    df_eng = engineer_features_for_df(
        df=df_prepared,
        lat_col='latitude',
        lon_col='longitude',
        alt_col='altitude',
        speed_col='ground_speed',
        heading_col='course',
        time_col='timestamp',
    )
    if 'height' not in df_eng.columns:
        df_eng['height'] = df_eng['altitude']
    return df_eng.copy()


def get_or_preprocess_dji_dataset(filter_length_100: bool = False, features=None) -> pd.DataFrame:
    """
    Loads DJI dataset from cache or precomputes and caches the master dataset.
    """
    from implement.utils.helper import get_consistent_dataset_dir, get_dji_engineered_dir, get_dji_master_file

    dji_master_csv = get_dji_master_file()
    engineered_dir = get_dji_engineered_dir()
    consistent_dir = get_consistent_dataset_dir()

    if features is None:
        features = INTERSECTING_FEATURES

    # Fast Path: Load from master preprocessed CSV
    if dji_master_csv.exists():
        print(f"Loading DJI dataset from cached master: {dji_master_csv}")
        dji_df = pd.read_csv(dji_master_csv)
        if 'height' not in dji_df.columns and 'altitude' in dji_df.columns:
            dji_df['height'] = dji_df['altitude']
        if filter_length_100 and 'flight_id' in dji_df.columns:
            flight_counts = dji_df['flight_id'].value_counts()
            valid_flights = flight_counts[flight_counts >= 100].index
            dji_df = dji_df[dji_df['flight_id'].isin(valid_flights)]
        dji_df['nature'] = 0
        return dji_df

    if not consistent_dir.exists() or not any(consistent_dir.glob('*.csv')):
        print('\nconsistent_dataset not found or empty. Running raw -> trimmed -> standard pipeline...')
        from implement.utils.dataset_processing.dji_prep import preprocess_dji_points
        from implement.utils.helper import get_dji_raw_dir
        preprocess_dji_points(get_dji_raw_dir(), output_master_file=dji_master_csv)

    all_points = []
    print('\nPreprocessing DJI flights from standard schema files...')
    engineered_dir.mkdir(parents=True, exist_ok=True)
    consistent_files = sorted(list(consistent_dir.glob('*.csv')))

    for file_path in consistent_files:
        df = pd.read_csv(file_path, low_memory=False)
        if filter_length_100 and len(df) < 100:
            continue

        df_final = engineer_consistent_telemetry_features(df)
        df_final['flight_id'] = file_path.name
        df_final.to_csv(engineered_dir / file_path.name, index=False)
        all_points.append(df_final)

    if not all_points:
        return pd.DataFrame(columns=features + ['nature', 'flight_id'])

    dji_df = pd.concat(all_points, ignore_index=True)
    dji_df['nature'] = 0

    # Save to master cache for instant loading on subsequent calls
    dji_master_csv.parent.mkdir(parents=True, exist_ok=True)
    dji_df.to_csv(dji_master_csv, index=False)
    print(f"✓ Cached DJI master dataset to: {dji_master_csv} ({len(dji_df)} rows)")

    return dji_df


def get_genuine_dji_flights_with_device_split(filter_length_100: bool = False) -> pd.DataFrame:
    """
    Loads DJI normal flights and simulated normal flights, and attaches drone_model.
    """
    from implement.utils.helper import get_consistent_dataset_dir

    consistent_dir = get_consistent_dataset_dir()

    def extract_model_from_filename(filename):
        from pathlib import Path
        stem = Path(filename).stem
        if '_flight_' in stem:
            parts = stem.split('_flight_')[1]
            return '_'.join(parts.split('_')[1:])
        return 'unknown'

    all_points = []
    for file_path in sorted(consistent_dir.glob('*.csv')):
        df = pd.read_csv(file_path, low_memory=False)
        if filter_length_100 and len(df) < 100:
            continue

        df_eng = engineer_consistent_telemetry_features(df)
        df_eng['flight_id'] = file_path.name
        df_eng['drone_model'] = extract_model_from_filename(file_path.name)
        all_points.append(df_eng)

    if not all_points:
        return pd.DataFrame()

    combined = pd.concat(all_points, ignore_index=True)
    combined['nature'] = 0
    print(
        f"Loaded DJI + SimNormal device-split dataset: "
        f"unique_models={combined['drone_model'].unique()} | {len(combined)} rows"
    )
    return combined


def get_or_preprocess_esp32_dataset() -> pd.DataFrame:
    """
    Ensures raw ESP32 data and simulated spoofed data are preprocessed and cached.
    """
    esp32_raw_file = get_esp32_raw_file()
    esp32_final_csv = get_esp32_final_file()

    if esp32_final_csv.exists():
        print(f'Loading ESP32 dataset from cached master: {esp32_final_csv}')
        esp32_df = pd.read_csv(esp32_final_csv)
    else:
        print('Preprocessing raw ESP32 bluetooth Remote ID logs...')
        esp32_df = preprocess_esp32_points(esp32_raw_file, output_file=esp32_final_csv)

    if {'prediction_error', 'ground_speed', 'vertical_acceleration'}.issubset(esp32_df.columns):
        engineered = esp32_df.copy()
    else:
        engineered = engineer_consistent_telemetry_features(esp32_df)

    engineered['nature'] = 1
    engineered['flight_id'] = 'esp32_flight'

    from implement.utils.helper import get_spoofed_flights_dir
    sim_dir = get_spoofed_flights_dir()
    sim_engineered_dir = get_simulated_engineered_dir()
    sim_master_csv = sim_engineered_dir / 'all_spoofed_records.csv'

    # Fast Path for spoofed flights
    if sim_master_csv.exists():
        print(f'Loading simulated spoofed flights from cached master: {sim_master_csv}')
        sim_df = pd.read_csv(sim_master_csv)
    else:
        sim_files = sorted(list(sim_dir.glob('*.csv')))
        sim_dfs = []

        if sim_files:
            print(f'Loading spoofed flights from {sim_dir}...')
            sim_engineered_dir.mkdir(parents=True, exist_ok=True)
            for file_path in sim_files:
                if 'normal_flight' in file_path.name:
                    continue
                df = pd.read_csv(file_path)
                if 'timestamp' not in df.columns:
                    continue
                df_final = engineer_consistent_telemetry_features(df)
                df_final['nature'] = 1
                df_final['flight_id'] = f'sim_{file_path.stem}'
                df_final.to_csv(sim_engineered_dir / file_path.name, index=False)
                sim_dfs.append(df_final)

            sim_df = pd.concat(sim_dfs, ignore_index=True) if sim_dfs else pd.DataFrame(columns=engineered.columns)
            if sim_dfs:
                sim_df.to_csv(sim_master_csv, index=False)
                print(f'✓ Cached spoofed master dataset to: {sim_master_csv}')
        else:
            print(f'[Warning] No spoofed flights found in {sim_dir}.')
            sim_df = pd.DataFrame(columns=engineered.columns)

    combined_spoofed_df = pd.concat([engineered, sim_df], ignore_index=True)
    print(f'Combined Spoofed Class Size: {len(combined_spoofed_df)} rows (ESP32: {len(engineered)}, Simulated: {len(sim_df)})')

    return combined_spoofed_df

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from implement.utils.helper import UNSUPERVISED_FEATURES

def class_balance_and_combine(dji_df, esp32_df, balance_ratio=None, random_state=42):
    """
    Concatenates DJI and ESP32. If balance_ratio is not None, downsamples DJI to be at most balance_ratio times the size of ESP32.
    """
    if balance_ratio is not None:
        n_esp = len(esp32_df)
        if len(dji_df) > n_esp * balance_ratio:
            dji_sampled = dji_df.sample(n=int(n_esp * balance_ratio), random_state=random_state)
        else:
            dji_sampled = dji_df.copy()
    else:
        dji_sampled = dji_df.copy()
        
    combined = pd.concat([dji_sampled, esp32_df], ignore_index=True)
    return combined.sample(frac=1, random_state=random_state).reset_index(drop=True)

def impute_and_scale_data(X_train, X_test, imputer=None, scaler=None):
    """
    Imputes missing values and scales features. Fits new models if not provided.
    Returns scaled features, fitted scaler, and fitted imputer.
    """
    if imputer is None:
        imputer = SimpleImputer(strategy="mean")
        X_train_imputed = imputer.fit_transform(X_train)
    else:
        X_train_imputed = imputer.transform(X_train)
        
    X_test_imputed = imputer.transform(X_test)
    
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
    else:
        X_train_scaled = scaler.transform(X_train_imputed)
        
    X_test_scaled = scaler.transform(X_test_imputed)
    
    return X_train_scaled, X_test_scaled, scaler, imputer

def combine_and_split_unsupervised_data(dji_df, esp32_df, train_ratio=0.8, split_mode='random', intermediate_dir=None, output_dir=None, random_state=42, features=None):
    """
    Merges DJI and ESP32 datasets for unsupervised anomaly detection.
    The training set contains ONLY real/clean DJI samples (nature=0), meaning Fake count is 0.
    The test set contains a combination of DJI (nature=0) and ESP32 (nature=1) samples.
    """
    if features is None:
        features = UNSUPERVISED_FEATURES

    # Save intermediate combined steps first if intermediate_dir is provided
    if intermediate_dir:
        intermediate_dir = Path(intermediate_dir)
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Save DJI records only
        dji_aligned = dji_df[features + ['nature']].copy()
        dji_aligned.to_csv(intermediate_dir / '01_sampled_dji_records.csv', index=False)
        
        esp32_aligned = esp32_df[features + ['nature']].copy()
        combined_records = pd.concat([dji_aligned, esp32_aligned], ignore_index=True)
        combined_records.to_csv(intermediate_dir / '02_combined_records.csv', index=False)
        combined_records.to_csv(intermediate_dir / '03_combined_records_without_lat_long.csv', index=False)
        
        updated_dataset = combined_records.sample(frac=1, random_state=random_state).reset_index(drop=True)
        updated_dataset.to_csv(intermediate_dir / '04_updated_dataset.csv', index=False)

    dji_clean = dji_df.copy()
    esp32_clean = esp32_df.copy()
    
    esp32_df_clean = esp32_clean.drop(columns=['flight_id'], errors='ignore')
    
    if split_mode == 'flight' and 'flight_id' in dji_clean.columns:
        print("[Split Mode] Executing Flight-wise partition split on DJI flights for unsupervised mode...")
        unique_flights = dji_clean['flight_id'].unique()
        train_flights, test_flights = train_test_split(
            unique_flights, test_size=(1.0 - train_ratio), random_state=random_state
        )
        
        dji_train = dji_clean[dji_clean['flight_id'].isin(train_flights)].drop(columns=['flight_id'])
        dji_test = dji_clean[dji_clean['flight_id'].isin(test_flights)].drop(columns=['flight_id'])
    else:
        print("[Split Mode] Executing Point-wise random split for unsupervised mode...")
        dji_no_id = dji_clean.drop(columns=['flight_id'], errors='ignore')
        
        dji_train, dji_test = train_test_split(
            dji_no_id, test_size=(1.0 - train_ratio), random_state=random_state
        )
        
    # Unsupervised: training split consists ONLY of the normal DJI train subset (nature=0)
    train_df = dji_train.copy()
    
    # Test set combines DJI test and clean ESP32 fakes (downsampled to 1:1 if fakes > real)
    if len(esp32_df_clean) > len(dji_test):
        print(f"Downsampling fake test points from {len(esp32_df_clean)} to {len(dji_test)} (1:1 ratio with real)")
        esp32_df_clean = esp32_df_clean.sample(n=len(dji_test), random_state=random_state)
    test_df = class_balance_and_combine(dji_test, esp32_df_clean, balance_ratio=None, random_state=random_state)

    train_df['nature'] = train_df['nature'].astype(int)
    test_df['nature'] = test_df['nature'].astype(int)
    
    # Save splits to disk if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_dir / 'train_dataset.csv', index=False)
        test_df.to_csv(output_dir / 'test_dataset.csv', index=False)
        print(f"Saved split datasets to: {output_dir}")
        
    X_train = train_df[features]
    y_train = train_df['nature'].to_numpy()
    
    X_test = test_df[features]
    y_test = test_df['nature'].to_numpy()
    
    # Impute and standardize (fitted ONLY on normal/real train data)
    X_train_scaled, X_test_scaled, scaler, imputer = impute_and_scale_data(X_train, X_test)
    
    print(f"Unsupervised dataset split completed ({split_mode}): Train={X_train_scaled.shape} (Fakes=0) | Test={X_test_scaled.shape}")
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler, imputer

def inject_pca_pc1(X_train, X_test, pca=None):
    """
    Fits PCA (n_components=1) on training data, extracts the first principal component,
    and appends it as an additional feature column on train and test splits.
    Returns the expanded arrays and the fitted PCA model.
    """
    if pca is None:
        pca = PCA(n_components=1)
        PC1_train = pca.fit_transform(X_train)
    else:
        PC1_train = pca.transform(X_train)
        
    PC1_test = pca.transform(X_test)
    
    X_train_final = np.hstack((X_train, PC1_train))
    X_test_final = np.hstack((X_test, PC1_test))
    
    print(f"PCA PC1 appended. Expanded feature dimensions from {X_train.shape[1]} to {X_train_final.shape[1]}")
    return X_train_final, X_test_final, pca

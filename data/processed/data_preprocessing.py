import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

file_path = 'C:/AISE4010/FinalProject/AISE4010_Canada_Energy/data/raw/canada_energy.csv'
data = pd.read_csv(file_path)

#Convert date to DateTime and verify chronological order
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by='date')

#one-hot encoding for producer and generation_type(categorical variables)
data_encoded = pd.get_dummies(data, columns=['producer', 'generation_type'])

#replace negative production values with 0
data_encoded['megawatt_hours'] = data_encoded['megawatt_hours'].clip(lower=0)
#create new feature for net generation by province each month (sum of all producers)
# Calculate 'typeNetGeneration' - sum of 'megawatt_hours' grouped by 'province' and 'date'
type_NetGeneration = data.groupby(['province', 'date'])['megawatt_hours'].sum().reset_index()
type_NetGeneration.rename(columns={'megawatt_hours': 'typeNetGeneration'}, inplace=True)
# Keep only 'date', 'province', and 'typeNetGeneration'
type_NetGeneration = type_NetGeneration[['date', 'province', 'typeNetGeneration']]

scaler = MinMaxScaler()
data_encoded['megawatt_hours'] = scaler.fit_transform(data_encoded[['megawatt_hours']])
type_NetGeneration['typeNetGeneration'] = scaler.fit_transform(type_NetGeneration[['typeNetGeneration']])

#separate dataset by province
province_dfs = {province: data_encoded[data_encoded['province'] == province] for province in data_encoded['province'].unique()}
type_NetGeneration_dfs = {province: type_NetGeneration[type_NetGeneration['province'] == province] for province in type_NetGeneration['province'].unique()}

# Reorder type_NetGeneration_dfs to match the order of province_dfs
ordered_type_NetGeneration_dfs = {province: type_NetGeneration_dfs[province] for province in province_dfs.keys()}

# If needed, overwrite the original dictionary with the ordered one
type_NetGeneration_dfs = ordered_type_NetGeneration_dfs

#for province, df in province_dfs.items():
#    print(f"Head of for province {province}:")
#    print(df.head())

#for province, df in type_NetGeneration_dfs.items():
#    print(f"Head of Net Generation for province {province}:")
#    print(df.head())

# Function to create a sliding window for the encoded province subset
def create_feature_sliding_window(features, date_column, window_size=6):
   
    
    grouped = features.groupby(features[date_column].dt.to_period('M'))  # Group by month
    windows = []
    months = list(grouped.groups.keys())

    # Slide over 6 months at a time
    for i in range(len(months) - window_size):
        # Gather rows from each month within the window
        window_rows = pd.concat(
            [grouped.get_group(month).drop(columns=[date_column]).reset_index(drop=True)
             for month in months[i:i + window_size]],
            axis=0
        )
        # Flatten into a single row
        windows.append(window_rows.values.flatten())
    return pd.DataFrame(windows)

def create_target_sliding_window(targets, window_size=6):
    windows = []
    for i in range(len(targets) - window_size):
        windows.append(targets.iloc[i + window_size].values)
    return pd.DataFrame(windows)

# Create sliding windows for features and targets
province_feature_windows = {}
province_target_windows = {}
window_size = 6

for province, df in province_dfs.items():
    # Feature sliding window
    sliding_features = create_feature_sliding_window(df, date_column='date', window_size=window_size)
    province_feature_windows[province] = sliding_features

    # Target sliding window
    targets = type_NetGeneration_dfs[province][['typeNetGeneration']]
    sliding_targets = create_target_sliding_window(targets, window_size=window_size)
    province_target_windows[province] = sliding_targets

# Combine sliding window datasets for all provinces
all_features = pd.concat(province_feature_windows.values(), axis=0).reset_index(drop=True)
all_targets = pd.concat(province_target_windows.values(), axis=0).reset_index(drop=True)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(all_features, all_targets, test_size=0.2, random_state=42, shuffle=True)


# Outputs
print(f"Training features shape: {X_train.shape}")
print(f"Training targets shape: {y_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Test targets shape: {y_test.shape}")


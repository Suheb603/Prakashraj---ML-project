import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
# CONFIGURATION
# ========================================================================
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

CLIP_FLOOR = 1.0
FREQ_THRESHOLD = 0.01
RANDOM_STATE = 42

print("=" * 80)
print("ELASTIC NET MODEL FOR MEDICAL EQUIPMENT COST PREDICTION")
print("=" * 80)

# ========================================================================
# 1. DATA LOADING
# ========================================================================
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

test_ids = test_df['Hospital_Id'].copy()
train_df.set_index('Hospital_Id', inplace=True)
test_df.set_index('Hospital_Id', inplace=True)

y_train = train_df["Transport_Cost"].copy()
X_train = train_df.drop(columns=["Transport_Cost"]).copy()
X_test = test_df.copy()

# ========================================================================
# 2. TARGET TRANSFORMATION
# ========================================================================
y_train_transformed = np.log(y_train.clip(lower=CLIP_FLOOR))

# ========================================================================
# 3. FEATURE ENGINEERING
# ========================================================================
combined_df = pd.concat([X_train, X_test], axis=0)

# --- Date features ---
combined_df['Order_Placed_Date'] = pd.to_datetime(combined_df['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
combined_df['Delivery_Date'] = pd.to_datetime(combined_df['Delivery_Date'], format='%m/%d/%y', errors='coerce')

combined_df['Delivery_Lag_Days'] = (combined_df['Delivery_Date'] - combined_df['Order_Placed_Date']).dt.days.fillna(0).astype(int)
combined_df['Order_Day_of_Week'] = combined_df['Order_Placed_Date'].dt.dayofweek
combined_df['Order_Month'] = combined_df['Order_Placed_Date'].dt.month
combined_df['Order_Quarter'] = combined_df['Order_Placed_Date'].dt.quarter
combined_df['Order_Year'] = combined_df['Order_Placed_Date'].dt.year
combined_df['Delivery_Month'] = combined_df['Delivery_Date'].dt.month
combined_df['Is_Weekend_Order'] = (combined_df['Order_Day_of_Week'] >= 5).astype(int)

# --- Geometric features ---
combined_df['Equipment_Volume'] = combined_df['Equipment_Height'] * combined_df['Equipment_Width']
combined_df['Equipment_Density'] = combined_df['Equipment_Weight'] / (combined_df['Equipment_Volume'] + 1e-6)
combined_df['Equipment_Surface_Area'] = 2 * combined_df['Equipment_Height'] * combined_df['Equipment_Width'] + 4 * combined_df['Equipment_Width'] ** 2
combined_df['Height_Width_Ratio'] = combined_df['Equipment_Height'] / (combined_df['Equipment_Width'] + 1e-6)

# --- Value/Cost features ---
combined_df['Value_per_Unit_Weight'] = combined_df['Equipment_Value'] / (combined_df['Equipment_Weight'] + 1e-6)
combined_df['Fee_per_Unit_Weight'] = combined_df['Base_Transport_Fee'] / (combined_df['Equipment_Weight'] + 1e-6)
combined_df['Value_to_Fee_Ratio'] = combined_df['Equipment_Value'] / (combined_df['Base_Transport_Fee'] + 1e-6)
combined_df['Log_Equipment_Value'] = np.log1p(combined_df['Equipment_Value'])
combined_df['Log_Base_Transport_Fee'] = np.log1p(combined_df['Base_Transport_Fee'])
combined_df['Log_Equipment_Weight'] = np.log1p(combined_df['Equipment_Weight'])
combined_df['Log_Equipment_Volume'] = np.log1p(combined_df['Equipment_Volume'].clip(lower=0))

# --- Supplier reliability ---
combined_df['Reliability_Squared'] = combined_df['Supplier_Reliability'] ** 2
combined_df['Reliability_Cubed'] = combined_df['Supplier_Reliability'] ** 3
combined_df['Low_Reliability'] = (combined_df['Supplier_Reliability'] < 50).astype(int)
combined_df['High_Reliability'] = (combined_df['Supplier_Reliability'] > 80).astype(int)

# --- Complex interactions ---
combined_df['Urgency_Score'] = combined_df['Urgent_Shipping'].map({'Yes': 1, 'No': 0}).fillna(0) * combined_df['Delivery_Lag_Days']
combined_df['Fragility_Score'] = combined_df['Fragile_Equipment'].map({'Yes': 1, 'No': 0}).fillna(0) * combined_df['Equipment_Weight']
combined_df['Service_Complexity'] = (
    combined_df['CrossBorder_Shipping'].map({'Yes': 1, 'No': 0}).fillna(0) +
    combined_df['Installation_Service'].map({'Yes': 1, 'No': 0}).fillna(0) +
    combined_df['Urgent_Shipping'].map({'Yes': 1, 'No': 0}).fillna(0) +
    combined_df['Fragile_Equipment'].map({'Yes': 1, 'No': 0}).fillna(0)
)
combined_df['Heavy_Valuable'] = (
    (combined_df['Equipment_Weight'] > combined_df['Equipment_Weight'].median()) &
    (combined_df['Equipment_Value'] > combined_df['Equipment_Value'].median())
).astype(int)

combined_df.drop(columns=['Order_Placed_Date', 'Delivery_Date', 'Supplier_Name', 'Hospital_Location'],
                 inplace=True, errors='ignore')

# --- Binary mapping ---
binary_map = {'Yes': 1, 'No': 0}
binary_cols_to_map = ['CrossBorder_Shipping', 'Installation_Service', 'Rural_Hospital', 'Urgent_Shipping', 'Fragile_Equipment']
for col in binary_cols_to_map:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].map(binary_map).fillna(0).astype(int)

# --- Handle categorical frequencies ---
categorical_cols_to_group = ['Equipment_Type', 'Transport_Method', 'Hospital_Info']
for col in categorical_cols_to_group:
    if col in combined_df.columns:
        train_counts = combined_df.iloc[:len(X_train)][col].value_counts(normalize=True)
        low_freq_cats = train_counts[train_counts < FREQ_THRESHOLD].index
        combined_df[col] = np.where(combined_df[col].isin(low_freq_cats), 'Other', combined_df[col])

X_train_clean = combined_df.iloc[:len(X_train)].copy()
X_test_clean = combined_df.iloc[len(X_train):].copy()

# ========================================================================
# 4. PREPROCESSING PIPELINE
# ========================================================================
numeric_cols = []
binary_cols = binary_cols_to_map
categorical_cols = []

for col in X_train_clean.columns:
    if col in binary_cols:
        continue
    elif X_train_clean[col].dtype == 'object':
        categorical_cols.append(col)
    else:
        numeric_cols.append(col)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'
)

# ========================================================================
# 5. ELASTIC NET MODEL PIPELINE
# ========================================================================
elastic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', ElasticNet(
        alpha=0.4,          # Regularization strength
        l1_ratio=0.5,       # Mix between L1 (Lasso) and L2 (Ridge)
        max_iter=5000,
        random_state=RANDOM_STATE
    ))
])

# ========================================================================
# 6. TRAINING
# ========================================================================
print("Training the Elastic Net model...")
elastic_pipeline.fit(X_train_clean, y_train_transformed)
print("✅ Model training completed.")

# ========================================================================
# 7. MODEL EVALUATION (on training data)
# ========================================================================
y_train_pred_log = elastic_pipeline.predict(X_train_clean)
y_train_pred = np.exp(y_train_pred_log)

rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
mae = mean_absolute_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

print("\n=== TRAINING PERFORMANCE METRICS ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ========================================================================
# 8. PREDICTION & EXPORT
# ========================================================================
y_test_pred_log = elastic_pipeline.predict(X_test_clean)
y_test_pred = np.exp(y_test_pred_log)

submission = pd.DataFrame({
    'Hospital_Id': test_ids,
    'Transport_Cost': y_test_pred
})

submission.to_csv("elasticnet_predictions.csv", index=False)
print("\n✅ Predictions saved as 'elasticnet_predictions.csv'")
print("=" * 80)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest

df = pd.read_csv('../data/processed/01_cleaned_data.csv')
df['log_harga'] = np.log1p(df['harga'])

# Total fasilitas
fas_cols = ['ac', 'akses_24_jam', 'kasur', 'kloset_duduk', 'kamar_mandi_dalam', 'wifi']
df['total_fasilitas'] = 0
for f in fas_cols:
    df['total_fasilitas'] += df[f].astype(str).str.lower().isin(['ya', 'ada']).astype(int)

# One hot encoding 
cat_cols = ['wilayah', 'kecamatan', 'jenis', 'ac', 'akses_24_jam', 'kasur', 'kloset_duduk', 'kamar_mandi_dalam', 'wifi']
df = pd.get_dummies(df, columns=[col for col in cat_cols if col in df.columns])

X = df.drop(columns=['harga', 'log_harga', 'nama', 'url'], errors='ignore')
for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = X[col].astype(int)

y = df['log_harga']

# Isolation forest on X and y
iso = IsolationForest(contamination=0.05, random_state=42)
# We can fit on X and y combined
data_for_iso = pd.concat([X, y], axis=1)
yhat = iso.fit_predict(data_for_iso)
mask = yhat != -1
X_clean, y_clean = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.1, random_state=42)

model = xgb.XGBRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R2 with Isolation Forest:", r2_score(y_test, y_pred))
print("Data size:", len(X_clean))

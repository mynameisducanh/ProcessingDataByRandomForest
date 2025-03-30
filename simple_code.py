import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ===== Cấu hình cố định =====
input_file = "data_input.xlsx" # file đầu vào
output_file = "data_output.xlsx" # file đầu ra  
target_column = "Color" # cột cần dự đoán
feature_columns = ["Product", "Category"] # cột đầu vào

# ===== Đọc dữ liệu =====
df = pd.read_excel(input_file)

# ===== Lưu kiểu dữ liệu gốc =====
original_dtype = df[target_column].dtype

# ===== Chuẩn hóa + xác định thiếu =====
if df[target_column].dtype == object:
    df[target_column] = df[target_column].astype(str).str.strip().str.lower()

missing_mask = df[target_column].isna() | df[target_column].astype(str).str.strip().str.lower().isin(['nan', 'n/a', 'unknown', 'na'])
df_missing = df[missing_mask].copy()
df_clean = df[~missing_mask].copy()

# ===== Mã hóa nhãn và TF-IDF =====
label_encoder = LabelEncoder()
df_clean['TargetEncoded'] = label_encoder.fit_transform(df_clean[target_column])

X_clean = df_clean[feature_columns].astype(str)
X_missing = df_missing[feature_columns].astype(str)

preprocessor = ColumnTransformer([
    (f"tfidf_{i}", TfidfVectorizer(max_features=100), col)
    for i, col in enumerate(feature_columns)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# ===== Train model & fill missing =====
model.fit(X_clean, df_clean['TargetEncoded'])

if not df_missing.empty:
    pred_encoded = model.predict(X_missing)
    pred_labels = label_encoder.inverse_transform(pred_encoded)
    df_missing[target_column] = pred_labels
    df_missing[target_column] = df_missing[target_column].astype(original_dtype)
    df_result = pd.concat([df_clean.drop(columns='TargetEncoded'), df_missing], ignore_index=True)
else:
    df_result = df_clean.drop(columns='TargetEncoded')

# ===== Lưu kết quả =====
df_result.to_excel(output_file, index=False)
print(f"✅ Fill xong và lưu ra: {output_file}")

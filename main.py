import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data():
    file_path = input("🔍 Nhập tên file Excel (vd: data.xlsx): ").strip()
    
    xl = pd.ExcelFile(file_path)
    print("\n📑 Danh sách sheet có trong file:")
    for idx, sheet in enumerate(xl.sheet_names):
        print(f"{idx + 1}. {sheet}")

    sheet_choice = input("📄 Nhập tên sheet bạn muốn dùng (hoặc nhập số thứ tự): ").strip()

    if sheet_choice.isdigit():
        sheet_index = int(sheet_choice) - 1
        if 0 <= sheet_index < len(xl.sheet_names):
            sheet_name = xl.sheet_names[sheet_index]
        else:
            print("❌ Số thứ tự không hợp lệ. Thoát.")
            exit()
    else:
        sheet_name = sheet_choice
        if sheet_name not in xl.sheet_names:
            print("❌ Tên sheet không tồn tại. Thoát.")
            exit()

    df = xl.parse(sheet_name)
    print(f"\n✅ Đã tải sheet: {sheet_name}")
    print("\n📋 Các cột trong bảng dữ liệu:")
    print(df.columns.tolist())
    return df


def show_missing_info(df):
    print("\n❗ Các cột có giá trị thiếu (có thể dự đoán được):")
    found_missing = False
    column_empty = ""
    for col in df.columns:
        count = df[col].astype(str).str.strip().str.lower().isin(['nan', 'n/a', 'unknown', 'na', '']).sum()
        count += df[col].isna().sum()
        if count > 0:
            print(f"🔸 {col}: {count} dòng thiếu")
            if not column_empty:
                column_empty = col
            found_missing = True
    if not found_missing:
        print("✅ Không có cột nào bị thiếu")
    return column_empty

def get_user_input(default_target):
    print("\n💡 Khi nhập tên cột nhớ nhập chính xác cả in hoa và thường")
    target_column = input(f"\n🎯 Nhập tên cột mục tiêu cần dự đoán (vd: {default_target}): ").strip()
    print("\n💡 Khi nhập các cột đầu vào, nên chọn các cột liên quan trực tiếp đến cột cần dự đoán.")
    feature_columns = input("🧩 Nhập các cột đầu vào (cách nhau bằng dấu phẩy, vd: Product,Category): ").strip().split(',')
    print(f"\n👩‍💻 Chờ xíu nha người đẹp, máy đang xử lý nè...")
    feature_columns = [f.strip() for f in feature_columns]
    return target_column, feature_columns

def split_data(df, target_column):
    original_dtype = df[target_column].dtype
    if df[target_column].dtype == object:
        df[target_column] = df[target_column].astype(str).str.strip().str.lower()

    missing_mask = df[target_column].isna() | df[target_column].astype(str).str.strip().str.lower().isin(['nan', 'n/a', 'unknown', 'na'])
    df_missing = df[missing_mask].copy()
    df_clean = df[~missing_mask].copy()
    
    return df_clean, df_missing, original_dtype

def build_model(feature_columns):
    text_transformers = []
    for i, col in enumerate(feature_columns):
        text_transformers.append((f'tfidf_{i}', TfidfVectorizer(max_features=100), col))
    preprocessor = ColumnTransformer(transformers=text_transformers)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    return model

def train_and_evaluate(model, df_clean, feature_columns):
    label_encoder = LabelEncoder()
    df_clean['TargetEncoded'] = label_encoder.fit_transform(df_clean[target_column])
    X_clean_str = df_clean[feature_columns].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean_str, df_clean['TargetEncoded'], test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    unique_labels = np.unique(y_test)
    target_names_raw = label_encoder.inverse_transform(unique_labels)
    target_names = [str(name) for name in target_names_raw]
    print("\n📈 Báo cáo đánh giá mô hình:")
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

    return model, label_encoder

def predict_missing(model, df_clean, df_missing, feature_columns, label_encoder, original_dtype):
    if not df_missing.empty:
        X_missing_str = df_missing[feature_columns].astype(str)
        predicted_encoded = model.predict(X_missing_str)
        predicted_labels = label_encoder.inverse_transform(predicted_encoded)
        df_missing[target_column] = predicted_labels
        df_missing[target_column] = df_missing[target_column].astype(original_dtype)

        df_final = pd.concat([df_clean.drop(columns='TargetEncoded'), df_missing], ignore_index=True)
    else:
        df_final = df_clean.drop(columns='TargetEncoded')
    return df_final

def sort_result(df_final):
    print(f"\n📋 Các cột hiện có: {df_final.columns.tolist()}")
    sort_column = input("📊 Bạn muốn sắp xếp tăng dần theo cột nào? (nhấn Enter để sắp theo cột đầu tiên): ").strip()
    if sort_column == "":
        sort_column = df_final.columns[0]
        print(f"⚙️ Đã sắp xếp theo cột mặc định: {sort_column}")
    elif sort_column not in df_final.columns:
        print(f"⚠️ Cột '{sort_column}' không tồn tại. Bỏ qua bước sắp xếp.")
        return df_final
    df_final = df_final.sort_values(by=sort_column, ascending=True)
    return df_final

def save_file(df_final):
    default_name = "data_filled.xlsx"
    output_file = input(f"\n💾 Nhập tên file muốn lưu (nhấn Enter để dùng mặc định: {default_name}): ").strip()
    print(f"\n☕ Chill xíu nghen, dữ liệu đang được điền vào file ...")
    if output_file == "":
        output_file = default_name
    elif not output_file.endswith(".xlsx"):
        output_file += ".xlsx"
    df_final.to_excel(output_file, index=False)
    print(f"\n✅ Done , Dữ liệu đã được điền và lưu vào file: {output_file}")

# ========== MAIN ==========

if __name__ == "__main__":
    df = load_data()                                                                                                                                    
    column_empty = show_missing_info(df)
    target_column, feature_columns = get_user_input(column_empty)
    df_clean, df_missing, original_dtype = split_data(df, target_column)
    model = build_model(feature_columns)
    model, label_encoder = train_and_evaluate(model, df_clean, feature_columns)
    df_final = predict_missing(model, df_clean, df_missing, feature_columns, label_encoder, original_dtype)
    df_final = sort_result(df_final)
    save_file(df_final)

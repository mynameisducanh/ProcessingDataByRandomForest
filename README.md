# 🧠 Tự động điền dữ liệu thiếu bằng mô hình học máy (Random Forest)

Code này sử dụng mô hình học máy `RandomForestClassifier` kết hợp với `TfidfVectorizer` để dự đoán và điền các giá trị còn thiếu trong một cột bất kỳ (ví dụ: `Color`, `Brand`, `ShipDateKey`, ...), dựa vào các cột đầu vào liên quan (như `ProductName`, `Category`, ...).

---

## 📦 Thư viện cần cài

```bash
pip install pandas numpy scikit-learn openpyxl
```

## 📦 Cách chạy chương trình

```bash
python main.py
```

```bash
python simple_code.py
```

---

## 📚 Tài liệu tham khảo

- [`RandomForestClassifier` - scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

---

## ✅ Khi nào nên dùng mô hình học máy để điền dữ liệu thiếu?

Bạn nên dùng mô hình này khi:

---

### 1. **Bạn có nhiều dữ liệu (thường ≥ 100 dòng)**

Mô hình cần học được mối quan hệ giữa các cột → càng nhiều dòng, kết quả càng chính xác.

📌 _Ví dụ:_ Có 2.000 sản phẩm, một số bị thiếu `Color`.

---

### 2. **Giá trị bị thiếu thuộc loại phân loại (categorical)**

Ví dụ các giá trị có số lượng giới hạn như: `red`, `blue`, `black`,...

Phù hợp với các cột như:

- `Color`
- `Brand`
- `Category`
- `ShipDateKey` (nếu là mã phân loại, không phải ngày thực)

---

### 3. **Có các cột đầu vào liên quan mật thiết đến cột bị thiếu**

Mô hình cần "manh mối" để suy đoán.

📌 _Ví dụ:_  
Nếu `Color` bị thiếu, mà `ProductName = "Adidas Ultraboost Black Edition"`  
→ khả năng cao `Color = black`.

---

### 4. **Khi bạn không thể điền thủ công hoặc tra cứu được**

- Không có quy tắc rõ ràng
- Không có API tra cứu
- Dữ liệu quá nhiều để làm tay

---

### 5. **Khi bạn lười :))**

Chỉ cần chạy code, mô hình sẽ dự đoán giúp bạn.  
Chấp nhận độ chính xác dao động khoảng **70–99%** tùy dữ liệu :)))(các nhiều dữ liệu để train độ chính xác càng cao).

---

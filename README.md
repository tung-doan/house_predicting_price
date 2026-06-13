# Dự đoán Giá Nhà Hà Nội 🏠 (30 quận/huyện) – Bản đã chỉnh theo yêu cầu

Bản này đã được chỉnh lại để:

- ✅ **Bỏ thuộc tính `Nhom_Khu_vuc`** ("Nhóm khu vực") khỏi dữ liệu *và* giao diện
- ✅ **UI có ràng buộc theo quận/huyện dựa trên dữ liệu thô**:
  - Numeric: min/max + P05/Median/P95 theo từng quận/huyện
  - Categorical/Binary: chỉ hiện những lựa chọn *có thật trong dữ liệu* của quận/huyện đó
- ✅ Sửa cách xử lý **biến phân loại** bằng sklearn **Pipeline + OneHotEncoder(handle_unknown='ignore')**  
  → đổi các thuộc tính phân loại sẽ **làm giá dự đoán thay đổi** (không còn bị “đứng giá” do lệch schema)
- ✅ Thêm **sơ đồ dự báo 12 tháng** (lãi kép theo công thức: r_tháng = (1+r_năm)^(1/12) - 1; mặc định 2025: 17%/năm, 2026: 15%/năm)
- ✅ Bổ sung **nhiều thuật toán** để so sánh và chọn mô hình tốt nhất (Multi-Model Training)

---

## 1) Dữ liệu & các file đầu ra

Nguồn dữ liệu đầu vào: `HaNoi_Housing_Ultimate_Full.csv` (≈ 50.000 dòng)

Sau khi chạy `preprocessing.py`, repo sẽ có:

- `HN_Houseprice.csv` : dữ liệu gốc (đã loại `Nhom_Khu_vuc`)
- `HN_Houseprice_Cleaned.csv` : dữ liệu làm sạch (≤ 20.000 dòng theo yêu cầu)
- `HN_Houseprice_Encoded.csv` : dữ liệu one-hot để EDA
- `HN_Houseprice_Processed.csv` : thêm cột `Gia_ban_ty_log` để hỗ trợ phân tích
- `feature_schema.json` : schema cho Streamlit UI (danh sách category + min/max/median)
- `cleaning_report.json` : log tóm tắt làm sạch

> Lưu ý: Dữ liệu làm sạch được lấy mẫu **có stratify theo `Quan_Huyen`** để đảm bảo đủ **30 quận/huyện**.

---

## 2) Train model (Multi-Model)

Model được train theo log-target (`log1p(Gia_ban_ty)`) bằng sklearn Pipeline:

- Numeric + binary: `StandardScaler`
- Categorical: `OneHotEncoder(handle_unknown='ignore')`
- Thuật toán so sánh (tuỳ máy / tuỳ cài):
  - Linear Regression
  - Ridge Regression
  - Random Forest
  - Extra Trees
  - Gradient Boosting
  - HistGradientBoosting
  - KNN Regression
  - XGBoost Regressor (nếu cài `xgboost`)

Chạy nhanh (khuyến nghị):

```bash
python model_training.py --sample 15000 --fast
```

Chạy đầy đủ + lưu tất cả mô hình (để chọn trong UI):

```bash
python model_training.py --save_all
```

Kết quả:
- `best_model.pkl` (pipeline model tốt nhất theo MAE)
- `model_comparison.csv` (bảng so sánh)
- `model_info.json` (best model + metrics)
- `models/*.pkl` (nếu dùng `--save_all`)

---

## 3) Chạy app Streamlit

```bash
streamlit run app.py
```

Trong app:
- Chọn **Quận/Huyện** → UI tự ràng buộc range & lọc option dựa theo dữ liệu
- (Nếu bạn đã `--save_all`) có thể chọn **mô hình dự đoán** trong sidebar
- Bấm **Dự đoán giá**
- App sẽ hiển thị:
  - Giá dự đoán (tỷ VNĐ)
  - Quy đổi VND
  - Giá/m² ước tính
  - **Biểu đồ dự báo 12 tháng** theo kịch bản (Cơ sở / Thận trọng / Tăng nhanh)

---

## 4) Dự báo 12 tháng (file mẫu)

Có sẵn script để sinh 1 biểu đồ mẫu:

```bash
python forecast_12m.py
```

Tạo ra:
- `forecast_12m.csv`
- `forecast_12m.png`

---

## 5) Ghi chú quan trọng

- Kết quả chỉ mang tính tham khảo (mô phỏng), không phải khuyến nghị đầu tư.
- Biểu đồ 12 tháng là **kịch bản mô phỏng** (có shock dương/âm để phản ánh biến động thị trường).

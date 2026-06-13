import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Cấu hình font tiếng Việt cho matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'

def evaluate_model(y_true_log, y_pred_log, model_name):
    # Nghịch đảo log (y_log = log(1+y) -> y = exp(y_log) - 1)
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true_log, y_pred_log) # R2 thường tính trên quy mô log nếu train trên log
    
    return {
        "Model": model_name,
        "MAE (Tỷ VNĐ)": mae,
        "RMSE (Tỷ VNĐ)": rmse,
        "R2 Score": r2
    }

def main():
    print("--- 🤖 KHỞI ĐỘNG GIAI ĐOẠN HUẤN LUYỆN MÔ HÌNH (SỬA LỖI LEAKAGE) ---")
    
    # Load data
    df = pd.read_csv('HN_Houseprice_Encoded.csv')
    
    # Xác định các cột cần loại bỏ (Metadata, Target, và các cột dễ gây đa cộng tuyến)
    # Giữ lại: Area_m2, Bedrooms, Bathrooms, Floors, Width, Entrance_width
    # Giữ lại time features có tương quan tốt nhất: Days_Since_Posted (-0.043), PostingQuarter (0.007)
    # Loại bỏ: PostingYear, PostingMonth (đa cộng tuyến), Month_sin/cos, Half_Year (correlation < 0.01)
    drop_cols = [
        'Title', 'Address', 'PostingDate', 'PostType', 'Area', 'Direction', 
        'Width_meters', 'Legal', 'Interior', 'Entrancewidth', 'Price', 'Price_per_m2',
        'PostingYear', 'PostingMonth', 'Month_sin', 'Month_cos', 'Half_Year'
    ]
    
    # Chỉ giữ lại các cột số thực sự là features
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['Price']
    
    # Fill NaNs for baseline models (Linear Regression)
    X = X.fillna(X.median())
    
    print(f"Số lượng Features sử dụng: {X.shape[1]}")
    print(f"Các features quan trọng: {X.columns[:10].tolist()}...")
    
    # 1. Chia tập Train/Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Xử lý biến mục tiêu: Log Transformation
    print("[1/7] Đang áp dụng Log Transformation...")
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    # Chuẩn hóa dữ liệu cho KNN (KNN yêu cầu dữ liệu được scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    
    # ============== MODEL 1: LINEAR REGRESSION ==============
    # Người phụ trách: Thành viên 1
    # Mô tả: Thuật toán hồi quy tuyến tính cơ bản (Baseline)
    print("[2/7] Đang huấn luyện Model 1: Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train_log)
    y_pred_lr = lr.predict(X_test_scaled)
    results.append(evaluate_model(y_test_log, y_pred_lr, "Linear Regression"))
    
    # ============== MODEL 2: RIDGE REGRESSION ==============
    # Người phụ trách: Thành viên 2
    # Mô tả: Hồi quy tuyến tính với regularization L2, giảm overfitting
    # Tham số alpha: độ mạnh của regularization (alpha càng lớn, model càng đơn giản)
    print("[3/7] Đang huấn luyện Model 2: Ridge Regression...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train_log)
    y_pred_ridge = ridge.predict(X_test_scaled)
    results.append(evaluate_model(y_test_log, y_pred_ridge, "Ridge Regression"))
    
    # ============== MODEL 3: K-NEAREST NEIGHBORS (KNN) ==============
    # Người phụ trách: Thành viên 3
    # Mô tả: Dự đoán dựa trên K điểm dữ liệu gần nhất
    # Lưu ý: KNN cần dữ liệu được chuẩn hóa (scaled) để tính khoảng cách chính xác
    print("[4/7] Đang huấn luyện Model 3: K-Nearest Neighbors (KNN)...")
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)
    knn.fit(X_train_scaled, y_train_log)
    y_pred_knn = knn.predict(X_test_scaled)
    results.append(evaluate_model(y_test_log, y_pred_knn, "KNN (K=5)"))
    
    # ============== MODEL 4: RANDOM FOREST ==============
    # Người phụ trách: Thành viên 4
    # Mô tả: Thuật toán ensemble sử dụng nhiều cây quyết định
    print("[5/7] Đang huấn luyện Model 4: Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_log)
    y_pred_rf = rf.predict(X_test)
    results.append(evaluate_model(y_test_log, y_pred_rf, "Random Forest"))
    
    # ============== MODEL 5: XGBOOST ==============
    # Người phụ trách: Thành viên 5
    # Mô tả: Thuật toán Gradient Boosting tối ưu hóa hiệu suất cao
    print("[6/7] Đang huấn luyện Model 5: XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train_log)
    y_pred_xgb = xgb_model.predict(X_test)
    results.append(evaluate_model(y_test_log, y_pred_xgb, "XGBoost"))
    
    # 6. So sánh kết quả
    print("\n" + "="*70)
    print("📊 BẢNG SO SÁNH KẾT QUẢ 5 MÔ HÌNH MACHINE LEARNING")
    print("="*70)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Xếp hạng theo MAE
    print("\n" + "-"*70)
    print("📈 XẾP HẠNG THEO MAE (Mean Absolute Error - Thấp hơn = Tốt hơn):")
    print("-"*70)
    results_sorted_mae = results_df.sort_values('MAE (Tỷ VNĐ)')
    for i, row in enumerate(results_sorted_mae.itertuples(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"   {medal} {i}. {row.Model}: MAE = {row._2:.2f} tỷ VNĐ")
    
    # Xếp hạng theo RMSE
    print("\n" + "-"*70)
    print("📈 XẾP HẠNG THEO RMSE (Root Mean Squared Error - Thấp hơn = Tốt hơn):")
    print("-"*70)
    results_sorted_rmse = results_df.sort_values('RMSE (Tỷ VNĐ)')
    for i, row in enumerate(results_sorted_rmse.itertuples(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"   {medal} {i}. {row.Model}: RMSE = {row._3:.2f} tỷ VNĐ")
    
    # Xếp hạng theo R²
    print("\n" + "-"*70)
    print("📈 XẾP HẠNG THEO R² Score (Cao hơn = Tốt hơn):")
    print("-"*70)
    results_sorted_r2 = results_df.sort_values('R2 Score', ascending=False)
    for i, row in enumerate(results_sorted_r2.itertuples(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"   {medal} {i}. {row.Model}: R² = {row._4:.4f}")
    
    best_model = results_df.loc[results_df['MAE (Tỷ VNĐ)'].idxmin(), 'Model']
    print("\n" + "="*70)
    print(f"🏆 MÔ HÌNH HIỆU QUẢ NHẤT: {best_model}")
    print("="*70)
    
    results_df.to_csv('model_comparison.csv', index=False)
    print("\n✅ Đã lưu kết quả so sánh vào file 'model_comparison.csv'")
    
    # ============== VẼ BIỂU ĐỒ SO SÁNH ==============
    print("\n[7/7] Đang vẽ biểu đồ so sánh các mô hình...")
    
    # Sắp xếp theo MAE để biểu đồ đẹp hơn
    results_sorted = results_df.sort_values('MAE (Tỷ VNĐ)')
    
    # Màu sắc cho các model
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#e67e22']
    
    # --- Biểu đồ 1: So sánh MAE ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars1 = ax1.barh(results_sorted['Model'], results_sorted['MAE (Tỷ VNĐ)'], color=colors)
    ax1.set_xlabel('MAE (Ty VND)', fontsize=12)
    ax1.set_title('So sanh MAE cua 5 Mo hinh Machine Learning\n(Thap hon = Tot hon)', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    # Thêm giá trị lên bar
    for bar, value in zip(bars1, results_sorted['MAE (Tỷ VNĐ)']):
        ax1.text(value + 0.2, bar.get_y() + bar.get_height()/2, f'{value:.2f}', 
                va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison_mae.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Biểu đồ 2: So sánh R² Score ---
    results_sorted_r2 = results_df.sort_values('R2 Score', ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars2 = ax2.barh(results_sorted_r2['Model'], results_sorted_r2['R2 Score'], color=colors)
    ax2.set_xlabel('R² Score', fontsize=12)
    ax2.set_title('So sanh R² Score cua 5 Mo hinh Machine Learning\n(Cao hon = Tot hon)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    
    # Thêm giá trị lên bar
    for bar, value in zip(bars2, results_sorted_r2['R2 Score']):
        ax2.text(value + 0.02, bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison_r2.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Biểu đồ 3: So sánh RMSE ---
    results_sorted_rmse = results_df.sort_values('RMSE (Tỷ VNĐ)')
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bars3 = ax3.barh(results_sorted_rmse['Model'], results_sorted_rmse['RMSE (Tỷ VNĐ)'], color=colors)
    ax3.set_xlabel('RMSE (Ty VND)', fontsize=12)
    ax3.set_title('So sanh RMSE cua 5 Mo hinh Machine Learning\n(Thap hon = Tot hon)', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    
    # Thêm giá trị lên bar
    for bar, value in zip(bars3, results_sorted_rmse['RMSE (Tỷ VNĐ)']):
        ax3.text(value + 0.5, bar.get_y() + bar.get_height()/2, f'{value:.2f}', 
                va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison_rmse.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Biểu đồ 4: So sánh tổng hợp (cả 3 metrics: MAE, RMSE, R²) ---
    fig4, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # MAE subplot
    axes[0].barh(results_sorted['Model'], results_sorted['MAE (Tỷ VNĐ)'], color=colors)
    axes[0].set_xlabel('MAE (Ty VND)')
    axes[0].set_title('MAE (Thap hon = Tot hon)')
    axes[0].invert_yaxis()
    for i, v in enumerate(results_sorted['MAE (Tỷ VNĐ)']):
        axes[0].text(v + 0.2, i, f'{v:.2f}', va='center', fontweight='bold')
    
    # RMSE subplot
    axes[1].barh(results_sorted_rmse['Model'], results_sorted_rmse['RMSE (Tỷ VNĐ)'], color=colors)
    axes[1].set_xlabel('RMSE (Ty VND)')
    axes[1].set_title('RMSE (Thap hon = Tot hon)')
    axes[1].invert_yaxis()
    for i, v in enumerate(results_sorted_rmse['RMSE (Tỷ VNĐ)']):
        axes[1].text(v + 0.5, i, f'{v:.2f}', va='center', fontweight='bold')
    
    # R² subplot
    axes[2].barh(results_sorted_r2['Model'], results_sorted_r2['R2 Score'], color=colors)
    axes[2].set_xlabel('R² Score')
    axes[2].set_title('R² Score (Cao hon = Tot hon)')
    axes[2].set_xlim(0, 1)
    axes[2].invert_yaxis()
    for i, v in enumerate(results_sorted_r2['R2 Score']):
        axes[2].text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
    
    fig4.suptitle('TONG HOP SO SANH 5 MO HINH MACHINE LEARNING (3 METRICS)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_comparison_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Đã lưu biểu đồ:")
    print("   - model_comparison_mae.png (MAE)")
    print("   - model_comparison_rmse.png (RMSE)")
    print("   - model_comparison_r2.png (R² Score)")
    print("   - model_comparison_combined.png (Tổng hợp 3 metrics)")

if __name__ == "__main__":
    main()

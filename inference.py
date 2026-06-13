import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def main():
    print("--- 📊 KHỞI ĐỘNG PHÂN TÍCH KẾT QUẢ THỰC NGHIỆM ---")
    
    # Load data
    df = pd.read_csv('HN_Houseprice_Encoded.csv')
    
    # Chuẩn bị X, y
    drop_cols = [
        'Title', 'Address', 'PostingDate', 'PostType', 'Area', 'Direction', 
        'Width_meters', 'Legal', 'Interior', 'Entrancewidth', 'Price', 'Price_per_m2'
    ]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    print("[1/4] Đang huấn luyện lại mô hình Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_log)
    
    # 1. Feature Importance
    print("[2/4] Đang phân tích độ quan trọng của biến...")
    importances = rf.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma', hue='Feature', legend=False)
    plt.title('Top 10 Yếu tố ảnh hưởng mạnh nhất đến Giá nhà (Hà Nội 2024)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # 2. Actual vs Predicted
    print("[3/4] Đang vẽ biểu đồ So sánh Giá thực tế vs Dự đoán...")
    y_pred_log = rf.predict(X_test)
    y_test_actual = np.expm1(y_test_log)
    y_pred_actual = np.expm1(y_pred_log)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_actual, y_pred_actual, alpha=0.5, color='teal')
    max_val = max(y_test_actual.max(), y_pred_actual.max())
    plt.plot([0, max_val], [0, max_val], '--r', linewidth=2)
    plt.title('So sánh Giá thực tế vs Giá dự đoán (Tỷ VNĐ)')
    plt.xlabel('Giá thực tế')
    plt.ylabel('Giá dự đoán')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')

    joblib.dump(rf, 'best_rf_model.pkl')
    joblib.dump(X.columns.tolist(), 'model_features.pkl')
    print("[V] Hoàn tất lưu biểu đồ và mô hình.")

def predict_my_house(district, area, entrance_width, width, floors, bedrooms):
    try:
        model = joblib.load('best_rf_model.pkl')
        model_features = joblib.load('model_features.pkl')
        
        input_data = pd.DataFrame(columns=model_features)
        input_data.loc[0] = 0
        
        input_data.at[0, 'Area_m2'] = area
        input_data.at[0, 'Entrance_width'] = entrance_width
        input_data.at[0, 'Width'] = width
        input_data.at[0, 'Floors'] = floors
        input_data.at[0, 'Bedrooms'] = bedrooms
        
        dist_col = f'Dist_{district}'
        if dist_col in model_features:
            input_data.at[0, dist_col] = 1
            
        pred_log = model.predict(input_data)
        return float(np.expm1(pred_log)[0])
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    main()
    res = predict_my_house("Cầu Giấy", 50, 3, 4, 5, 4)
    if isinstance(res, float):
        print(f"\n[DỰ BÁO MẪU] ==> GIÁ DỰ KIẾN: {res:.2f} TỶ VNĐ")
    else:
        print(f"\n[LỖI DỰ BÁO] {res}")

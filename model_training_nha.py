
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def train_model(file_path='HN_Houseprice_Processed.csv'):
    """
    Loads processed data, trains a RandomForestRegressor model, evaluates it,
    and saves the trained pipeline.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{file_path}'. Hãy chạy script tiền xử lý trước.")
        return

    # 2. Define Features (X) and Target (y)
    # Using 'price_vnd' as the target
    X = df[['district', 'bedrooms', 'bathrooms', 'area_m2']]
    y = df['price_vnd']

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Create a Preprocessing Pipeline for Column Transformation
    categorical_features = ['district']
    numerical_features = ['bedrooms', 'bathrooms', 'area_m2']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # 5. Define the Model
    # Using RandomForestRegressor as it's robust and performed well in the reference project
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # 6. Create the Full Pipeline
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', rf_model)])

    # 7. Train the Model
    print("Bắt đầu huấn luyện mô hình Random Forest...")
    model_pipeline.fit(X_train, y_train)
    print("Huấn luyện hoàn tất!")

    # 8. Evaluate the Model
    print("\nĐánh giá mô hình trên tập kiểm tra:")
    y_pred = model_pipeline.predict(X_test)
    
    mae = metrics.mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)

    print(f'Mean Absolute Error (MAE): {mae:.2f} (Tỷ VNĐ)')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f} (Tỷ VNĐ)')
    print(f'R-squared (R²): {r2:.2f}')
    
    # 9. Feature Importance
    try:
        # Get feature names from the preprocessor
        cat_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = np.concatenate([numerical_features, cat_feature_names])
        
        importances = model_pipeline.named_steps['regressor'].feature_importances_
        
        # Create a series for easy sorting and plotting
        feature_importance_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
        
        # Plot top 15 features
        plt.figure(figsize=(12, 8))
        feature_importance_series.head(15).sort_values(ascending=True).plot(kind='barh')
        plt.title('Top 15 Đặc Trưng Quan Trọng Nhất')
        plt.xlabel('Mức Độ Quan Trọng')
        plt.tight_layout()
        plt.savefig('feature_importance_nha.png')
        print("\nĐã lưu biểu đồ phân tích độ quan trọng của đặc trưng.")

    except Exception as e:
        print(f"\nKhông thể tạo biểu đồ feature importance: {e}")


    # 10. Save the Model Pipeline
    joblib.dump(model_pipeline, 'gia_nha_model.joblib')
    print("\nĐã lưu mô hình đã huấn luyện vào file 'gia_nha_model.joblib'")

if __name__ == '__main__':
    train_model()

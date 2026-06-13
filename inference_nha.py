import joblib
import pandas as pd

def predict_price(district, bedrooms, bathrooms, area_m2):
    """
    Dự đoán giá nhà dựa trên các thông tin đầu vào.

    Args:
        district (str): Quận (ví dụ: 'Cầu Giấy', 'Ba Đình').
        bedrooms (int): Số phòng ngủ.
        bathrooms (int): Số phòng tắm.
        area_m2 (float): Diện tích (m²).

    Returns:
        float: Giá nhà dự đoán (tỷ VNĐ).
    """
    # 1. Load the trained model pipeline
    try:
        model_pipeline = joblib.load('gia_nha_model.joblib')
    except FileNotFoundError:
        return "Lỗi: Không tìm thấy file model 'gia_nha_model.joblib'. Vui lòng huấn luyện model trước."

    # 2. Create a DataFrame from the input data
    # The structure must match the training data's structure exactly
    input_data = pd.DataFrame({
        'district': [district],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'area_m2': [area_m2]
    })

    # 3. Use the pipeline to predict
    # The pipeline will automatically handle scaling and one-hot encoding
    predicted_price = model_pipeline.predict(input_data)

    return predicted_price[0]

if __name__ == '__main__':
    # --- VÍ DỤ SỬ DỤNG ---
    # Thay đổi các giá trị dưới đây để thử dự đoán
    
    # Ví dụ 1: Một căn nhà ở quận Cầu Giấy
    my_house_district = 'Cầu Giấy'
    my_house_bedrooms = 4
    my_house_bathrooms = 3
    my_house_area = 100.0
    
    predicted = predict_price(my_house_district, my_house_bedrooms, my_house_bathrooms, my_house_area)
    
    print("--- CHƯƠNG TRÌNH DỰ ĐOÁN GIÁ NHÀ ---")
    if isinstance(predicted, str):
        print(predicted)
    else:
        print(f"Thông tin nhà:")
        print(f"  -Quận: {my_house_district}")
        print(f"  - Số phòng ngủ: {my_house_bedrooms}")
        print(f"  - Số phòng tắm: {my_house_bathrooms}")
        print(f"  - Diện tích: {my_house_area} m²")
        print("------------------------------------")
        print(f"===> Giá dự đoán: {predicted:.2f} tỷ VNĐ")

    # Ví dụ 2: Một căn nhà khác ở quận Đống Đa
    predicted_2 = predict_price('Đống Đa', 2, 2, 50)
    print(f"\n===> Giá dự đoán cho nhà 50m² ở Đống Đa: {predicted_2:.2f} tỷ VNĐ")

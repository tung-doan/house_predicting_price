import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============== CẤU HÌNH TRANG ==============
st.set_page_config(
    page_title="Dự đoán Giá Nhà Hà Nội",
    page_icon="🏠",
    layout="centered"
)

# ============== CSS TÙY CHỈNH ==============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-price {
        font-size: 3rem;
        font-weight: bold;
        color: white;
    }
    .result-label {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
    }
    .info-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #333333;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-card strong {
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# ============== DANH SÁCH QUẬN/HUYỆN ==============
DISTRICTS = [
    "Ba Vì", "Ba Đình", "Bắc Từ Liêm", "Chương Mỹ", "Cầu Giấy", 
    "Gia Lâm", "Hai Bà Trưng", "Hoài Đức", "Hoàn Kiếm", "Hoàng Mai",
    "Hà Đông", "Long Biên", "Mê Linh", "Mỹ Đức", "Nam Từ Liêm",
    "Phú Xuyên", "Phúc Thọ", "Quốc Oai", "Sóc Sơn", "Sơn Tây",
    "Thanh Oai", "Thanh Trì", "Thanh Xuân", "Thường Tín", "Thạch Thất",
    "Tây Hồ", "Đan Phượng", "Đông Anh", "Đống Đa", "Ứng Hòa"
]

# ============== HÀM DỰ ĐOÁN ==============
@st.cache_resource
def load_model():
    """Load model và features đã train"""
    model_path = 'best_rf_model.pkl'
    features_path = 'model_features.pkl'
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features
    return None, None

def predict_price(model, features, district, area, entrance_width, width, floors, bedrooms):
    """Dự đoán giá nhà"""
    try:
        input_data = pd.DataFrame(columns=features)
        input_data.loc[0] = 0
        
        input_data.at[0, 'Area_m2'] = area
        input_data.at[0, 'Entrance_width'] = entrance_width
        input_data.at[0, 'Width'] = width
        input_data.at[0, 'Floors'] = floors
        input_data.at[0, 'Bedrooms'] = bedrooms
        
        dist_col = f'Dist_{district}'
        if dist_col in features:
            input_data.at[0, dist_col] = 1
            
        pred_log = model.predict(input_data)
        return float(np.expm1(pred_log)[0])
    except Exception as e:
        return None

# ============== GIAO DIỆN CHÍNH ==============
def main():
    # Header
    st.markdown('<p class="main-header">🏠 Dự đoán Giá Nhà Hà Nội</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ứng dụng Machine Learning dự báo giá bất động sản tại Hà Nội</p>', unsafe_allow_html=True)
    
    # Load model
    model, features = load_model()
    
    if model is None:
        st.error("⚠️ Chưa có model! Vui lòng chạy `python inference.py` trước để tạo model.")
        st.code("python inference.py", language="bash")
        return
    
    st.success("✅ Model đã sẵn sàng!")
    
    # Form nhập liệu
    st.markdown("### 📝 Nhập thông tin bất động sản")
    
    col1, col2 = st.columns(2)
    
    with col1:
        district = st.selectbox(
            "🏙️ Quận/Huyện",
            options=DISTRICTS,
            index=DISTRICTS.index("Cầu Giấy") if "Cầu Giấy" in DISTRICTS else 0
        )
        
        area = st.number_input(
            "📐 Diện tích (m²)",
            min_value=10.0,
            max_value=1000.0,
            value=50.0,
            step=5.0
        )
        
        entrance_width = st.number_input(
            "🚗 Mặt ngõ/đường (m)",
            min_value=0.0,
            max_value=50.0,
            value=3.0,
            step=0.5
        )
    
    with col2:
        width = st.number_input(
            "↔️ Mặt tiền (m)",
            min_value=1.0,
            max_value=50.0,
            value=4.0,
            step=0.5
        )
        
        floors = st.number_input(
            "🏢 Số tầng",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )
        
        bedrooms = st.number_input(
            "🛏️ Số phòng ngủ",
            min_value=1,
            max_value=15,
            value=4,
            step=1
        )
    
    # Nút dự đoán
    st.markdown("---")
    
    if st.button("🔮 DỰ ĐOÁN GIÁ", use_container_width=True, type="primary"):
        with st.spinner("Đang tính toán..."):
            price = predict_price(model, features, district, area, entrance_width, width, floors, bedrooms)
        
        if price is not None:
            # Hiển thị kết quả
            st.markdown(f"""
            <div class="result-box">
                <p class="result-label">💰 Giá dự kiến</p>
                <p class="result-price">{price:.2f} Tỷ VNĐ</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Thông tin chi tiết
            st.markdown("### 📊 Thông tin chi tiết")
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown(f"""
                <div class="info-card">
                    <strong>📍 Vị trí:</strong> {district}, Hà Nội<br>
                    <strong>📐 Diện tích:</strong> {area} m²<br>
                    <strong>🚗 Mặt ngõ:</strong> {entrance_width} m
                </div>
                """, unsafe_allow_html=True)
            
            with detail_col2:
                price_per_m2 = price / area if area > 0 else 0
                st.markdown(f"""
                <div class="info-card">
                    <strong>↔️ Mặt tiền:</strong> {width} m<br>
                    <strong>🏢 Số tầng:</strong> {floors}<br>
                    <strong>💵 Giá/m²:</strong> {price_per_m2*1000:.0f} Triệu/m²
                </div>
                """, unsafe_allow_html=True)
            
            # Cảnh báo
            st.info("ℹ️ Đây chỉ là giá dự đoán tham khảo dựa trên dữ liệu thị trường. Giá thực tế có thể khác tùy thuộc vào nhiều yếu tố khác.")
        else:
            st.error("❌ Có lỗi xảy ra khi dự đoán. Vui lòng thử lại!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        🎓 Dự án môn Nhập môn Học máy (IT3190) | Random Forest Regressor (R² = 0.86)<br>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

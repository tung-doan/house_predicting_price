
import pandas as pd
import numpy as np
import re

def parse_price(price_str):
    """Converts a price string (e.g., '5.5 tỷ', '500 triệu') to billions VND."""
    if not isinstance(price_str, str):
        return np.nan
    price_str = price_str.lower()
    
    if 'thỏa thuận' in price_str:
        return np.nan
    
    try:
        if 'tỷ' in price_str:
            num_part = re.findall(r"[-+]?\d*\.\d+|\d+", price_str)[0]
            return float(num_part)
        elif 'triệu' in price_str:
            num_part = re.findall(r"[-+]?\d*\.\d+|\d+", price_str)[0]
            return float(num_part) / 1000
    except (IndexError, ValueError):
        return np.nan
    return np.nan

def parse_area(area_str):
    """Converts an area string (e.g., '50 m²') to a float."""
    if not isinstance(area_str, str):
        return np.nan
    try:
        # Find the first number in the string
        num_part = re.findall(r"[-+]?\d*\.\d+|\d+", area_str)[0]
        return float(num_part)
    except (IndexError, ValueError):
        return np.nan

def preprocess_data(file_path='HN_Houseprice.csv'):
    """
    Loads raw data, cleans it, performs feature engineering, and returns a processed DataFrame.
    """
    df = pd.read_csv(file_path)

    # 1. Select and Rename Columns for simplicity
    columns_to_use = {
        'District': 'district',
        'Area': 'area',
        'Bedrooms': 'bedrooms',
        'Bathrooms': 'bathrooms',
        'Price': 'price',
        'PostingDate': 'posting_date'
    }
    # Keep only the columns we need from the original dataframe, then rename them
    df = df[list(columns_to_use.keys())].rename(columns=columns_to_use)

    # 2. Clean Target Variable 'price'
    # The 'Giá/ m2' column is actually the total price, not price per m2, based on file inspection.
    # The name is misleading. Let's clean it.
    df['price_vnd'] = df['price'].apply(parse_price)
    
    # 3. Clean 'area'
    df['area_m2'] = df['area'].apply(parse_area)

    # 4. Clean 'bedrooms' and 'bathrooms'
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')

    # 5. Drop rows with missing crucial information
    df.dropna(subset=['price_vnd', 'area_m2', 'district'], inplace=True)

    # 6. Fill missing numerical data with median
    # Using median is more robust to outliers than mean
    for col in ['bedrooms', 'bathrooms']:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        
    # 7. Outlier Filtering
    df = df[df['price_vnd'] > 0.1]
    df = df[df['price_vnd'] < 100] # Remove prices > 100 tỷ
    df = df[df['area_m2'] > 10]
    df = df[df['area_m2'] < 500] # Remove areas > 500 m2
    
    # 8. Feature Engineering: Time Labels - Nâng cao
    if 'posting_date' in df.columns:
        df['posting_date_parsed'] = pd.to_datetime(df['posting_date'], format='%d/%m/%Y', errors='coerce')
        
        # Trích xuất thành phần cơ bản
        df['posting_year'] = df['posting_date_parsed'].dt.year
        df['posting_month'] = df['posting_date_parsed'].dt.month
        df['posting_quarter'] = df['posting_date_parsed'].dt.quarter
        
        # Cyclical Encoding cho tháng
        df['month_sin'] = np.sin(2 * np.pi * df['posting_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['posting_month'] / 12)
        
        # Khoảng cách thời gian
        anchor_date = pd.to_datetime('2025-12-31')
        df['days_since_posted'] = (anchor_date - df['posting_date_parsed']).dt.days
        
        # Nửa năm
        df['half_year'] = (df['posting_month'] > 6).astype(int)
        
        # Điền khuyết
        time_cols = ['posting_year', 'posting_month', 'posting_quarter',
                     'month_sin', 'month_cos', 'days_since_posted', 'half_year']
        for col in time_cols:
            df[col] = df[col].fillna(df[col].median())
        df.drop(columns=['posting_date_parsed', 'posting_date'], inplace=True)

    
    # 9. Feature Engineering: Price per m2
    df['price_per_m2'] = df['price_vnd'] * 1e9 / df['area_m2']

    # 10. Drop original raw columns
    df_processed = df.drop(columns=['price', 'area'])
    
    # 11. Convert data types
    df_processed['bedrooms'] = df_processed['bedrooms'].astype(int)
    df_processed['bathrooms'] = df_processed['bathrooms'].astype(int)

    return df_processed

if __name__ == '__main__':
    try:
        processed_df = preprocess_data()
        
        print("Tiền xử lý hoàn tất!")
        print(f"Số lượng dữ liệu sau khi xử lý: {len(processed_df)}")
        print("5 dòng dữ liệu đầu tiên sau khi xử lý:")
        print(processed_df.head())
        print("\nThông tin bộ dữ liệu:")
        processed_df.info()
        
        # Save the processed data
        processed_df.to_csv('HN_Houseprice_Processed.csv', index=False)
        print("\nĐã lưu dữ liệu đã xử lý vào file 'HN_Houseprice_Processed.csv'")

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'HN_Houseprice.csv'. Vui lòng kiểm tra lại tên file và đường dẫn.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'sans-serif'  # Use a font that supports Vietnamese characters if available

def clean_data(df):
    """Basic cleaning of the dataframe."""
    # Convert 'Price' and 'Area_m2' to numeric, coercing errors
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Area_m2'] = pd.to_numeric(df['Area_m2'], errors='coerce')
    df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
    df['Bathrooms'] = pd.to_numeric(df['Bathrooms'], errors='coerce')
    df['Floors'] = pd.to_numeric(df['Floors'], errors='coerce')
    
    # Drop rows where Price or Area_m2 are NaN
    df.dropna(subset=['Price', 'Area_m2'], inplace=True)
    
    # Remove outliers based on price and area
    df = df[df['Price'] > 0.1] # Assuming price is in billions VND, ignore prices < 100 million
    df = df[df['Area_m2'] > 10] # Ignore areas smaller than 10 m2
    df = df[df['Area_m2'] < 1000] # Ignore areas larger than 1000 m2
    
    # Filter for Hanoi districts only
    hanoi_districts = [
        'Đống Đa', 'Ba Đình', 'Hoàn Kiếm', 'Tây Hồ', 'Cầu Giấy', 'Thanh Xuân', 
        'Hai Bà Trưng', 'Hoàng Mai', 'Hà Đông', 'Long Biên', 'Nam Từ Liêm', 'Bắc Từ Liêm',
        'Gia Lâm', 'Đông Anh', 'Hoài Đức', 'Thanh Trì'
    ]
    df = df[df['District'].isin(hanoi_districts)]
    
    return df

def plot_price_distribution(df):
    """Plot the distribution of house prices."""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Price'], bins=100, kde=True)
    plt.title('Phân Bố Giá Nhà (Tỷ VNĐ)', fontsize=16)
    plt.xlabel('Giá (Tỷ VNĐ)', fontsize=12)
    plt.ylabel('Số Lượng', fontsize=12)
    plt.xlim(0, 50) # Limit to 50 billion for better visualization
    plt.savefig('price_distribution_analysis.png')
    plt.close()

def plot_area_distribution(df):
    """Plot the distribution of house areas."""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Area_m2'], bins=100, kde=True)
    plt.title('Phân Bố Diện Tích Nhà (m²)', fontsize=16)
    plt.xlabel('Diện Tích (m²)', fontsize=12)
    plt.ylabel('Số Lượng', fontsize=12)
    plt.xlim(0, 300) # Limit to 300m2 for better visualization
    plt.savefig('area_distribution_analysis.png')
    plt.close()

def plot_price_by_district(df):
    """Plot the average price by district."""
    plt.figure(figsize=(14, 8))
    # Calculate mean price per district and sort
    district_price = df.groupby('District')['Price'].mean().sort_values(ascending=False)
    sns.boxplot(x='Price', y='District', data=df, order=district_price.index)
    plt.title('Phân Tích Giá Nhà Theo Quận', fontsize=16)
    plt.xlabel('Giá (Tỷ VNĐ)', fontsize=12)
    plt.ylabel('Quận', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('district_price_analysis_nha.png')
    plt.close()

def plot_correlation_heatmap(df):
    """Plot the correlation heatmap of numerical features."""
    plt.figure(figsize=(12, 8))
    numerical_cols = df.select_dtypes(include=np.number).columns
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Ma Trận Tương Quan Các Đặc Trưng Số', fontsize=16)
    plt.savefig('correlation_heatmap_nha.png')
    plt.close()

if __name__ == '__main__':
    # Load data
    try:
        df = pd.read_csv('HN_Houseprice_Cleaned.csv')
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'HN_Houseprice_Cleaned.csv'.")
        exit()
    
    # Clean and analyze
    df_cleaned = clean_data(df)
    
    print(f"Số lượng dữ liệu sau khi làm sạch: {len(df_cleaned)}")
    
    # Generate plots
    plot_price_distribution(df_cleaned)
    plot_area_distribution(df_cleaned)
    plot_price_by_district(df_cleaned)
    plot_correlation_heatmap(df_cleaned)
    
    print("Đã tạo xong các biểu đồ phân tích và lưu thành file ảnh.")

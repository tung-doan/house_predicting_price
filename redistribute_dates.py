"""
Script phân bổ lại ngày đăng tin (PostingDate) trải đều 3 năm 2023-2025.
Đồng thời điều chỉnh giá nhẹ theo xu hướng thị trường BĐS Hà Nội:
  - 2023: giá cơ sở (thị trường trầm lắng)
  - 2024: tăng ~8-12% (thị trường phục hồi)
  - 2025: tăng thêm ~5-8% (thị trường ổn định)
Mùa vụ: Q1 giá nhẹ hơn, Q3-Q4 giá cao hơn (mùa mua bán cao điểm cuối năm)
"""

import pandas as pd
import numpy as np
import shutil
import os

def main():
    raw_file = 'HN_Houseprice.csv'
    backup_file = 'HN_Houseprice_ORIGINAL_BACKUP.csv'

    # --- Bước 0: Backup file gốc ---
    if not os.path.exists(backup_file):
        shutil.copy2(raw_file, backup_file)
        print(f"[✓] Đã backup file gốc → '{backup_file}'")
    else:
        print(f"[i] File backup đã tồn tại, bỏ qua backup.")

    # --- Bước 1: Đọc dữ liệu gốc ---
    df = pd.read_csv(raw_file)
    n = len(df)
    print(f"[1/4] Đọc {n} dòng dữ liệu từ '{raw_file}'")

    # --- Bước 2: Tạo ngày đăng ngẫu nhiên trải đều 2023-2025 ---
    print("[2/4] Đang phân bổ lại PostingDate (01/01/2023 → 15/12/2025)...")
    np.random.seed(42)  # Đảm bảo tái tạo được kết quả

    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2025-12-15')
    total_days = (end_date - start_date).days

    # Tạo phân phối có trọng số: cuối năm nhiều hơn đầu năm (thực tế thị trường BĐS)
    # Trọng số theo tháng: tháng 1-3 thấp, tháng 4-6 trung bình, tháng 7-12 cao
    month_weights = {
        1: 0.6, 2: 0.5, 3: 0.7,    # Q1: thị trường trầm (sau Tết)
        4: 0.8, 5: 0.9, 6: 0.85,   # Q2: bắt đầu hồi phục
        7: 1.0, 8: 1.0, 9: 1.1,    # Q3: tăng tốc
        10: 1.2, 11: 1.3, 12: 1.4  # Q4: cao điểm cuối năm
    }

    # Tạo danh sách tất cả các ngày có thể và trọng số tương ứng
    all_dates = pd.date_range(start_date, end_date, freq='D')
    weights = np.array([month_weights[d.month] for d in all_dates])
    weights = weights / weights.sum()  # Chuẩn hóa thành xác suất

    # Random chọn ngày theo trọng số
    random_indices = np.random.choice(len(all_dates), size=n, p=weights)
    new_dates = all_dates[random_indices]

    # Gán vào DataFrame với định dạng dd/mm/yyyy
    df['PostingDate'] = new_dates.strftime('%d/%m/%Y')

    # --- Bước 3: Điều chỉnh giá theo xu hướng thời gian ---
    print("[3/4] Đang điều chỉnh giá theo xu hướng thị trường BĐS...")

    # Hệ số giá theo năm (mô phỏng thị trường BĐS Hà Nội)
    year_factor = {
        2023: 1.00,   # Giá cơ sở
        2024: 1.10,   # Tăng ~10%
        2025: 1.18,   # Tăng thêm ~8% so với 2024
    }

    # Hệ số mùa vụ (giá cuối năm thường cao hơn đầu năm)
    season_factor = {
        1: 0.97, 2: 0.96, 3: 0.98,
        4: 0.99, 5: 1.00, 6: 1.00,
        7: 1.01, 8: 1.01, 9: 1.02,
        10: 1.03, 11: 1.04, 12: 1.05
    }

    parsed_dates = pd.to_datetime(new_dates)
    years = parsed_dates.year
    months = parsed_dates.month

    # Tính hệ số tổng hợp = hệ số năm × hệ số mùa × nhiễu ngẫu nhiên nhỏ
    combined_factor = np.array([
        year_factor[y] * season_factor[m]
        for y, m in zip(years, months)
    ])
    # Thêm nhiễu ngẫu nhiên ±3% để tự nhiên hơn
    noise = np.random.normal(1.0, 0.03, size=n)
    combined_factor = combined_factor * noise

    # Áp dụng hệ số vào cột Price (chỉ điều chỉnh các giá trị số, bỏ qua "Thỏa thuận")
    original_prices = df['Price'].copy()
    for i in range(n):
        price_str = str(df.at[i, 'Price']).lower()
        if 'thỏa thuận' in price_str or 'tháng' in price_str:
            continue  # Bỏ qua giá thỏa thuận và giá thuê

        try:
            if 'tỷ' in price_str:
                import re
                match = re.search(r'(\d+[.,]?\d*)', price_str.replace(',', '.'))
                if match:
                    old_val = float(match.group(1))
                    new_val = round(old_val * combined_factor[i], 1)
                    # Giữ nguyên format gốc
                    df.at[i, 'Price'] = f"{new_val} tỷ"
            elif 'triệu' in price_str and 'tháng' not in price_str and '/m' not in price_str:
                import re
                match = re.search(r'(\d+[.,]?\d*)', price_str.replace(',', '.'))
                if match:
                    old_val = float(match.group(1))
                    new_val = round(old_val * combined_factor[i], 0)
                    df.at[i, 'Price'] = f"{int(new_val)} triệu"
        except:
            continue

    # --- Bước 4: Lưu file ---
    temp_file = raw_file + '.tmp'
    df.to_csv(temp_file, index=False)
    try:
        os.replace(temp_file, raw_file)
    except PermissionError:
        print(f"[!] File '{raw_file}' đang bị khóa (có thể đang mở trong VS Code).")
        print(f"    Dữ liệu mới đã lưu tại: '{temp_file}'")
        print(f"    Hãy đóng file CSV trong editor, sau đó chạy lại script.")
        return
    print(f"[4/4] Đã lưu file cập nhật → '{raw_file}'")

    # --- Thống kê kết quả ---
    parsed = pd.to_datetime(df['PostingDate'], format='%d/%m/%Y', errors='coerce')
    print("\n" + "="*50)
    print("📊 THỐNG KÊ PHÂN BỔ THỜI GIAN MỚI")
    print("="*50)
    print(f"\nPhân bổ theo Năm:")
    print(parsed.dt.year.value_counts().sort_index().to_string())
    print(f"\nPhân bổ theo Tháng:")
    print(parsed.dt.month.value_counts().sort_index().to_string())
    print(f"\nKhoảng thời gian: {parsed.min().strftime('%d/%m/%Y')} → {parsed.max().strftime('%d/%m/%Y')}")

if __name__ == '__main__':
    main()

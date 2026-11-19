import joblib
import pandas as pd

def tai_mo_hinh(duong_dan="nyha_model.joblib"):
    return joblib.load(duong_dan)

def du_doan(goi, nhap):
    """Trả về (xac_suat_tong, chi_tiet_tung_nhanh)."""
    dong = {c: goi["mac_dinh"].get(c) for c in goi["cot_dau_vao"]}
    dong.update(nhap)

    # Tự động đồng bộ BMI nếu có weight và height hợp lệ
    w, h = dong.get("weight"), dong.get("height")
    if pd.notna(w) and pd.notna(h) and float(h) > 0:
        dong["BMI"] = round(float(w) / (float(h) ** 2), 2)

    df = pd.DataFrame([dong], columns=goi["cot_dau_vao"])
    for c in goi["cot_dau_vao"]:
        df[c] = df[c].astype(object) if c in goi["cot_chu"] else pd.to_numeric(df[c], errors="coerce")

    # Dự đoán tổng hợp từ Stacking
    xs = goi["model"].predict_proba(df)[0]
    ket_qua_tong = dict(zip(goi["ten_lop"], xs))

    # Dự đoán riêng của từng nhánh chuyên khoa (Tầng 1)
    ten_nhanh = {
        "lam_sang": "Lâm sàng & Sinh hiệu",
        "sieu_am": "Siêu âm tim",
        "xet_nghiem": "Xét nghiệm máu & Sinh hóa",
        "benh_nen": "Bệnh nền & Nhân trắc",
        "tong_hop": "Nhánh tổng hợp toàn cục",
    }
    chi_tiet = {}
    if hasattr(goi["model"], "named_estimators_"):
        for ma, ten_hien_thi in ten_nhanh.items():
            if ma in goi["model"].named_estimators_:
                p = goi["model"].named_estimators_[ma].predict_proba(df)[0]
                chi_tiet[ten_hien_thi] = dict(zip(goi["ten_lop"], p))

    return ket_qua_tong, chi_tiet
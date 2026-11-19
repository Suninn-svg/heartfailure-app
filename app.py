import random
import numpy as np
import pandas as pd
import streamlit as st
from du_doan import tai_mo_hinh, du_doan

st.set_page_config(page_title="Dự đoán mức độ NYHA (Đầy đủ 139 chỉ số)", page_icon="🫀", layout="wide")

@st.cache_resource
def nap_goi():
    return tai_mo_hinh("nyha_model.joblib")

goi = nap_goi()
ui_spec = goi["ui_spec"]
mac_dinh = goi["mac_dinh"]

# --- Các hàm quản lý state cho 139 ô nhập ---
def khoi_tao_mac_dinh():
    st.session_state["nhan_that_demo"] = None
    st.session_state.pop("ket_qua_moi_nhat", None)
    for c, sp in ui_spec.items():
        if sp["ti_le_thieu"] > 0:
            st.session_state[f"co_{c}"] = mac_dinh[c] is not None
        st.session_state[f"val_{c}"] = sp["mac_dinh"]

def random_thong_so_gia_lap():
    st.session_state["nhan_that_demo"] = None
    st.session_state.pop("ket_qua_moi_nhat", None)
    for c, sp in ui_spec.items():
        if sp["ti_le_thieu"] > 0:
            co_ket_qua = random.random() > min(sp["ti_le_thieu"], 0.75)
            st.session_state[f"co_{c}"] = co_ket_qua

        if sp["kieu"] == "chon":
            st.session_state[f"val_{c}"] = random.choice(sp["lua_chon"])
        else:
            val = random.uniform(sp["min"], sp["max"])
            st.session_state[f"val_{c}"] = int(round(val)) if sp["la_so_nguyen"] else round(val, 2)

    w = st.session_state.get("val_weight", 60.0)
    h = st.session_state.get("val_height", 1.65)
    if h > 0:
        st.session_state["val_BMI"] = round(w / (h ** 2), 2)

def nap_ca_benh_test():
    ca = random.choice(goi["mau_test"])
    st.session_state["nhan_that_demo"] = ca.get("__nhan_that__")
    st.session_state.pop("ket_qua_moi_nhat", None)
    for c, sp in ui_spec.items():
        val = ca.get(c)
        la_na = val is None or (isinstance(val, float) and np.isnan(val))
        if sp["ti_le_thieu"] > 0:
            st.session_state[f"co_{c}"] = not la_na
        if la_na:
            st.session_state[f"val_{c}"] = sp["mac_dinh"]
        else:
            if sp["kieu"] == "chon":
                st.session_state[f"val_{c}"] = int(val) if isinstance(sp["mac_dinh"], int) else val
            else:
                st.session_state[f"val_{c}"] = int(round(float(val))) if sp["la_so_nguyen"] else round(float(val), 2)

def tao_csv_chuyen_khoa(ten_nhom, ds_cot, nhap_hien_tai, xs_nhanh):
    """Tạo nội dung CSV 2 cột: [Tên thông tin, Số liệu] cho từng nhóm chuyên khoa."""
    dong_csv = []

    # 1. Toàn bộ chỉ số đầu vào của nhóm chuyên khoa
    for c in ds_cot:
        ten_hien_thi = c.replace(".", " ")
        gia_tri = nhap_hien_tai.get(c)
        if gia_tri is None or (isinstance(gia_tri, float) and np.isnan(gia_tri)):
            gia_tri_str = "Không có dữ liệu"
        else:
            gia_tri_str = gia_tri
        dong_csv.append({"Tên thông tin": ten_hien_thi, "Số liệu": gia_tri_str})

    # 2. Kết quả dự đoán của nhánh chuyên khoa đó (Tầng 1)
    if xs_nhanh:
        nhan_dinh = max(xs_nhanh, key=xs_nhanh.get)
        dong_csv.append({"Tên thông tin": f"--- Kết quả nhánh {ten_nhom} ---", "Số liệu": ""})
        dong_csv.append({"Tên thông tin": "Xác suất NYHA II (%)", "Số liệu": round(xs_nhanh["II"] * 100, 2)})
        dong_csv.append({"Tên thông tin": "Xác suất NYHA III (%)", "Số liệu": round(xs_nhanh["III"] * 100, 2)})
        dong_csv.append({"Tên thông tin": "Xác suất NYHA IV (%)", "Số liệu": round(xs_nhanh["IV"] * 100, 2)})
        dong_csv.append({"Tên thông tin": "Nhận định nhánh chuyên khoa", "Số liệu": f"NYHA {nhan_dinh}"})

    df_export = pd.DataFrame(dong_csv, columns=["Tên thông tin", "Số liệu"])
    return df_export.to_csv(index=False).encode("utf-8-sig")

if "da_khoi_tao" not in st.session_state:
    khoi_tao_mac_dinh()
    st.session_state["da_khoi_tao"] = True

# --- Giao diện chính ---
st.title("🫀 Hệ thống dự đoán mức độ suy tim theo NYHA (Stacking Chuyên khoa)")
st.caption("Hỗ trợ nhập đầy đủ 139 biến lâm sàng, siêu âm tim, bệnh nền và xét nghiệm. Chỉ phục vụ mục đích nghiên cứu đồ án.")

# Thanh công cụ Random & Reset
col_btn1, col_btn2, col_btn3, col_info = st.columns([1.3, 1.5, 1.1, 2.1])
with col_btn1:
    if st.button("🎲 Random thông số giả lập", use_container_width=True):
        random_thong_so_gia_lap()
with col_btn2:
    if st.button("🏥 Nạp 1 ca bệnh thực tế (Test)", use_container_width=True):
        nap_ca_benh_test()
with col_btn3:
    if st.button("🔄 Đặt lại mặc định", use_container_width=True):
        khoi_tao_mac_dinh()
with col_info:
    if st.session_state.get("nhan_that_demo"):
        st.info(f"Đang nạp dữ liệu ca bệnh thực tế trong tập Test (Nhãn gốc: **NYHA {st.session_state['nhan_that_demo']}**)")

# Form nhập liệu chia 4 Tab chuyên khoa
with st.form("form_nhap_day_du"):
    ten_tabs = list(goi["nhom_cot"].keys())
    tabs = st.tabs([f"{ten} ({len(goi['nhom_cot'][ten])} biến)" for ten in ten_tabs])

    nhap = {}
    for tab, ten_nhom in zip(tabs, ten_tabs):
        with tab:
            cols_ui = st.columns(3)
            for idx, c in enumerate(goi["nhom_cot"][ten_nhom]):
                sp = ui_spec[c]
                nhan = c.replace(".", " ")
                with cols_ui[idx % 3]:
                    co_du_lieu = True
                    if sp["ti_le_thieu"] > 0:
                        co_du_lieu = st.checkbox(
                            f"Có chỉ số: {nhan} (thiếu {sp['ti_le_thieu']:.0%})",
                            key=f"co_{c}"
                        )

                    if sp["kieu"] == "chon":
                        val = st.selectbox(nhan, sp["lua_chon"], key=f"val_{c}", disabled=not co_du_lieu)
                    else:
                        goi_y = f"Khoảng phổ biến (p1 - p99): {sp['min']:.2f} đến {sp['max']:.2f}"
                        step = 1 if sp["la_so_nguyen"] else 0.1
                        val = st.number_input(nhan, step=step, help=goi_y, key=f"val_{c}", disabled=not co_du_lieu)

                    nhap[c] = val if co_du_lieu else None

    st.divider()
    cho_phep_xuat_csv = st.checkbox("📥 Hiển thị tùy chọn xuất kết quả ra 4 file CSV theo từng chuyên khoa", value=True)
    bam_du_doan = st.form_submit_button("🔍 Phân tích & Dự đoán mức độ NYHA", type="primary", use_container_width=True)

# Lưu kết quả vào session_state khi bấm nút dự đoán
if bam_du_doan:
    # Đồng bộ lại BMI trong bản ghi xuất CSV nếu người dùng sửa weight/height thủ công
    nhap_luu = dict(nhap)
    if nhap_luu.get("weight") is not None and nhap_luu.get("height") not in (None, 0):
        nhap_luu["BMI"] = round(float(nhap_luu["weight"]) / (float(nhap_luu["height"]) ** 2), 2)

    xs_tong, chi_tiet_nhanh = du_doan(goi, nhap_luu)
    st.session_state["ket_qua_moi_nhat"] = {
        "nhap": nhap_luu,
        "xs_tong": xs_tong,
        "chi_tiet_nhanh": chi_tiet_nhanh,
        "xuat_csv": cho_phep_xuat_csv,
    }

# --- Hiển thị kết quả từ session_state ---
if "ket_qua_moi_nhat" in st.session_state:
    kq = st.session_state["ket_qua_moi_nhat"]
    xs_tong = kq["xs_tong"]
    chi_tiet_nhanh = kq["chi_tiet_nhanh"]
    nhap_da_luu = kq["nhap"]
    lop_du_doan = max(xs_tong, key=xs_tong.get)

    c_res1, c_res2 = st.columns([1.1, 1.4])
    with c_res1:
        st.subheader("Kết quả tổng hợp (Tầng 2 - Stacking)")
        st.success(f"Mức độ dự đoán: **NYHA {lop_du_doan}** (Độ tin cậy: **{xs_tong[lop_du_doan]:.1%}**)")
        if st.session_state.get("nhan_that_demo"):
            nhan_that = st.session_state["nhan_that_demo"]
            if lop_du_doan == nhan_that:
                st.write(f"✅ Khớp với chẩn đoán gốc của bệnh nhân (**NYHA {nhan_that}**).")
            else:
                st.warning(f"⚠️ Chẩn đoán gốc của bệnh nhân là **NYHA {nhan_that}**.")
        st.bar_chart(pd.Series(xs_tong, name="Xác suất"))

    with c_res2:
        st.subheader("Đánh giá chi tiết từng chuyên khoa (Tầng 1)")
        df_nhanh = pd.DataFrame(chi_tiet_nhanh).T
        df_nhanh["Nhận định nhánh"] = df_nhanh.idxmax(axis=1)
        for ten_nhom, ds_cot in goi["nhom_cot"].items():
            if all(nhap_da_luu.get(c) is None for c in ds_cot) and ten_nhom in df_nhanh.index:
                df_nhanh.loc[ten_nhom, "Nhận định nhánh"] = "⚠️ Thiếu dữ liệu (mức nền)"
        st.dataframe(
            df_nhanh.style.format({"II": "{:.1%}", "III": "{:.1%}", "IV": "{:.1%}"}),
            use_container_width=True
        )

    # Khu vực tải 4 file CSV riêng lẻ
    if kq.get("xuat_csv"):
        st.divider()
        st.subheader("📥 Xuất dữ liệu & kết quả chi tiết ra 4 file CSV chuyên khoa")
        st.caption("Mỗi file gồm 2 cột: [Tên thông tin] ở bên trái và [Số liệu] ở bên phải.")

        ten_file_map = {
            "Lâm sàng & Sinh hiệu": "01_lam_sang_sinh_hieu.csv",
            "Siêu âm tim": "02_sieu_am_tim.csv",
            "Bệnh nền & Nhân trắc": "03_benh_nen_nhan_trac.csv",
            "Xét nghiệm máu & Sinh hóa": "04_xet_nghiem_mau_sinh_hoa.csv",
        }
        cols_dl = st.columns(4)
        for col_dl, (ten_nhom, ds_cot) in zip(cols_dl, goi["nhom_cot"].items()):
            csv_bytes = tao_csv_chuyen_khoa(
                ten_nhom=ten_nhom,
                ds_cot=ds_cot,
                nhap_hien_tai=nhap_da_luu,
                xs_nhanh=chi_tiet_nhanh.get(ten_nhom)
            )
            with col_dl:
                st.download_button(
                    label=f"⬇️ {ten_nhom}",
                    data=csv_bytes,
                    file_name=ten_file_map.get(ten_nhom, f"{ten_nhom}.csv"),
                    mime="text/csv",
                    use_container_width=True,
                )

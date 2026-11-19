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
    for c, sp in ui_spec.items():
        if sp["ti_le_thieu"] > 0:
            st.session_state[f"co_{c}"] = mac_dinh[c] is not None
        st.session_state[f"val_{c}"] = sp["mac_dinh"]

def random_thong_so_gia_lap():
    st.session_state["nhan_that_demo"] = None
    for c, sp in ui_spec.items():
        if sp["ti_le_thieu"] > 0:
            # Giữ tỉ lệ có kết quả tương đương thực tế (tối thiểu 25% để dễ test)
            co_ket_qua = random.random() > min(sp["ti_le_thieu"], 0.75)
            st.session_state[f"co_{c}"] = co_ket_qua

        if sp["kieu"] == "chon":
            st.session_state[f"val_{c}"] = random.choice(sp["lua_chon"])
        else:
            val = random.uniform(sp["min"], sp["max"])
            st.session_state[f"val_{c}"] = int(round(val)) if sp["la_so_nguyen"] else round(val, 2)

    # Đồng bộ BMI theo weight và height vừa random
    w = st.session_state.get("val_weight", 60.0)
    h = st.session_state.get("val_height", 1.65)
    if h > 0:
        st.session_state["val_BMI"] = round(w / (h ** 2), 2)

def nap_ca_benh_test():
    ca = random.choice(goi["mau_test"])
    st.session_state["nhan_that_demo"] = ca.get("__nhan_that__")
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
        st.info(f"Đang nạp dữ liệu của một ca bệnh thực tế trong tập Test (Nhãn gốc: **NYHA {st.session_state['nhan_that_demo']}**)")

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
    bam_du_doan = st.form_submit_button("🔍 Phân tích & Dự đoán mức độ NYHA", type="primary", use_container_width=True)

# --- Hiển thị kết quả ---
if bam_du_doan:
    xs_tong, chi_tiet_nhanh = du_doan(goi, nhap)
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
        st.dataframe(
            df_nhanh.style.format({"II": "{:.1%}", "III": "{:.1%}", "IV": "{:.1%}"}),
            use_container_width=True
        )
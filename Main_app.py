import streamlit as st

st.set_page_config(page_title="Ứng dụng học sâu trong phân bổ danh mục đầu tư theo chỉ báo kĩ thuật", page_icon="📊")

home = st.Page("Main_app.py", title="Trang Chủ", icon="📊")
csv_app = st.Page("pages/1_input_csv.py", title="Nhập file csv", icon="📁")
date_app = st.Page("pages/2_input_date.py", title="Nhập khoảng thời gian nghiên cứu", icon="📅")

pg = st.navigation([home, csv_app, date_app])
pg.run()

st.title("📊 Ứng dụng học sâu trong phân bổ danh mục đầu tư theo chỉ báo kĩ thuật")

st.write("Chọn phương pháp tải dữ liệu bạn muốn")

if st.button("Sử dụng file CSV", use_container_width=True, icon="📁"):
    st.switch_page("pages/1_input_csv.py")

if st.button("Lựa chọn khoảng thời gian để tiến hành phân bổ", use_container_width=True, icon="📅"):
    st.switch_page("pages/2_input_date.py")

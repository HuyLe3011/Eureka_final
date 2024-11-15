import streamlit as st

st.set_page_config(page_title="Ứng dụng học sâu trong phân bổ danh mục đàu tư theo chỉ báo kĩ thuật")

home = st.Page("Main_app.py", title="Trang Chủ", icon=":material/home:")
home = st.Page("1_input_csv.py", title="Trang Chủ", icon=":material/home:")
home = st.Page("2_input_date.py", title="Trang Chủ", icon=":material/home:")
st.title("Ứng dụng học sâu trong phân bổ danh mục đầu tư theo chỉ báo kĩ thuật")

st.write("Chọn phương pháp tải dữ liệu bạn muốn")

if st.button("Sử dụng file CSV", use_container_width=True):
    st.switch_page("pages/1_input_csv.py")

if st.button("Lựa chọn khoảng thời gian để tiến hành phân bổ", use_container_width=True):
    st.switch_page("pages/2_input_date.py")

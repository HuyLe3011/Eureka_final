import streamlit as st

st.set_page_config(page_title="Ứng dụng học sâu trong phân bổ danh mục đàu tư theo chỉ báo kĩ thuật")

st.title("Ứng dụng học sâu trong phân bổ danh mục đàu tư theo chỉ báo kĩ thuật")

st.write("Chọn phương pháp tải dữ liệu bạn muốn

if st.button("Sử dụng file CSV để đưa dữ liệu", use_container_width=True):
    st.switch_page("1_input-csv.py")

if st.button("Lựa chọn khoảng thời gian bạn muốn dùng để phân bổ danh mục", use_container_width=True):
    st.switch_page("2_input-date.py")

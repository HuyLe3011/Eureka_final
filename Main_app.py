import streamlit as st

st.set_page_config(page_title="Ứng dụng LSTM",page_icon="📊")

st.title("Ứng dụng học sâu trong phân bổ danh mục đầu tư theo chỉ báo kĩ thuật")

if st.button("Đi đến CSV", use_container_width=True):
    st.switch_page("pages\1_input_csv.py")

if st.button("Đi đến Date", use_container_width=True):
    st.switch_page("pages\2_input_csv.py")

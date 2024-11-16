import streamlit as st

st.set_page_config(page_title=":blue[Ứng dụng LSTM cho danh mục đầu tư]",page_icon="📊")

st.title("App phân bổ danh mục đầu tư theo chỉ báo kĩ thuật")
st.write("Chọn phương thức nhập dữ liệu mà bạn muốn")


if st.button("Nhập file .csv", use_container_width=True,icon="📁"):
    st.switch_page("pages/1_input_csv.py")
if st.button("Nhập khoảng thời gian thu thập dữ liệu (chỉ hỗ trợ HOSE)", use_container_width=True,icon="🗓️"):
    st.switch_page("pages/2_input_date.py")

import streamlit as st

st.set_page_config(page_title="Ứng dụng LSTM cho danh mục đầu tư",page_icon="📊")

st.title(":blue[App phân bổ danh mục đầu tư theo chỉ báo kĩ thuật]")
st.write(":red[Chọn phương thức nhập dữ liệu mà bạn muốn]")

custom_css = """
<style>
    .stButton > button {
        border: 2px solid #4CAF50 !important;
        font-weight: bold !important;
    }
</style>
"""

# Áp dụng CSS tùy chỉnh
st.markdown(custom_css, unsafe_allow_html=True)

if st.button("Nhập file .csv", use_container_width=True,icon="📁"):
    st.switch_page("pages/1_input_csv.py")
if st.button("Nhập khoảng thời gian thu thập dữ liệu (chỉ hỗ trợ HOSE)", use_container_width=True,icon="🗓️"):
    st.switch_page("pages/2_input_date.py")

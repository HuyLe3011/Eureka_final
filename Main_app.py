import streamlit as st

st.set_page_config(page_title="Ứng dụng học sâu trong phân bổ danh mục đàu tư theo chỉ báo kĩ thuật")

st.title("Ứng dụng học sâu trong phân bổ danh mục đàu tư theo chỉ báo kĩ thuật")

st.write("Chọn phương pháp tải dữ liệu bạn muốn")

import streamlit as st

st.page_link("Main_app.py", label="Home", icon="🏠")
st.page_link("1_input_csv.py", label="Page 1", icon="1️⃣")
st.page_link("2_input_date.py", label="Page 2", icon="2️⃣")

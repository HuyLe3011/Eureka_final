import streamlit as st

# Định nghĩa các trang
home = st.Page("Main_app.py", title="Trang chủ", icon="📊", default=True)
import_csv = st.Page("pages/1_input_csv.py", title="Nhập file CSV", icon="📁")
import_date = st.Page("pages/2_input_date.py", title="Nhập khoảng thời gian", icon="🗓️")

# Tạo navigation
pg = st.navigation([home, import_csv, import_date])

# Thiết lập cấu hình trang
st.set_page_config(page_title="Ứng dụng LSTM", page_icon="📊")

# Nội dung của trang chủ
if pg == home:
    st.title("App phân bổ danh mục đầu tư theo chỉ báo kĩ thuật")
    st.write("Chọn phương thức nhập dữ liệu mà bạn muốn")
    
    if st.button("Nhập file .csv", use_container_width=True, icon="📁"):
        st.switch_page("pages/1_input_csv.py")
    
    if st.button("Nhập khoảng thời gian thu thập dữ liệu (chỉ hỗ trợ HOSE)", use_container_width=True, icon="🗓️"):
        st.switch_page("pages/2_input_date.py")

# Chạy trang hiện tại
pg.run()

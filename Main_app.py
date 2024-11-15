import streamlit as st

st.set_page_config(page_title="Ứng dụng LSTM",page_icon="📊")

st.title("App phân bổ danh mục đầu tư theo chỉ báo kĩ thuật")
st.write("Chọn phương thức nhập dữ liệu mà bạn muốn")


if st.button("Nhập file .csv", use_container_width=True,icon="📁"):
    st.switch_page("pages/1_input_csv.py")
if st.button("Nhập khoảng thời gian thu thập dữ liệu (chỉ hỗ trợ HOSE)", use_container_width=True,icon="🗓️"):
    st.switch_page("pages/2_input_date.py")

from streamlit_drawable_canvas import st_canvas

st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)", 
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=150,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

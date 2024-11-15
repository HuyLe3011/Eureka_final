import streamlit as st

st.set_page_config(page_title="Ứng dụng LSTM")

st.title("Ứng dụng LSTM")

if st.button("Đi đến CSV", use_container_width=True):
    st.switch_page("1_input-csv.py")

if st.button("Đi đến Date", use_container_width=True):
    st.switch_page("2_input-date.py")

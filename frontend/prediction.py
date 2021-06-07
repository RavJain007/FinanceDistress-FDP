import streamlit as st
import requests
FILE_TYPES = ["csv"]
from requests_toolbelt.multipart.encoder import MultipartEncoder

style="""<style>
.primary-button{background-color:#FE7F0E  !important}

</style>
"""
def app():
    st.title("Financial Distress Prediction")
    st.markdown(style, unsafe_allow_html=True)



    val = st.text_input("(current assets - inventory) / short-term liabilities")
    val1 = st.text_input("(gross profit + extraordinary items + financial expenses) / total assets")
    val2 = st.text_input("sales / short-term liabilities")
    val3 = st.text_input("constant capital / total assets")
    val4 = st.text_input("sales / inventory")
    val5 = st.text_input("(short-term liabilities *365) / sales")
    val6 = st.text_input("short-term liabilities / total assets")
    val7 = st.text_input("total assets / total liabilities")
    val8 = st.text_input("book value of equity / total liabilities")
    val9 = st.text_input("current assets / total liabilities")

    # displays a button
    if st.button("Predict"):

        res = requests.post(f"http://localhost:8000/predictclient/{val}/{val1}/{val2}/{val3}/{val4}/{val5}/{val6}/{val7}/{val8}/{val9}/")
        path = res.json()
        st.warning(path.get("message"))







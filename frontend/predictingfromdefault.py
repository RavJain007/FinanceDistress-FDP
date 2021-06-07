import streamlit as st
import requests
import json
FILE_TYPES = ["csv"]
from requests_toolbelt.multipart.encoder import MultipartEncoder

style="""<style>
.primary-button{background-color:#FE7F0E  !important}

</style>
"""
def app():
    st.title("Financial Distress Prediction")
    st.markdown(style, unsafe_allow_html=True)

    val = st.text_input("Enter the Path:")
    val = "/media/mrinal/Windows8_OS/Work/CANVAS/Intern/financeDistress-FDP-Phase1/Prediction_Batch_files"
    url = "http://localhost:8000/predict"
    data = {"json_data": val}
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    if st.button("Predicting from Default Path"):
        res = requests.post(url, data=json.dumps(data), headers=headers)
        path = res.json()
        st.warning(path.get("message"))







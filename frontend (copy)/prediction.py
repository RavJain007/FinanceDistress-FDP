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



    val = st.text_input("First Parameter")
    val1 = st.text_input("Second Parameter")
    val2 = st.text_input("Third Parameter")
    val3 = st.text_input("Fourth Parameter")
    val4 = st.text_input("Fifth Parameter")
    val5 = st.text_input("Sixth Parameter")
    val6 = st.text_input("Seventh Parameter")




    # displays a button
    if st.button("Predict"):

        res = requests.post(f"http://localhost:8000/predictclient/{val}/{val1}/{val2}/{val3}/{val4}/{val5}/{val6}")
        path = res.json()
        st.warning(path.get("message"))

    file = st.file_uploader("File upload", type=FILE_TYPES)
    if st.button("Predict using file upload"):

        m = MultipartEncoder(
            fields={'file': ('Test2', file, 'text/csv')}
        )
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))

        if file:
            res = requests.post(f"http://localhost:8000/predictFileupload", data=m,
                                headers={'Content-Type': m.content_type},
                                timeout=8000)
            path = res.json()
            st.warning(path.get("message"))
            file.close()






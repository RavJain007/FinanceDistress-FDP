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
    file1 = st.file_uploader("File upload")
    if st.button("Predict using file upload"):

        m = MultipartEncoder(
            fields={'file': ('Test2', file1, 'text/arff')}
        )
        show_file = st.empty()
        if not file1:
            show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))

        if file1:
            res = requests.post(f"http://localhost:8000/predictFileupload", data=m,
                                headers={'Content-Type': m.content_type},
                                timeout=8000)
            path = res.json()
            st.warning(path.get("messag-e"))
            file1.close()
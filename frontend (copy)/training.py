import streamlit as st
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
FILE_TYPES = ["csv"]
style="""<style>
.primary-button{background-color:#FE7F0E  !important}
</style>
"""

def app():


    st.title('Training Financial Prediction')
    st.markdown(style, unsafe_allow_html=True)
    file = st.file_uploader("File upload", type=FILE_TYPES)

    m = MultipartEncoder(
        fields={'file': ('Test2', file, 'text/csv')}
    )

    if st.button("Training using file upload"):
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))
        if file:
            res = requests.post(f"http://localhost:8000/trainclient",data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000)
            path = res.json()
            st.warning(path.get("message"))

        file.close()


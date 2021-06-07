import streamlit as st
import requests
import pandas as pd
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
    file1 = st.file_uploader("File upload")
    if st.button("Predict using file upload"):

        m = MultipartEncoder(
            fields={'file': ('Test2', file1, 'text/arff')}
        )
        show_file = st.empty()
        if not file1:
            show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))

        if file1:
            df = st.cache(pd.read_csv)(file1)
            res = requests.post(f"http://localhost:8000/predictFileupload", data=m,
                                headers={'Content-Type': m.content_type},
                                timeout=8000)
            path = res.json()
            #df=df.drop('class',axis=1)
            #df=df.drop('_id',axis=1)

            #dt=json.dumps(path)



            data= json.loads(path)
            df1 = pd.json_normalize(data['0'])
            df2=df1.T
            df2["Result"]=df2
            f_column = df2["Result"]

            df["Result"]=df2["Result"].values;
            st.write(df)
            #st.table(data=df)



            file1.close()
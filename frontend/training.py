import streamlit as st
import requests
from scipy.io.arff import loadarff
import pandas as pd
import csv
from requests_toolbelt.multipart.encoder import MultipartEncoder

FILE_TYPES = ["csv"]
style = """<style>
.primary-button{background-color:#FE7F0E  !important}
</style>
"""


def app():
    st.title('Training Financial Prediction')
    st.markdown(style, unsafe_allow_html=True)
    file = st.file_uploader("File upload")


    if st.button("Training using file upload"):
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))
        if file:
            out_file = open("out-file", "wb")  # open for [w]riting as [b]inary
            out_file.write(file.getvalue())

            with open("out-file", "r") as inFile:
                content = inFile.readlines()
                new = toCsv(content)

                with open("Test.csv", "w") as outFile:
                    outFile.writelines(new)
            with open("Test.csv", 'rb') as csvfile:
                m = MultipartEncoder(
                    fields={'file': ('Test2', csvfile, 'text/csv')}
                )

                res = requests.post(f"http://localhost:8000/trainclient", data=m,
                                    headers={'Content-Type': m.content_type},
                                    timeout=8000)

                path = res.json()
                st.warning(path)

        file.close()
        out_file.close()

    # displays a button
    #val = st.text_input("Training From Default Path")
    #if st.button("Training"):
    #    st.warning(val)
    #    res = requests.post(
    #        f"http://localhost:8000/train/{val}/")
    #    path = res.json()
    #    st.warning(path.get("message"))


def toCsv(content):
    data = False
    header = ""

    newContent = []
    for line in content:
        if not data:
            if "@attribute" in line:
                attri = line.split()
                columnName = attri[attri.index("@attribute") + 1]
                header = header + columnName + ","
            elif "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                newContent.append(header)
        else:
            newContent.append(line)
    return newContent

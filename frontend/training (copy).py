import streamlit as st
import requests
from scipy.io import arff
import pandas as pd
from requests_toolbelt.multipart.encoder import MultipartEncoder
FILE_TYPES = ["csv"]
style="""<style>
.primary-button{background-color:#FE7F0E  !important}
</style>
"""

def app():


    st.title('Training Financial Prediction')
    st.markdown(style, unsafe_allow_html=True)
    file = st.file_uploader("File upload")

    out_file = open("out-file", "wb")  # open for [w]riting as [b]inary
    out_file.write(file.getvalue())
    out_file.close()

    with open("out-file", "r") as inFile:
        content = inFile.readlines()
        new = toCsv(content)


        with open("Test.csv", "w") as outFile:
            outFile.writelines(new)

    dfTest=pd.read_csv("Test.csv")
    st.write(dfTest)
    m = dfTest.to_html()
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

def toCsv(content):
    data = False
    header = ""

    newContent = []
    for line in content:
        if not data:
            if "@attribute" in line:
                attri = line.split()
                columnName = attri[attri.index("@attribute")+1]
                header = header + columnName + ","
            elif "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                newContent.append(header)
        else:
            newContent.append(line)
    return newContent

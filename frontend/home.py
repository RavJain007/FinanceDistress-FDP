import training
import prediction
import  dashboard
import predictionFileupload
import trainingingfromdefault
import predictingfromdefault
import streamlit as st


PAGES = {
   # "Dashboard":dashboard,
    "Training": training,
    "TrainingFromDefault": trainingingfromdefault,
    "Prediction": prediction,
    "PredictionFileUpload":predictionFileupload,
    "Predictionbatch": predictingfromdefault
}
st.sidebar.title('Menu')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
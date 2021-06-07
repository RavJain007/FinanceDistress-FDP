import training
import prediction
import  dashboard
import streamlit as st
PAGES = {
    "Dashboard":dashboard,
    "Training": training,
    "Prediction": prediction
}
st.sidebar.title('Menu')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
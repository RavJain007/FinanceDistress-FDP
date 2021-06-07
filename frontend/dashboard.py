import streamlit as st
import pandas as pd
def app():
    st.title('Dashboard')
    df = st.cache(pd.read_csv)("test2.csv")

    is_check = st.checkbox("Display Data")
    if is_check:
        st.write(df)


    st.bar_chart(df)
import streamlit as st
import pandas as pd
import plotly.express as px
from model import train_model

st.set_page_config(
    page_title="Spotify ML Classification",
    layout="wide"
)

st.title("🎵 Spotify Data Analysis & Classification")

uploaded_file = st.file_uploader(
    "Upload Spotify CSV Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include="number")

    fig = px.imshow(
        numeric_df.corr(),
        text_auto=True,
        aspect="auto"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Popularity Distribution")

    fig2 = px.histogram(
        df,
        x="Popularity",
        nbins=20
    )

    st.plotly_chart(fig2, use_container_width=True)

    if st.button("Train Classification Model"):

        accuracy, report = train_model(df)

        st.success("Model Trained Successfully")

        st.metric(
            "Accuracy",
            f"{accuracy:.2f}%"
        )

        st.text("Classification Report")
        st.text(report)

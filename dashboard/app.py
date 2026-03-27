import streamlit as st
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import gdown
import os

st.set_page_config(page_title="Movie Analytics Dashboard", layout="wide")

@st.cache_data
def load_data():
    # return pd.read_csv("../data/ml_movies.csv")
    return pd.read_csv("https://drive.google.com/uc?id=1JRrRte-rqTEoC04AtGHl4odncoh0_mL4")

df = load_data()

# model = joblib.load("../models/movie_success_model.pkl")
# # model = joblib.load("https://drive.google.com/uc?id=1U818Tu3jUmC9kQgvZh_0AlR3UjKdziqa")

# features = joblib.load("../models/features.pkl")
# # features = joblib.load("https://drive.google.com/uc?id=1IGnWQM9oPu6lgXNAgujghAHxnQD-qeQj")

# Movie model
MODEL_URL = "https://drive.google.com/uc?id=1U818Tu3jUmC9kQgvZh_0AlR3UjKdziqa"
MODEL_PATH = "movie_success_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading movie_success_model.pkl...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = joblib.load(MODEL_PATH)

# Features file
FEATURES_URL = "https://drive.google.com/uc?id=1IGnWQM9oPu6lgXNAgujghAHxnQD-qeQj"
FEATURES_PATH = "features.pkl"

if not os.path.exists(FEATURES_PATH):
    print("Downloading features.pkl...")
    gdown.download(FEATURES_URL, FEATURES_PATH, quiet=False)

features = joblib.load(FEATURES_PATH)

##############################################################################

st.title("Movie Analytics Dashboard")

st.sidebar.header("Movie Inputs")

budget = st.sidebar.number_input("Budget ($)", 1000000, 300000000, 50000000)
release_month = st.sidebar.slider("Release Month", 1,12)
release_year = st.sidebar.slider("Release Year", 1980, 2025, 2020)
num_top_actors = st.sidebar.slider("Number of Top Actors", 0, 5, 1)

input_dict = {
    "budget": budget,
    "release_month": release_month,
    "release_year": release_year,
    "num_top_actors": num_top_actors
}

for genre in features:
    if genre not in input_dict:
        input_dict[genre] = 0

input_df = pd.DataFrame([input_dict])

prob = float(model.predict_proba(input_df)[0][1])

tab1, tab2 = st.tabs(["Movie Success Predictor", "Market Insights"])


with tab1:

    # st.subheader("Predicted Movie Success")

    # st.metric("Probability of Success", f"{prob:.2%}")

    # st.write("Success defined as movies in the top 25% of revenue.")

    # st.progress(prob)

#######################
    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text': "Chance of Success (Top 25% Revenue) (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "#ffcccc"},
                {'range': [40, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#d4edda"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
#######################



with tab2:

    st.subheader("Budget vs Revenue")

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=df,
        x="budget",
        y="revenue",
        alpha=0.3
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    def money(x, pos):
        if x >= 1e9:
            return f'${x/1e9:.0f}B'
        if x >= 1e6:
            return f'${x/1e6:.0f}M'
        if x >= 1e3:
            return f'${x/1e3:.0f}K'
        return f'${x:.0f}'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(money))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(money))

    ax.set_title("Budget vs Revenue (Log Scale)")

    st.pyplot(fig)

    ########################

    st.subheader("Revenue by Release Month")

    revenue_by_month = df.groupby("release_month")["revenue"].sum()

    mapping_months = {1.0:"January",2.0:"February",3.0:"March",4.0:"April",5.0:"May",6.0:"June",7.0:"July",8.0:"August",9.0:"September",10.0:"October",11.0:"November",12.0:"December",}
    revenue_by_month.index = revenue_by_month.index.map(mapping_months)

    fig2, ax2 = plt.subplots()

    revenue_by_month.plot(kind="bar", ax=ax2)

    ax2.set_xlabel("Month")
    ax2.set_ylabel("Total Revenue")

    st.pyplot(fig2)

import streamlit as st
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Movie Analytics Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("./data/ml_movies.csv")

df = load_data()

model = joblib.load("./models/movie_success_model.pkl")
features = joblib.load("./models/features.pkl")

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

    st.subheader("Predicted Movie Success")

    st.metric("Probability of Success", f"{prob:.2%}")

    st.write("Success defined as movies in the top 25% of revenue.")

    st.progress(prob)

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
    ax.set_title("Budget vs Revenue (Log Scale)")

    st.pyplot(fig)

    st.subheader("Revenue by Release Month")

    revenue_by_month = df.groupby("release_month")["revenue"].sum()

    fig2, ax2 = plt.subplots()

    revenue_by_month.plot(kind="bar", ax=ax2)

    ax2.set_xlabel("Month")
    ax2.set_ylabel("Total Revenue")

    st.pyplot(fig2)

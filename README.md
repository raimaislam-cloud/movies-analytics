# Movie Content Performance
A data science project analyzing what drives box office success and predicting high-performing films using machine learning, time series forecasting, and causal inference. Analysis is based on the TMDB 5000 Movie Dataset from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). 

### This project explores movie revenue performance through multiple approaches:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning (Logistic Regression, Random Forest, and XGBoost Classification Models)
- Time Series Forecasting (SARIMAX)
- Causal Inference

### Key Findings
- Models were evaluated using cross-validated ROC-AUC
- Top actors increase success odds by ~5%, holding other variables constant
- Budget is the strongest predictor of financial success
- Revenue trends are non-stationary and generally increasing over time

### Tools
- Language(s): Python
- Data Visualization: Pandas, NumPy, Matplotlib, Seaborn
- Machine Learning: Scikit-learn, XGBoost
- Time Series: Statsmodels (SARIMAX)
- Causal Modeling: Statsmodels (Logit)

### Dashboards

Streamlit App: [https://movies-analytics.streamlit.app/](https://movies-analytics.streamlit.app/)\
Tableau Dashboard: [https://public.tableau.com/views/MovieContentPerformance](https://public.tableau.com/views/MovieContentPerformance/MovieContentPerformance?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

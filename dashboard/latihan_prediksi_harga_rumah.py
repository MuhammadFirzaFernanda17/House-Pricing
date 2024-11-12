import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model

# Load datasets
@st.cache_data
def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

# Data cleaning function
def clean_data(df):
    missing_values = df.isnull().sum()
    less = missing_values[missing_values < 0.75*len(df)].index
    more = missing_values[missing_values >= 0.75*len(df)].index
    numeric_features = df[less].select_dtypes(include=['number']).columns
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
    categorical_features = df[less].select_dtypes(include=['object']).columns
    for col in categorical_features:
        df[col] = df[col].fillna(df[col].mode()[0])
    df = df.drop(columns=more)
    return df

# Data processing
train, test = load_data()
train = clean_data(train)
test = clean_data(test)

# Train-test split
X = train.drop(columns=['SalePrice'])
y = train['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize and train models
lars = linear_model.Lars(n_nonzero_coefs=1).fit(x_train, y_train)
LR = LinearRegression().fit(x_train, y_train)
GBR = GradientBoostingRegressor(random_state=184).fit(x_train, y_train)

# Evaluation function
def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mae, mse, r2

# Streamlit app layout
st.title("Housing Price Prediction Dashboard")
st.sidebar.header("Options")
selected_model = st.sidebar.selectbox("Select Model", ["Least Angle Regression", "Linear Regression", "Gradient Boosting Regressor"])

st.subheader("Dataset Preview")
st.write(train.head())

# Show evaluation results
if selected_model == "Least Angle Regression":
    mae, mse, r2 = evaluate_model(lars, x_test, y_test)
elif selected_model == "Linear Regression":
    mae, mse, r2 = evaluate_model(LR, x_test, y_test)
else:
    mae, mse, r2 = evaluate_model(GBR, x_test, y_test)

st.subheader(f"Evaluation Results for {selected_model}")
st.write(f"MAE: {mae}")
st.write(f"MSE: {mse}")
st.write(f"R2: {r2}")

# Visualization: Correlation heatmap
st.subheader("Correlation Heatmap")
correlation_matrix = train.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Model saving
if st.button("Save Gradient Boosting Model"):
    joblib.dump(GBR, 'model/gbr_model.joblib')
    st.success("Model saved successfully!")

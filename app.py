import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# App title
st.title("Linear Regression App")
st.markdown("Upload a dataset, select the features and target variable, and perform linear regression.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Feature and target selection
    st.write("### Select Features and Target")
    all_columns = list(data.columns)

    features = st.multiselect("Select Features", all_columns)
    target = st.selectbox("Select Target", all_columns)

    if features and target:
        # Split data into train and test sets
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Perform linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Performance")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R2): {r2:.2f}")

        # Coefficients
        st.write("### Model Coefficients")
        coefficients = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
        st.write(coefficients)

        # Plot
        st.write("### Data Visualization")
        for feature in features:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=data[feature], y=data[target], label="Actual")
            sns.lineplot(x=X_test[feature], y=y_pred, color="red", label="Predicted")
            plt.title(f"{feature} vs {target}")
            plt.xlabel(feature)
            plt.ylabel(target)
            plt.legend()
            st.pyplot(plt)

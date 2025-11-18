import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Sustainable Infrastructure Analytics", layout="wide")

st.title("🏙️ Smart & Sustainable Infrastructure Analytics System")
st.write("This app supports SDG 9: Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your infrastructure dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📌 Dataset Preview")
    st.dataframe(data)

    # Select features and target
    target_col = st.selectbox("Select Target Column (Infrastructure Score)", data.columns)
    feature_cols = st.multiselect("Select Feature Columns", [col for col in data.columns if col != target_col], default=[col for col in data.columns if col != target_col])

    X = data[feature_cols]
    y = data[target_col]

    # ML Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    st.subheader("📈 Model Performance")
    st.write(f"**Mean Absolute Error:** {mean_absolute_error(y_test, predictions):.2f}")
    st.write(f"**R2 Score:** {r2_score(y_test, predictions):.2f}")

    # Feature Importance
    st.subheader("🔥 Feature Importance")
    importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values(by="importance", ascending=False)
    st.dataframe(importance)

    fig, ax = plt.subplots()
    ax.barh(importance['feature'], importance['importance'])
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance Graph")
    st.pyplot(fig)

    # Clustering for Development Zones
    st.subheader("🧩 Development Zone Clustering")
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster_group'] = kmeans.fit_predict(X)

    cluster_mean = data.groupby('cluster_group')[target_col].mean().reset_index().sort_values(by=target_col)
    st.write("Cluster Priority:")
    st.dataframe(cluster_mean)

    st.success("Cluster with **lowest infrastructure score** needs **priority development**.")

    # Plot cluster visualization
    st.subheader("📊 Cluster Plot")
    fig2, ax2 = plt.subplots()
    ax2.scatter(data[feature_cols[0]], data[target_col], c=data['cluster_group'])
    ax2.set_xlabel(feature_cols[0])
    ax2.set_ylabel(target_col)
    st.pyplot(fig2)

else:
    st.info("Please upload the CSV file to proceed.")

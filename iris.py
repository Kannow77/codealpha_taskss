import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# App title
st.set_page_config(page_title="🌸 Iris Flower Classifier", layout="centered")
st.title("🌸 Iris Flower Classifier with Decision Tree")
st.markdown("Dooro cabbirrada ubaxa (flower measurements) si aad u saadaaliso nooca species-ka.")

# Sidebar sliders for input features
st.sidebar.header("🔧 Gelinta Sample-ka")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width  = st.sidebar.slider("Sepal Width (cm)",  float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width  = st.sidebar.slider("Petal Width (cm)",  float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Predict button
if st.button("🔍 Saadaali Nooca Ubaxa"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    predicted_class = target_names[prediction[0]]
    
    st.success(f"🌼 Ubaxa waxaa la saadaaliyay inuu yahay: **{predicted_class.title()}**")
    
    st.info(f"""
    **Input Cabirradan Aad Dooratay:**
    - Sepal Length: {sepal_length} cm  
    - Sepal Width: {sepal_width} cm  
    - Petal Length: {petal_length} cm  
    - Petal Width: {petal_width} cm  
    """)

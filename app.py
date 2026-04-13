import streamlit as st
import pickle
import numpy as np

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="ML Model Predictor",
    page_icon="🤖",
    layout="centered"
)

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ------------------ Custom CSS for Animation ------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #667eea, #764ba2);
        color: white;
    }
    .stButton>button {
        background-color: #ff4b2b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #ff416c;
        transform: scale(1.05);
        transition: 0.3s;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(255,255,255,0.2);
        text-align: center;
        font-size: 24px;
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Title ------------------
st.markdown("<h1 style='text-align: center;'>🤖 ML Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter inputs and get predictions instantly</p>", unsafe_allow_html=True)

# ------------------ Input Section ------------------
st.subheader("🔢 Enter Input Features")

# 👉 MODIFY THESE BASED ON YOUR MODEL
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

# Convert to array
input_data = np.array([[feature1, feature2, feature3]])

# ------------------ Prediction ------------------
if st.button("🚀 Predict"):
    try:
        prediction = model.predict(input_data)

        st.markdown(
            f"<div class='result-box'>✅ Prediction: {prediction[0]}</div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error: {e}")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)

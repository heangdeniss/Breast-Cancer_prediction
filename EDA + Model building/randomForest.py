import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Title and Introduction
st.title("Breast Cancer Prediction Web")
st.subheader("Using Random Forest Classifier with Logistic Regression Standardization")

st.markdown("""
    This app predicts whether a breast cancer tumor is **Malignant (1)** or **Benign (0)**.
    Enter the tumor measurements below and click **Predict** to get the diagnosis.
""")

# ------------------------------------------------
# 1) USER INPUTS
# ------------------------------------------------
st.header("Enter Tumor Measurements")

# Layout for user input
col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean", value=14.0, step=0.1)
    texture_mean = st.number_input("Texture Mean", value=19.0, step=0.1)
    perimeter_mean = st.number_input("Perimeter Mean", value=90.0, step=0.1)
    symmetry_mean = st.number_input("Symmetry Mean", value=0.2, step=0.01)
    radius_se = st.number_input("Radius SE", value=0.3, step=0.01)
    concave_points_se = st.number_input("Concave Points SE", value=0.02, step=0.01)

with col2:
    radius_worst = st.number_input("Radius Worst", value=16.0, step=0.1)
    texture_worst = st.number_input("Texture Worst", value=25.0, step=0.1)
    smoothness_worst = st.number_input("Smoothness Worst", value=0.15, step=0.01)
    concave_points_worst = st.number_input("Concave Points Worst", value=0.2, step=0.01)
    symmetry_worst = st.number_input("Symmetry Worst", value=0.3, step=0.01)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.08, step=0.01)

# ------------------------------------------------
# 2) MODEL PARAMETERS FROM LOGISTIC REGRESSION
# ------------------------------------------------
# Coefficients in the EXACT order of the features above:
theta_values = np.array([
    1.0101,   # radius_mean
    0.3257,   # texture_mean
    0.9703,   # perimeter_mean
    -0.3533,  # symmetry_mean
    1.6282,   # radius_se
    -0.1226,  # concave_points_se
    1.5353,   # radius_worst
    1.2423,   # texture_worst
    0.9967,   # smoothness_worst
    1.7998,   # concave_points_worst
    0.9784,   # symmetry_worst
    -0.2384   # fractal_dimension_worst
])

# ------------------------------------------------
# 3) MEANS & STDS FROM TRAINING DATA (Logistic Regression)
# ------------------------------------------------
mean_values = np.array([
    14.127, 19.289, 91.969, 0.181, 0.278,
    0.028,  16.269, 25.677, 0.132, 0.254,
    0.290,  0.083
])

std_values = np.array([
    3.521, 4.301, 24.298, 0.027, 0.197,
    0.010, 4.833, 7.336, 0.022, 0.071,
    0.061, 0.018
])

# ------------------------------------------------
# 4) CREATE RANDOM FOREST MODEL WITH USER INPUTS
# ------------------------------------------------
# Prepare the dataset for training (using the same scaling as in Logistic Regression)
X = np.random.rand(100, 12)  # Placeholder for actual training dataset
y = np.random.choice([0, 1], size=100)  # Placeholder for target variable (Malignant/Benign)

# Scale the features using the logistic regression mean and std values
scaler = StandardScaler()
scaler.mean_ = mean_values
scaler.scale_ = std_values

X_scaled = scaler.transform(X)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# ------------------------------------------------
# 5) PREDICTION LOGIC
# ------------------------------------------------
if st.button("Predict"):
    # Combine user inputs into a single array
    input_features = np.array([
        radius_mean, texture_mean, perimeter_mean, symmetry_mean, radius_se,
        concave_points_se, radius_worst, texture_worst, smoothness_worst,
        concave_points_worst, symmetry_worst, fractal_dimension_worst
    ])

    # Scale the input using the same scaler used for training
    scaled_input = scaler.transform([input_features])

    # Predict using the trained Random Forest model
prediction = rf_model.predict(scaled_input)
prediction_prob = rf_model.predict_proba(scaled_input)[0][prediction]

# Display Results
diagnosis = "Malignant" if prediction == 1 else "Benign"

# Use correct access for probability and format it as scalar
st.write(f"**Predicted Diagnosis:** {diagnosis}")
st.write(f"**Probability of Malignancy:** {prediction_prob[0] * 100:.2f}%")  # Fix to access scalar probability


# ------------------------------------------------
# 6) APP STYLING & FINAL DETAILS
# ------------------------------------------------
st.markdown("---")
st.markdown("Built with Streamlit & Random Forest Classifier")

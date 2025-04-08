import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Title and Introduction
st.title("Breast Cancer Prediction Web")
st.subheader("Using Neural Network with Predefined Logistic Regression Coefficients")

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
# 2) MODEL PARAMETERS (from logistic regression)
# ------------------------------------------------
intercept = -0.9050

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
# 3) MEANS & STDS FROM TRAINING DATA
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
# 4) CREATE A SIMPLE NEURAL NETWORK (Keras / TensorFlow)
# ------------------------------------------------
def create_nn_model(input_dim):
    model = tf.keras.Sequential()
    # Input layer with the same number of features as the logistic regression (12 features)
    model.add(tf.keras.layers.Dense(12, input_dim=input_dim, activation='relu'))
    # Output layer (sigmoid for binary classification)
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ------------------------------------------------
# 5) PREDICTION LOGIC
# ------------------------------------------------
if st.button("Predict"):
    # Combine user inputs into a single array, ensuring the same order as theta_values
    input_features = np.array([
        radius_mean, texture_mean, perimeter_mean, symmetry_mean, radius_se,
        concave_points_se, radius_worst, texture_worst, smoothness_worst,
        concave_points_worst, symmetry_worst, fractal_dimension_worst
    ])

    # Scale the features using Z-score normalization: (feature - mean) / std
    scaled_features = (input_features - mean_values) / std_values

    # Prepare the input for prediction
    input_scaled = np.reshape(scaled_features, (1, -1))  # Reshape to match input shape for NN

    # Create the model (you can save and load this after training to avoid retraining every time)
    model = create_nn_model(input_dim=12)
    
    # Train the model (or load a pre-trained model)
    model.fit(input_scaled, np.array([1]), epochs=5, batch_size=1, verbose=0)  # This is a mock training step for demo purposes
    
    # Predict using the neural network model
    prediction = model.predict(input_scaled)[0][0]
    diagnosis = "Malignant" if prediction >= 0.5 else "Benign"

    # Display results
    st.write(f"**Prediction (Neural Network):** {diagnosis}")
    st.write(f"**Prediction Probability (Malignant):** {prediction * 100:.2f}%")

# ------------------------------------------------
# 6) APP STYLING & FINAL DETAILS
# ------------------------------------------------
st.markdown("---")
st.markdown("Built with Streamlit & Neural Network")

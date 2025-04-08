import streamlit as st
import numpy as np

st.title("Breast Cancer Prediction web")
st.subheader("Using Logistic Regression Coefficients & Intercept")

st.markdown("""
Enter the following tumor measurements. 
Then click **Predict** to see if the tumor is likely **Malignant = 1** or **Benign=0**.
""")

# ------------------------------------------------
# 1) USER INPUTS
# ------------------------------------------------
# Adjust the default values and step sizes as needed:
radius_mean              = st.number_input("Radius Mean",               value=14.0,  step=0.1)
texture_mean             = st.number_input("Texture Mean",              value=19.0,  step=0.1)
perimeter_mean           = st.number_input("Perimeter Mean",            value=90.0,  step=0.1)
symmetry_mean            = st.number_input("Symmetry Mean",             value=0.2,   step=0.01)
radius_se                = st.number_input("Radius SE",                 value=0.3,   step=0.01)
concave_points_se        = st.number_input("Concave Points SE",         value=0.02,  step=0.01)
radius_worst             = st.number_input("Radius Worst",              value=16.0,  step=0.1)
texture_worst            = st.number_input("Texture Worst",             value=25.0,  step=0.1)
smoothness_worst         = st.number_input("Smoothness Worst",          value=0.15,  step=0.01)
concave_points_worst     = st.number_input("Concave Points Worst",      value=0.2,   step=0.01)
symmetry_worst           = st.number_input("Symmetry Worst",            value=0.3,   step=0.01)
fractal_dimension_worst  = st.number_input("Fractal Dimension Worst",   value=0.08,  step=0.01)

# ------------------------------------------------
# 2) MODEL PARAMETERS (from your logistic regression)
# ------------------------------------------------
# Intercept (beta_0)
intercept = -0.9050

# Coefficients in the EXACT order of the features above:
# [ radius_mean, texture_mean, perimeter_mean, symmetry_mean, radius_se,
#   concave_points_se, radius_worst, texture_worst, smoothness_worst,
#   concave_points_worst, symmetry_worst, fractal_dimension_worst ]
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
#    (Replace these with the actual values obtained during training)
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
# 4) PREDICTION LOGIC
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

    # Compute logit(p) using the logistic regression formula:
    # logit(p) = intercept + dot(theta_values, scaled_features)
    logit_p = intercept + np.dot(scaled_features, theta_values)

    # Convert logit(p) to probability using the sigmoid function:
    probability = 1 / (1 + np.exp(-logit_p))

    # Determine diagnosis using a 0.5 threshold:
    diagnosis = "Malignant" if probability >= 0.5 else "Benign"

    # Display the results:
    st.write(f"**Logit(p):** {logit_p:.4f}")
    st.write(f"**Probability of Malignancy:** {probability*100:.2f}%")
    st.success(f"**Predicted Diagnosis: {diagnosis}**")

#st.markdown("---")
#st.markdown("Built with Streamlit & Logistic Regression")

import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.efficientnet import preprocess_input


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Solar Panel Classification",
    page_icon="☀️",
    layout="wide"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f1f5f9;
}

h1 {
    color: #172554;
}

h2, h3 {
    color: #1e3a8a;
}

/* Prediction metric cards */
[data-testid="stMetric"] {
    background-color: #172554;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #334e80;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #bfdbfe !important;
}

/* Prediction values */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: bold !important;
}

/* Metric text */
[data-testid="stMetric"] * {
    color: #ffffff;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. MODEL CONFIGURATION
# ==========================================

MODEL_PATH = Path(__file__).parent / "trained_effnet_finetune.h5"

IMG_SIZE = (224, 224)

CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-Damage",
    "Physical-damage",
    "Snow-Covered"
]


# ==========================================
# 4. CLASS DESCRIPTIONS
# ==========================================

CLASS_DESCRIPTIONS = {
    "Bird-drop": "The solar panel appears to have bird droppings on its surface.",
    "Clean": "The solar panel appears clean and free from visible contamination or damage.",
    "Dusty": "The solar panel appears to have accumulated dust on its surface.",
    "Electrical-Damage": "The solar panel shows signs associated with electrical damage.",
    "Physical-damage": "The solar panel appears to have physical damage.",
    "Snow-Covered": "The solar panel appears to be covered with snow."
}


# ==========================================
# 5. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ==========================================
# 6. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    img_array = np.asarray(
        image,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

    return img_array


# ==========================================
# 7. MAKE PREDICTION
# ==========================================

def predict_image(image, model):

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    probabilities = predictions[0]

    if len(probabilities) != len(CLASSES):
        raise ValueError(
            "Model output does not match the six configured classes."
        )

    predicted_idx = np.argmax(probabilities)

    predicted_class = CLASSES[predicted_idx]

    confidence = float(probabilities[predicted_idx])

    return predicted_class, confidence, probabilities


# ==========================================
# 8. APPLICATION HEADER
# ==========================================

st.title("☀️ Solar Panel Classification")

st.markdown("""
Upload an image of a solar panel to identify its condition using
a deep learning model built with **EfficientNetB0**.
""")

st.divider()


# ==========================================
# 9. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("About the Project")

    st.write("""
    This application uses a fine-tuned EfficientNetB0
    deep learning model to classify solar panel images
    into six different categories.
    """)

    st.subheader("Classification Categories")

    for category in CLASSES:
        st.write(f"• {category}")

    st.divider()

    st.caption("Model: EfficientNetB0")


# ==========================================
# 10. IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a Solar Panel Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(
            [1, 1],
            gap="large"
        )

        # ==================================
        # DISPLAY UPLOADED IMAGE
        # ==================================

        with col1:

            st.subheader("Uploaded Image")

            st.image(
                image,
                caption="Solar Panel Image",
                use_container_width=True
            )

        # ==================================
        # CLASSIFICATION
        # ==================================

        with col2:

            st.subheader("Classification Results")

            with st.spinner("Analyzing solar panel image..."):

                model = load_model()

                predicted_class, confidence, probabilities = predict_image(
                    image,
                    model
                )

            # ==================================
            # RESULTS
            # ==================================

            st.markdown("### Prediction")

            st.metric(
                label="Predicted Class",
                value=predicted_class
            )

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2%}"
            )

            # ==================================
            # CONFIDENCE MESSAGE
            # ==================================

            if confidence < 0.60:

                st.warning(
                    "The model has relatively low confidence "
                    "in this prediction. Consider uploading "
                    "a clearer image."
                )

            else:

                st.success(
                    "Classification completed successfully."
                )

            # ==================================
            # CLASS DESCRIPTION
            # ==================================

            st.markdown("### Interpretation")

            st.info(
                CLASS_DESCRIPTIONS[predicted_class]
            )

        # ==================================
        # TOP 3 PREDICTIONS
        # ==================================

        st.divider()

        st.subheader("Top 3 Predictions")

        top_indices = np.argsort(probabilities)[-3:][::-1]

        for index in top_indices:

            probability = float(probabilities[index])

            st.write(
                f"**{CLASSES[index]}** — {probability:.2%}"
            )

            st.progress(probability)

        # ==================================
        # ALL CLASS PROBABILITIES
        # ==================================

        with st.expander("View All Class Probabilities"):

            for index, class_name in enumerate(CLASSES):

                st.write(
                    f"{class_name}: {probabilities[index]:.2%}"
                )

    except Exception as e:

        st.error(
            f"An error occurred during classification: {e}"
        )


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.caption(
    "Solar Panel Classification | Powered by TensorFlow, "
    "Keras, EfficientNetB0 and Streamlit"
)

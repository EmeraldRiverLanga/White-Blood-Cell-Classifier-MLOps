import mlflow
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# model location is configurable via the MODEL_DIR environment variable
MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models"))
MODEL_PATH = MODEL_DIR / "best_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# the app reaches MLflow by its service name; locally it uses localhost.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
# registry model name and alias to load (the "best" model).
REGISTERED_MODEL_NAME = "wbc-classifier"
MODEL_ALIAS = "champion"

# must match the image size the model was trained on.
IMAGE_SIZE = (224, 224)


@st.cache_resource
def load_model_and_classes():
    """
    Load the trained model and class names once, then keep them in memory.

    @st.cache_resource ensures this runs only on first use, not on every
    user interaction — satisfying assignment 3.4 (model is not retrained
    or reloaded on each run).

    Returns:
        (model, class_names) or raises FileNotFoundError if files are missing.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found at: {CLASS_NAMES_PATH}")

    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    return model, class_names


# page setup
st.set_page_config(
    page_title="WBC Classifier",
    page_icon="🔬",
    layout="centered",
)
# sidebar: model info and instructions
with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies white blood cells using a deep learning model "
        "(MobileNetV2 with transfer learning), trained with Keras and tracked "
        "with MLflow."
    )

    st.header("How to use")
    st.write(
        "1. Upload a blood cell image (JPG or PNG).\n"
        "2. Click **Classify**.\n"
        "3. View the predicted cell type and class probabilities."
    )

    st.header("Cell types")
    st.write(
        "- **Eosinophil**\n" "- **Lymphocyte**\n" "- **Monocyte**\n" "- **Neutrophil**"
    )
st.title("🔬 White Blood Cell Classifier")
st.write(
    "Upload a blood cell image to classify it as one of four white blood "
    "cell types: Eosinophil, Lymphocyte, Monocyte, or Neutrophil."
)


# try to load the model. If it fails, shows a clear error and stop.
try:
    model, class_names = load_model_and_classes()
    st.success(f"Model loaded successfully. Classes: {', '.join(class_names)}")
    with st.sidebar:
        st.header("Model details")
        st.write(f"- Number of classes: **{len(class_names)}**")
        st.write(f"- Input image size: **{IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}**")
        st.write(f"- Total parameters: **{model.count_params():,}**")
except FileNotFoundError as e:
    st.error(
        "Could not load the model. Please train it first by running "
        "`python src/train.py`.\n\n"
        f"Details: {e}"
    )
    st.stop()
# image preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert an uploaded PIL image into the array shape the model expects.

    Steps: ensure RGB, resize to the training size, convert to a numpy
    array, and add a batch dimension. Pixel scaling is handled inside the
    model (preprocess_input layer), so we do NOT scale here.
    """
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.array(image, dtype="float32")
    # model expects a batch, so shape becomes (1, 224, 224, 3).
    array = np.expand_dims(array, axis=0)
    return array

# user input and prediction
st.header("Upload an image")

uploaded_file = st.file_uploader(
    "Choose a blood cell image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    # show the uploaded image so the user can confirm it.
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=300)

    # prediction button — nothing runs until the user clicks it.
    if st.button("Classify"):
        with st.spinner("Classifying..."):
            batch = preprocess_image(image)
            predictions = model.predict(batch, verbose=0)[0]  # shape (4,)

        # the predicted class is the one with the highest probability.
        top_index = int(np.argmax(predictions))
        top_class = class_names[top_index]
        top_confidence = float(predictions[top_index])

        # main prediction result.
        st.subheader("Prediction")
        st.success(f"**{top_class}** ({top_confidence:.1%} confidence)")

        # show probabilities for all classes so the user sees how
        # confident the model is across every option.
        st.subheader("Class probabilities")
        prob_dict = {
            class_names[i]: float(predictions[i]) for i in range(len(class_names))
        }
        st.bar_chart(prob_dict)

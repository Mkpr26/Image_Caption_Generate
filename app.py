# ---------------------------------------------------------------
# 🖼️ IMAGE CAPTION GENERATOR (FINAL WORKING STREAMLIT VERSION)
# Author: Manish Kumar
# ---------------------------------------------------------------

import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from PIL import Image

# ---------------------------------------------------------------
# 🌐 PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title("🖼️ Image Caption Generator")
st.write("Upload an image, and the trained deep learning model will describe it!")

# ---------------------------------------------------------------
# ⚙️ LOAD MODEL SAFELY
# ---------------------------------------------------------------
@st.cache_resource
def load_caption_model():
    try:
        model = tf.keras.models.load_model("best_model_fixed.keras", compile=False)
        return model
    except Exception as e:
        st.error("❌ Error loading model. Please ensure best_model_fixed.keras exists and is valid.")
        st.exception(e)
        st.stop()


model = load_caption_model()
st.success("✅ Model loaded successfully!")

# ---------------------------------------------------------------
# ⚙️ LOAD TOKENIZER + FEATURES
# ---------------------------------------------------------------
@st.cache_resource
def load_encodings():
    """Load tokenizer and features safely."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try both filenames
    tokenizer_paths = [
        os.path.join(base_dir, "tokenizer_data.pkl")
    ]
    
    tokenizer_path = next((p for p in tokenizer_paths if os.path.exists(p)), None)
    features_path = os.path.join(base_dir, "Processed_Feature", "features.pkl")

    if not tokenizer_path:
        st.error("❌ Tokenizer file not found! Please ensure 'tokenizer.pkl' or 'tokenizer_data.pkl' exists beside app.py.")
        st.stop()

    if not os.path.exists(features_path):
        st.error("❌ features.pkl not found in Processed_Feature folder!")
        st.stop()

    # Load tokenizer
    with open(tokenizer_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        tokenizer = data.get("tokenizer", None)
        max_length = data.get("max_length", 35)
    else:
        tokenizer = data
        max_length = 35

    # Load preprocessed CNN features
    with open(features_path, "rb") as f:
        features = pickle.load(f)

    return features, tokenizer, max_length


features, tokenizer, max_length = load_encodings()
st.success(f"✅ Tokenizer loaded successfully (max_length = {max_length})")

# ---------------------------------------------------------------
# 🧠 FEATURE EXTRACTOR (VGG16)
# ---------------------------------------------------------------
@st.cache_resource
def get_feature_extractor():
    base_model = VGG16()
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    return model


feature_extractor = get_feature_extractor()


def extract_features(img_path):
    """Extract CNN features from the uploaded image."""
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = feature_extractor.predict(image, verbose=0)
    return feature


# ---------------------------------------------------------------
# 🧾 CAPTION GENERATION (GREEDY SEARCH)
# ---------------------------------------------------------------
def word_for_id(integer, tokenizer):
    """Convert predicted integer back to word."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption_beam_search(model, tokenizer, photo, max_length=None, beam_width=5):
    """Generate accurate captions using Beam Search with automatic length detection."""
    # ✅ Automatically detect correct max_length from model input
    if max_length is None:
        try:
            # Model expects [photo_input, sequence_input]
            model_max_len = model.input_shape[1][1]  # e.g., (None, 35)
        except Exception:
            model_max_len = 35
    else:
        model_max_len = max_length

    st.write(f"🔍 Model expects sequence length: {model_max_len}")

    sequences = [['startseq', 0.0]]

    for _ in range(model_max_len):
        all_candidates = []

        for seq, score in sequences:
            # Convert text → tokens
            sequence = tokenizer.texts_to_sequences([seq])[0]

            # ✅ Trim or pad sequence EXACTLY to model_max_len
            if len(sequence) > model_max_len:
                sequence = sequence[-model_max_len:]
            else:
                sequence = sequence

            sequence = pad_sequences([sequence], maxlen=model_max_len, padding='post', truncating='post')

            # ✅ Predict next token
            preds = model.predict([photo, sequence], verbose=0)
            preds = preds[0]

            # Pick top candidates
            top = np.argsort(preds)[-beam_width:]

            for word_idx in top:
                word = word_for_id(word_idx, tokenizer)
                if word is None:
                    continue
                new_seq = seq + ' ' + word
                new_score = score - np.log(preds[word_idx] + 1e-7)
                all_candidates.append([new_seq, new_score])

        # ✅ Keep top beam_width candidates only
        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:beam_width]

    # ✅ Final caption cleanup
    final_caption = sequences[0][0]
    final_caption = final_caption.replace('startseq', '').replace('endseq', '').strip()
    return final_caption


# ---------------------------------------------------------------
# 📸 STREAMLIT INTERFACE
# ---------------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ✅ Save uploaded image inside 'Images' folder
    images_dir = "data/Images"
    os.makedirs(images_dir, exist_ok=True)

    image_name = uploaded_file.name
    image_path = os.path.join(images_dir, image_name)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ✅ Display the uploaded image
    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)

    # ✅ Extract CNN features using same preprocessing as training
    with st.spinner("🔍 Extracting image features..."):
        photo_features = extract_features(image_path)

    # ✅ Generate caption using beam search
    with st.spinner("🧠 Generating caption..."):
        caption = generate_caption_beam_search(model, tokenizer, photo_features, max_length, beam_width=5)

    st.success("✅ Caption Generated Successfully!")
    st.subheader("📝 Generated Caption:")
    st.markdown(f"**{caption}**")


# ---------------------------------------------------------------
# 🧩 FOOTER
# ---------------------------------------------------------------
st.markdown("---")
st.info("""
### 🧠 Notes:
- `best_model_fixed.keras` → your trained image captioning model  
- `tokenizer.pkl` or `tokenizer_data.pkl` → tokenizer & max_length  
- `Processed_Feature/features.pkl` → preprocessed CNN features  
- Run this file using:
""")
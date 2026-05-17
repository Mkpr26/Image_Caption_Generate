# Image Caption Generator 

An AI-powered **Image Caption Generator** that automatically describes the content of an image in natural language.  
This project combines **Computer Vision (VGG16)** and **Natural Language Processing (LSTM + Embedding)** to generate captions for real-world images.

Streamlit app link for Demo :  https://mkpr26-imagecaptiongenerate.streamlit.app/

## 🚀 Project Overview

-->The model learns to generate meaningful captions by training on the **Kaggle Flickr8k dataset**, which contains 8,000 images and 5 captions per image.

--> Each image is processed through a **CNN (VGG16)** for feature extraction, and captions are processed via **RNN (LSTM)** for sequence generation.

--> After training, the model is deployed using **Streamlit**, allowing users to upload any image and get an automatically generated caption.


## 📂 Dataset Information

**Dataset Used:** [Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

- **Images:** 8,091 JPEG images (`.jpg`)
- 
- **Captions:** Each image has 5 textual captions
- 
- **Total Captions:** ~40,456
- 
- **File Used:** `captions.txt` (contains mappings between images and their captions)
- 
- **Image Folder:** `Images/`

If you want to use the **Flickr30k** dataset, you can replace the Flickr8k images and caption file — the same preprocessing and model structure will work.

## 🧠 Model Architecture

| Component | Description |
|------------|-------------|
| **Feature Extractor** | Pre-trained **VGG16** model (removes last classification layer) |
| **Sequence Model** | LSTM network trained on caption sequences |
| **Embedding Layer** | Converts words into dense 256-dimensional vectors |
| **Dense Layers** | Combine CNN features + LSTM output |
| **Final Layer** | Softmax layer to predict the next word in the caption sequence |



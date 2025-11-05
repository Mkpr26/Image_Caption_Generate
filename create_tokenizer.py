
import pickle
import string

# Load your captions file
captions_path = "DATA/captions.txt"

print("Reading captions.txt ...")
with open(captions_path, 'r', encoding='utf-8') as f:
    captions = f.read().split('\n')

def clean_caption(caption):
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = ' '.join([w for w in caption.split() if w.isalpha()])
    return caption.strip()

cleaned = []
for cap in captions:
    if len(cap.strip()) > 0:
        cleaned.append("startseq " + clean_caption(cap) + " endseq")

print(f"{len(cleaned)} captions cleaned.")

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(cleaned)
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocab size: {vocab_size}")

max_length = max(len(c.split()) for c in cleaned)
print(f"Max caption length: {max_length}")

save_path = "tokenizer_data.pkl"
with open(save_path, "wb") as f:
    pickle.dump({"tokenizer": tokenizer, "max_length": max_length}, f)

print("Tokenizer and max_length saved successfully as 'tokenizer_data.pkl'")

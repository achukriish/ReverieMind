from flask import Flask, request, render_template
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__, static_folder="static")

# Load the dataset
try:
    dream_data = pd.read_csv("dream_dataset.csv")
    dream_data.columns = dream_data.columns.str.strip()

    if 'dream' not in dream_data.columns or 'interpretation' not in dream_data.columns:
        print("Error: Required columns ('dream', 'interpretation') not found in dataset.")
        dream_data = None
    else:
        print("Dataset loaded successfully!")

except Exception as e:
    print("Error loading dataset:", str(e))
    dream_data = None

# Load the BERT model for sentence similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for all dreams in the dataset
if dream_data is not None:
    dream_data["embedding"] = dream_data["dream"].apply(lambda x: model.encode(x, convert_to_tensor=True))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/interpret', methods=['GET', 'POST'])
def interpret():
    if dream_data is None:
        return "Error: Dataset not loaded properly. Please check the server logs."

    dream = request.args.get('dream', '').strip() if request.method == 'GET' else request.form.get('dream', '').strip()

    if not dream:
        return render_template('result.html', interpretation="Vanga sirrr! 😂 You forgot to type your dream! Ennada ippadi? 😆")

    try:
        # Convert input dream to embedding
        input_embedding = model.encode(dream, convert_to_tensor=True)

        # Compute similarity scores
        similarities = [util.pytorch_cos_sim(input_embedding, emb)[0].item() for emb in dream_data["embedding"]]

        # Get the best matching dream
        best_index = torch.argmax(torch.tensor(similarities)).item()
        best_score = similarities[best_index]

        # If the similarity score is high enough, return the best match
        if best_score >= 0.65:  # 0.65 means high confidence match
            interpretation = dream_data.iloc[best_index]['interpretation']
        else:
            interpretation = "Vanga sirrr! 😂 I couldn't find this dream in my book, but maybe it's just your brain playing a prank!"

        # Ensure "Free advice" is always there
        if "Free advice" not in interpretation:
            interpretation += " Free advice: Stay hydrated da! Even dream heroes need water! 💧😂"

    except Exception as e:
        interpretation = f"Unexpected error: {str(e)}"

    return render_template('result.html', interpretation=interpretation)

if __name__ == '__main__':
    app.run(debug=True)

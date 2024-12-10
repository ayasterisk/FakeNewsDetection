from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models import LSTMModel
import torch
from torch import nn
from joblib import load
import joblib
import numpy as np
from gensim.models import Word2Vec, FastText
import nltk
from transformers import BertModel, BertTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

app = Flask(__name__)

models = {
    "embed1_model1": load("models/model_w2v_nb.pkl"),
    "embed1_model2": load_model("models/model_w2v_lstm.h5"),
    "embed2_model1": load("models/model_glove_nb.joblib"),
    "embed2_model2": load_model("models/model_glove_lstm.h5"),
    "embed3_model1": load("models/model_fasttext_nb.pkl"),
    "embed3_model2": load_model("models/model_fasttext_lstm.h5"),
    "embed4_model1": load("models/model_bert_nb.joblib"),
    "embed4_model2": load("models/model_bert_lstm.joblib")
}


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news_text = request.form["news_text"]
        embedding_choice = request.form["embedding_choice"]
        model_choice = request.form["model_choice"]

        model_key = f"{embedding_choice}_{model_choice}"
        selected_model = models.get(model_key)

        if not selected_model:
            return render_template("index.html", error="Lựa chọn không hợp lệ.", news_text=news_text)
        # Word2Vec Embedding
        if embedding_choice == "embed1":
            def document_vector(doc, word2vec_model):
                doc = [word for word in doc if word in word2vec_model.wv]
                return np.mean([word2vec_model.wv[word] for word in doc], axis=0) if doc else np.zeros(100)
            
            word2vec_model = Word2Vec.load("models/word2vec_model.bin")
            processed_text = preprocess_text(news_text)
            if model_choice == "model1":
                vector = document_vector(processed_text, word2vec_model).reshape(1, -1)
                accuracy = 79.48
            else:
                text_input = [news_text for _ in range(32)]
                vector_list = [document_vector(preprocess_text(sentence), word2vec_model) for sentence in text_input]
                vector_array = np.array(vector_list)
                vector = np.expand_dims(vector_array, axis=1)
                accuracy = 90.61
            prediction = selected_model.predict(vector)
            result = "Real" if prediction[0] < 0.5 else "Fake"
        
        # GloVe Embedding    
        if embedding_choice == "embed2":
            loaded_word_index = load("models/word_index.joblib")
            loaded_embedding_matrix = load("models/embedding_matrix.joblib")
            processed_text = preprocess_text(news_text)
            if model_choice == "model1":                             
                sentence_vector = np.mean(loaded_embedding_matrix[[loaded_word_index[word] for word in processed_text if word in loaded_word_index]], axis=0)
                vector = sentence_vector.reshape(1, -1)
                prediction = selected_model.predict(vector)  
                accuracy = 72.38                             
            else:
                sequence = [[loaded_word_index[word] for word in processed_text if word in loaded_word_index]]
                padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
                prediction = selected_model.predict(padded_sequence)
                prediction = 1 - prediction[0]
                accuracy = 86.42
            result = "Real" if prediction[0] < 0.5 else "Fake"
                
        # fastText Embedding
        if embedding_choice == "embed3":
            processed_text = preprocess_text(news_text)
            def document_vector(doc, model):
                words = [word for word in doc if word in model.wv.key_to_index]
                if len(words) == 0:
                    return np.zeros(model.vector_size)
                return np.mean(model.wv[words], axis=0)

            if model_choice == "model1":
                scaler = joblib.load('models/minmax_scaler_fT.pkl')
                fasttext_model = FastText.load("models/fasttext_model.bin")
                doc_vec = document_vector(processed_text, fasttext_model).reshape(1, -1)
                scaled_vec = scaler.transform(doc_vec)
                prediction = selected_model.predict(scaled_vec)
                accuracy = 80.58
            else:
                fasttext_model = FastText.load("models/fasttext_model.bin")
                word_index = {word: idx + 1 for idx, word in enumerate(fasttext_model.wv.key_to_index)}
                def text_to_sequence(text, word_index):
                    return [word_index[word] for word in text if word in word_index]
                sequence = text_to_sequence(processed_text, word_index)
                padded_sequence = pad_sequences([sequence], maxlen=200, padding='post')
                prediction = selected_model.predict(padded_sequence)[0]
                accuracy = 84.61
            result = "Real" if prediction[0] < 0.5 else "Fake"
                
        # BERT embedding
        if embedding_choice == "embed4":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            if model_choice == "model1":
                def get_bert_embeddings_batch(text_list, tokenizer, bert_model, batch_size=32):
                    embeddings = []
                    for i in range(0, len(text_list), batch_size):
                        batch_texts = text_list[i:i + batch_size]
                        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
                        with torch.no_grad():
                            outputs = bert_model(**inputs)
                        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                        embeddings.append(cls_embeddings)
                    embeddings = np.vstack(embeddings)
                    return embeddings
                input_embedding = get_bert_embeddings_batch([news_text], tokenizer, bert_model, batch_size=1)
                prediction = selected_model.predict(input_embedding)
                accuracy = 82.32
                result = "Real" if prediction[0] < 0.5 else "Fake"
            else:
                accuracy = 89.66
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                selected_model.to(device)
                def get_bert_embeddings_lstm(text, tokenizer, bert_model):
                    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
                    with torch.no_grad():
                        outputs = bert_model(**inputs)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                    return cls_embeddings
                embeddings = get_bert_embeddings_lstm(news_text, tokenizer, bert_model)
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
                embeddings_tensor = embeddings_tensor.to(device)
                embeddings_tensor = embeddings_tensor.unsqueeze(1)
                with torch.no_grad():
                    output = selected_model(embeddings_tensor)
                    _, predicted = torch.max(output, 1)
                result = "Fake" if predicted.item() == 1 else "Real"


        return render_template(
            "index.html", accuracy=accuracy, prediction=result, news_text=news_text, embedding_choice=embedding_choice, model_choice=model_choice
        )

if __name__ == "__main__":
    app.run(debug=True)

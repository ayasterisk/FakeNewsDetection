from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import joblib
import numpy as np
from gensim.models import Word2Vec
import nltk
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
    # "embed2_model1": load("models/model_glove_nb.pkl"),
    # "embed2_model2": load_model("models/model_glove_lstm.h5"),
    # "embed3_model1": load("models/model_fasttext_nb.pkl"),
    # "embed3_model2": load_model("models/model_fasttext_lstm.h5"),
    # "embed4_model1": load("models/model_bert_nb.pkl"),
    # "embed4_model2": load_model("models/model_bert_lstm.h5")
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
        
        if embedding_choice == "embed1":
            def document_vector(doc, word2vec_model):
                doc = [word for word in doc if word in word2vec_model.wv]
                return np.mean([word2vec_model.wv[word] for word in doc], axis=0) if doc else np.zeros(100)
            
            word2vec_model = Word2Vec.load("models/word2vec_model.bin")
            processed_text = preprocess_text(news_text)
            if model_choice == "model1":
                vector = document_vector(processed_text, word2vec_model).reshape(1, -1)
            else:
                text_input = [news_text for _ in range(32)]
                vector_list = [document_vector(preprocess_text(sentence), word2vec_model) for sentence in text_input]
                vector_array = np.array(vector_list)
                vector = np.expand_dims(vector_array, axis=1)
            prediction = selected_model.predict(vector)
            
        result = "Real" if prediction[0] < 0.5 else "Fake"

        return render_template(
            "index.html", prediction=result, news_text=news_text, embedding_choice=embedding_choice, model_choice=model_choice
        )

if __name__ == "__main__":
    app.run(debug=True)

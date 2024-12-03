from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Khởi tạo Flask app
app = Flask(__name__)

# Load tất cả các model
models = {
    "embed1_model1": load_model("models/model1.h5"),
    "embed1_model2": load_model("models/model5.h5"),
    "embed2_model1": load_model("models/model2.h5"),
    "embed2_model2": load_model("models/model6.h5"),
    "embed3_model1": load_model("models/model3.h5"),
    "embed3_model2": load_model("models/model7.h5"),
    "embed4_model1": load_model("models/model4.h5"),
    "embed4_model2": load_model("models/model8.h5")
}

# Tokenizer giả định (tùy chỉnh lại với dữ liệu thật)
tokenizer = Tokenizer(num_words=10000)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Lấy dữ liệu từ form
        news_text = request.form["news_text"]
        embedding_choice = request.form["embedding_choice"]
        model_choice = request.form["model_choice"]

        # Ghép khóa để chọn đúng model
        model_key = f"{embedding_choice}_{model_choice}"
        selected_model = models.get(model_key)

        if not selected_model:
            return render_template("index.html", error="Lựa chọn không hợp lệ.", news_text=news_text)

        # Tiền xử lý văn bản
        sequences = tokenizer.texts_to_sequences([news_text])
        padded = pad_sequences(sequences, maxlen=100)  # maxlen phải phù hợp với mô hình

        # Dự đoán
        prediction = selected_model.predict(padded)
        result = "Fake" if prediction[0] < 0.5 else "Real"

        return render_template(
            "index.html", prediction=result, news_text=news_text, embedding_choice=embedding_choice, model_choice=model_choice
        )

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
from utils import extract_text, ask_question
import tempfile

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def handle_ask():
    file = request.files["file"]
    question = request.form["question"]
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        content = extract_text(tmp.name)
    
    answer = ask_question(content, question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run()

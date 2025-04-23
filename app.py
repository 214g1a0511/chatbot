import os
import re
from flask import Flask, request, jsonify
from transformers import pipeline
from docx import Document
import fitz  # PyMuPDF
from dotenv import load_dotenv
from together import Together

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load emotion classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

# Map emotions to emojis
emotion_to_emoji = {
    "joy": "😊", "anger": "😠", "sadness": "😢",
    "fear": "😨", "surprise": "😲", "disgust": "🤢",
    "neutral": "💬", "confusion": "🤔", "love": "❤️",
}

# Extract text from uploaded files
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        return "\n".join([page.get_text() for page in doc])
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format")

# Clean the response
def clean_response(answer):
    return re.sub(r'(These .*?Section\s*\d+.*?)$', '', answer, flags=re.IGNORECASE).strip()

# Ask question using Together API
def ask_question(content, question):
    prompt = f"""
You are an intelligent assistant helping the user understand the background verification process. 

The user may ask questions related to the background verification process, and you should answer based on the content provided. Do not refer to any document directly. Just explain the background verification process using the information available.

--- BEGIN CONTENT ---
{content}
--- END CONTENT ---

User Question: {question}
"""
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{"role": "user", "content": prompt}]
    )
    raw_answer = response.choices[0].message.content.strip()

    # Try to classify emotion
    try:
        emotion_result = emotion_classifier(raw_answer)
        if isinstance(emotion_result, list):
            emotion = emotion_result[0]['label'].lower()
        else:
            emotion = emotion_result['label'].lower()
        emoji = emotion_to_emoji.get(emotion, "💬")
    except Exception as e:
        print(f"Emotion classification error: {e}")
        emoji = "💬"

    return f"{emoji} {clean_response(raw_answer)}"

# API route to handle file upload and question
@app.route("/ask", methods=["POST"])
def handle_question():
    if "file" not in request.files or "question" not in request.form:
        return jsonify({"error": "File and question are required"}), 400

    file = request.files["file"]
    question = request.form["question"]
    filepath = os.path.join("uploads", file.filename)

    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    try:
        content = extract_text(filepath)
        answer = ask_question(content, question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Home route
@app.route("/", methods=["GET"])
def home():
    return "<h2>📄 Document Q&A API is Running</h2>"

# if __name__ == "__main__":
#     # Use environment variable PORT for dynamic port binding
#     app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    # Render will provide the port number via the environment variable "PORT"
    port = int(os.getenv("PORT", 5000))  # Default to 5000 if not set
    app.run(host="0.0.0.0", port=port)

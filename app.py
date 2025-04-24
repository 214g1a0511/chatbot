import os
import re
from flask import Flask, request, jsonify
from transformers import pipeline
from docx import Document
import fitz  # PyMuPDF
from dotenv import load_dotenv
from together import Together
from googletrans import Translator
from langdetect import detect

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
    """
    Extract text from PDF, DOCX, or TXT files.
    """
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

# Clean the response text (remove unnecessary sections)
def clean_response(answer):
    """
    Clean the answer to remove unwanted text such as notes or section headers.
    """
    return re.sub(r'(These .*?Section\s*\d+.*?)$', '', answer, flags=re.IGNORECASE).strip()

# Ask question using Together API
def ask_question(content, question):
    """
    Uses Together API to get an answer from the document content.
    """
    prompt = f"""
You are an intelligent assistant helping the user understand an uploaded document.

The user may ask questions related to the document, and you should answer based on the content provided. 

--- BEGIN CONTENT ---
{content}
--- END CONTENT ---

User Question: {question}
"""
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # Use correct model
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

    # Clean the response to remove any translation text or unnecessary commentary
    cleaned_answer = clean_response(raw_answer)
    return f"{emoji} {cleaned_answer}"

# Route 1: Accepts a file and a question
@app.route("/ask", methods=["POST"])
def ask_with_file():
    """
    API endpoint to accept file upload and question for processing.
    """
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

# Route 2: Accepts a question, uses a predefined file from the 'uploads' folder
@app.route("/ask_predefined", methods=["POST"])
def ask_with_predefined_file():
    """
    API endpoint to accept a question and return an answer using a predefined document.
    """
    question = request.form.get("question")
    
    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Path to the predefined document in the 'uploads' directory (absolute path)
    predefined_file_path = os.path.join(os.getcwd(), "uploads", "Sample_BGV_Document.pdf")
    
    # Debug print the path for debugging purposes
    print(f"Predefined file path: {predefined_file_path}")

    # Check if the predefined file exists
    if not os.path.exists(predefined_file_path):
        return jsonify({"error": "Predefined file not found in the uploads directory"}), 404

    try:
        content = extract_text(predefined_file_path)  # Use predefined file
        answer = ask_question(content, question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Home route
@app.route("/", methods=["GET"])
def home():
    """
    Simple home route to check if the API is running.
    """
    return "<h2>📄 Document Q&A API is Running</h2>"

# If running locally (not on a platform like Render), allow dynamic port binding
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Default to 5000 if not set
    app.run(host="0.0.0.0", port=port)

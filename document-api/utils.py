import os
from dotenv import load_dotenv
import fitz
from docx import Document
import re
from transformers import pipeline
from together import Together

# Load environment variables
load_dotenv()

# Ensure API key is available
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("TOGETHER_API_KEY is not set in the environment variables.")

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

emotion_to_emoji = {
    "joy": "😊", "anger": "😠", "sadness": "😢",
    "fear": "😨", "surprise": "😲", "disgust": "🤢",
    "neutral": "💬", "confusion": "🤔", "love": "❤️",
}

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        try:
            doc = fitz.open(file_path)
            return "\n".join([page.get_text() for page in doc])
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")
    elif file_path.endswith(".docx"):
        try:
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            raise ValueError(f"Failed to extract text from DOCX: {e}")
    elif file_path.endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read TXT file: {e}")
    else:
        raise ValueError("Unsupported file format")

def clean_response(answer):
    return re.sub(r'(These .*?Section\s*\d+.*?)$', '', answer, flags=re.IGNORECASE).strip()

def ask_question(content, question):
    prompt = f"""
You are an intelligent assistant helping the user understand an uploaded document.

Answer only using the document. No section references.

--- BEGIN DOCUMENT ---
{content}
--- END DOCUMENT ---

User Question: {question}
"""
    client = Together(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}]
        )
        raw_answer = response.choices[0].message.content.strip()
    except Exception as e:
        raise ValueError(f"Failed to get response from Together API: {e}")
    
    # Analyze the emotion of the answer
    emotion = emotion_classifier(raw_answer)[0]['label'].lower()
    emoji = emotion_to_emoji.get(emotion, "💬")
    return f"{emoji} {clean_response(raw_answer)}"


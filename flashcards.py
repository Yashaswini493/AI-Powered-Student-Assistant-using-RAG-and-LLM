import requests
import re
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import re

def generate_flashcards(text, num_cards, concept, groq_api_key, groq_api_url, groq_model):
    """
    Generate flashcards from PDF text using Groq's LLM.
    Each question and answer is exactly one sentence.
    """
    max_text_length = 3000
    if len(text) > max_text_length:
        summary_prompt = f"Summarize the following content in 500 words to create flashcards from:\n\n{text}"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": groq_model,
            "messages": [{"role": "user", "content": summary_prompt}],
            "max_tokens": 512,
            "temperature": 0.3
        }
        try:
            response = requests.post(groq_api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", text[:max_text_length]).strip()
        except Exception as e:
            text = text[:max_text_length]  # Fallback to truncated text

    if concept.lower() == "entire pdf":
        concept_instruction = "Use the contents in the entire PDF to generate flashcards."
    else:
        concept_instruction = f"Focus deeply ONLY on: **{concept}** (ignore unrelated content). All cards must be about this specific concept."

    prompt = f"""Create {num_cards} flashcards based on {concept_instruction} from the following content.
                Each question must be exactly one sentence.
                Each answer must be exactly one sentence.
                Format the output so that each flashcard is on two separate lines:
                First line: Q: <question sentence>
                Second line: A: <answer sentence>

                Separate each flashcard by one empty line.

                Content:
                {text}

                Flashcards: """
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": groq_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(groq_api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        flashcard_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        flashcards = parse_flashcards(flashcard_text)
        if not flashcards:
            flashcards = create_simple_flashcards(text, num_cards, groq_api_key, groq_api_url, groq_model)

        return flashcards[:num_cards]
    except Exception as e:
        return [{"question": f"Error generating flashcards: {e}", "answer": "Please try again."}]

def parse_flashcards(text):
    """
    Parses flashcards formatted as:
    Q: question sentence
    A: answer sentence

    Each flashcard is separated by a single empty line (\n)
    """
    cards = []
    raw_cards = text.strip().split("\n\n")  # split by single blank line

    for raw in raw_cards:
        raw = raw.strip()
        if not raw:
            continue

        # Extract question and answer lines
        lines = raw.split('\n')
        if len(lines) >= 2 and lines[0].startswith("Q:") and lines[1].startswith("A:"):
            question = lines[0][2:].strip()
            answer = lines[1][2:].strip()
            cards.append({"question": question, "answer": answer})

    return cards

def create_simple_flashcards(text, num_cards, groq_api_key, groq_api_url, groq_model):
    """
    Create simple flashcards when parsing fails.
    """
    # Extract key concepts and create simple Q&A
    prompt = f"""From this content, create {num_cards} simple flashcards.
For each concept, create a question and answer pair.
Format: "1. Q: [question] A: [answer]"

Content: {text[:2000]}

Simple flashcards:"""
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": groq_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(groq_api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        result = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # Simple parsing for numbered format
        flashcards = []
        lines = result.split('\n')
        current_q = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:') or line.startswith('Question:'):
                current_q = line.split(':', 1)[1].strip()
            elif line.startswith('A:') or line.startswith('Answer:') and current_q:
                answer = line.split(':', 1)[1].strip()
                flashcards.append({
                    "question": current_q,
                    "answer": answer
                })
                current_q = None
        
        return flashcards[:num_cards]
    except Exception as e:
        return [{"question": "What is the main topic?", "answer": "The content covers various topics."}]

def format_flashcard_display(flashcards):
    """
    Format flashcards for nice display in Streamlit.
    """
    if not flashcards:
        return "No flashcards generated."
    
    formatted = []
    for i, card in enumerate(flashcards, 1):
        formatted.append(f"**Card {i}:**\n")
        formatted.append(f"**Q:** {card['question']}\n")
        formatted.append(f"**A:** {card['answer']}\n")
        formatted.append("---\n")
    
    return "\n".join(formatted) 

def create_flashcards_pdf(flashcards_list, filename="flashcards.pdf"):
    """Create a PDF file from flashcards"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("📚 Flashcards", title_style))
    story.append(Spacer(1, 20))
    
    # Add each flashcard
    for i, card in enumerate(flashcards_list, 1):
        # Question
        question_text = f"<b>Question {i}:</b> {card['question']}"
        story.append(Paragraph(question_text, styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Answer
        answer_text = f"<b>Answer:</b> {card['answer']}"
        story.append(Paragraph(answer_text, styles['Normal']))
        story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_flashcards_csv(flashcards_list):
    """Create a CSV file from flashcards"""
    data = []
    for i, card in enumerate(flashcards_list, 1):
        data.append({
            'Card Number': i,
            'Question': card['question'],
            'Answer': card['answer']
        })
    
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue() 
import numpy as np
import requests
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import io
import re

def retrieve_top_k(query, embedder, index, chunks, k=2):
    """Retrieve top k most relevant chunks using FAISS index"""
    query_emb = embedder.encode([query]).astype("float32")
    D, I = index.search(query_emb, k)
    return [chunks[i] for i in I[0]]

def clean_generated_text(text):
    """
    Completely removes ALL <think>...</think> tags AND their content.
    Also handles malformed tags and multiple occurrences.
    """
    if not text:
        return text
    
    # Remove all <think>...</think> blocks (including content)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove any remaining orphaned <think> tags (no closing tag)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    
    # Remove any remaining </think> tags (no opening tag)
    text = re.sub(r'</think>', '', text)
    
    # Clean up extra whitespace
    return ' '.join(text.split()).strip()

def grok_generate_answer(context_chunks, question, grok_api_key, grok_api_url, grok_model):
    """Generate answer using Grok API with RAG context"""
    if isinstance(context_chunks, list):
        context_str = "\n".join(context_chunks)
    else:
        context_str = str(context_chunks)
        
    prompt = (
        "You are a helpful assistant. Use ONLY the context below to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
    )
    
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": grok_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(grok_api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        raw_output = response.json()["choices"][0]["message"]["content"].strip()
        return clean_generated_text(raw_output)
    except Exception as e:
        return f"Error querying Grok API: {str(e)}"

def answer_question(query, embedder, index, chunks, grok_api_key, grok_api_url, grok_model, top_k=2):
    """Main Q&A function with RAG pipeline"""
    top_chunks = retrieve_top_k(query, embedder, index, chunks, k=top_k)
    return grok_generate_answer(top_chunks, query, grok_api_key, grok_api_url, grok_model)

# ==================== QA Export Functions ====================
def create_qa_pdf(qa_list, filename="qa_pairs.pdf"):
    """Create a PDF file from Q&A pairs"""
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
        alignment=1
    )
    story.append(Paragraph("❓ Question & Answer Pairs", title_style))
    story.append(Spacer(1, 20))
    
    # Add each QA pair
    for i, qa in enumerate(qa_list, 1):
        # Question
        question_text = f"<b>Question {i}:</b> {qa['question']}"
        story.append(Paragraph(question_text, styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Answer
        answer_text = f"<b>Answer:</b> {qa['answer']}"
        story.append(Paragraph(answer_text, styles['Normal']))
        story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_qa_csv(qa_list):
    """Create a CSV file from Q&A pairs"""
    data = []
    for i, qa in enumerate(qa_list, 1):
        data.append({
            'Question Number': i,
            'Question': qa['question'],
            'Answer': qa['answer']
        })
    
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()
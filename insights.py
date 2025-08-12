import numpy as np
import requests
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import io
import re

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

def retrieve_top_k(query, embedder, index, chunks, k=2):
    """Retrieve top k most relevant chunks using FAISS index"""
    query_emb = embedder.encode([query]).astype("float32")
    D, I = index.search(query_emb, k)
    return [chunks[i] for i in I[0]]

def rag_generate_answer_insights(context_chunks, question, groq_api_key, groq_api_url, groq_model):
    """
    Generate insightful answers using RAG with Groq's LLM
    Args:
        context_chunks: Retrieved context chunks
        question: User's question
        groq_api_key: Groq API key
        groq_api_url: Groq API endpoint
        groq_model: Model name
    Returns:
        str: Insightful answer with real-world applications
    """
    if isinstance(context_chunks, list):
        context_str = "\n".join(context_chunks)
    else:
        context_str = str(context_chunks)
    
    prompt = (
        "You are an expert assistant. Use the context below to answer the question thoughtfully. "
        "Provide deep insights, real-world applications, and practical examples that go beyond "
        "the basic information in the context. Explain why this matters and how it applies in "
        "different scenarios. If the question can't be answered from the context, say so.\n\n"
        f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
    )
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": groq_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.5  # Balanced temperature for insightful yet focused answers
    }
    
    try:
        response = requests.post(groq_api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        raw_output = data.get("choices", [{}])[0].get("message", {}).get("content", "No answer generated.").strip()
        return clean_generated_text(raw_output)
    except Exception as e:
        return f"Error querying Groq API: {e}"

def answer_question(query, embedder, index, chunks, groq_api_key, groq_api_url, groq_model, top_k=2):
    """
    Answer a question with insights using RAG
    Args:
        query: User's question
        embedder: Sentence embedding model
        index: FAISS index
        chunks: Text chunks
        groq_api_key: Groq API key
        groq_api_url: Groq API endpoint
        groq_model: Model name
        top_k: Number of chunks to retrieve
    Returns:
        str: Insightful answer
    """
    top_chunks = retrieve_top_k(query, embedder, index, chunks, k=top_k)
    return rag_generate_answer_insights(top_chunks, query, groq_api_key, groq_api_url, groq_model)

def create_insights_pdf(qa_list, filename="insights_qa.pdf"):
    """Create a PDF file from insightful Q&A pairs"""
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
    story.append(Paragraph("💡 Insightful Q&A", title_style))
    story.append(Spacer(1, 20))
    
    # Add each QA pair
    for i, qa in enumerate(qa_list, 1):
        # Question
        question_text = f"<b>Question {i}:</b> {qa['question']}"
        story.append(Paragraph(question_text, styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Answer
        answer_text = f"<b>Insightful Answer:</b> {qa['answer']}"
        story.append(Paragraph(answer_text, styles['Normal']))
        story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_insights_csv(qa_list):
    """Create a CSV file from insightful Q&A pairs"""
    data = []
    for i, qa in enumerate(qa_list, 1):
        data.append({
            'Question Number': i,
            'Question': qa['question'],
            'Insightful Answer': qa['answer']
        })
    
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()
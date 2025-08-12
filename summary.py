import requests
import math
import io
import pandas as pd
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re

# Configuration parameters now come from the calling function
def grok_generate(prompt, grok_api_key, grok_api_url, grok_model, 
                 system_prompt=None, temperature=0.7, max_tokens=1024):
    """Helper to query Grok API. Automatically cleans <think> tags from output."""
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": grok_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(grok_api_url, headers=headers, json=payload)
        response.raise_for_status()
        raw_output = response.json()["choices"][0]["message"]["content"]
        return clean_generated_text(raw_output)  # Clean here
    except Exception as e:
        return f"[Error] {str(e)}"

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

def summarize_pdf(text, num_words, concept, grok_api_key, grok_api_url, grok_model):
    """
    Summarize the given text using Grok API.
    Args:
        text: Content to summarize
        num_words: Target summary length
        concept: Focus area
        grok_api_key: API key
        grok_api_url: API endpoint
        grok_model: Model to use
    """
    max_chunk_size = 10000
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    if concept.lower() == "entire pdf":
        concept_instruction = "Summarize the entire content comprehensively."
    else:
        concept_instruction = f"Focus exclusively on: {concept}. Omit unrelated information."

    if len(chunks) == 1:
        prompt = f"""Create a concise summary of approximately {num_words} words. 
        {concept_instruction}
        Content to summarize:
        {text}"""
        return grok_generate(prompt, grok_api_key, grok_api_url, grok_model)

    # Multi-chunk processing
    chunk_summaries = []
    words_per_chunk = max(30, math.floor(num_words / len(chunks)))

    for chunk in chunks:
        prompt = f"""Create a brief summary of about {words_per_chunk} words:
        {chunk}"""
        summary = grok_generate(prompt, grok_api_key, grok_api_url, grok_model)
        chunk_summaries.append(summary)

    final_prompt = f"""Combine these into one cohesive summary of {num_words} words:
    {" ".join(chunk_summaries)}
    Focus: {concept if concept.lower() != 'entire pdf' else 'all key aspects'}"""
    
    return grok_generate(final_prompt, grok_api_key, grok_api_url, grok_model)

def create_summary_pdf(summary_history, filename="summaries.pdf"):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter
    y_position = page_height - 50  # top margin
    c.setFont("Helvetica", 12)

    for i, summary_item in enumerate(summary_history, 1):
        # Summary number
        c.drawString(50, y_position, f"Summary {i}:")
        y_position -= 20

        # Metadata
        c.drawString(60, y_position, f"Original length: {summary_item['original_length']} words")
        y_position -= 15
        c.drawString(60, y_position, f"Requested summary length: {summary_item['requested_words']} words")
        y_position -= 20

        # Summary text with wrapping
        c.drawString(60, y_position, "Summary:")
        y_position -= 15

        wrap_width = 80  # characters per line
        for line in summary_item['summary_text'].split("\n"):
            wrapped_lines = textwrap.wrap(line, width=wrap_width)
            for wrapped_line in wrapped_lines:
                c.drawString(70, y_position, wrapped_line)
                y_position -= 15
                if y_position < 50:  # new page if near bottom
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y_position = page_height - 50

        y_position -= 20  # space between summaries

        if y_position < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = page_height - 50

    c.save()
    buffer.seek(0)
    return buffer

def create_summary_csv(summaries):
    """Create CSV content from summaries."""
    data = []
    for i, summary in enumerate(summaries, 1):
        data.append({
            "Summary #": i,
            "Original Length": summary['original_length'],
            "Requested Length": summary['requested_words'],
            "Summary Text": summary['summary_text']
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)
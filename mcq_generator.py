import requests
import re
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import re

def generate_mcqs(text, num_questions, difficulty, concept, groq_api_key, groq_api_url, groq_model):
    """
    Generate multiple choice questions from PDF text using Groq's LLM.
    Args:
        text (str): The PDF text content
        num_questions (int): Number of MCQs to generate
        difficulty (str): Difficulty level (Easy, Medium, Hard)
        concept(str): topic name or entire pdf
        groq_api_key (str): Groq API key
        groq_api_url (str): Groq API endpoint
        groq_model (str): Model name
    Returns:
        list: List of dictionaries with 'question', 'options', 'correct_answer' keys
    """
    # If text is too long, use a summary first
    max_text_length = 3000
    if len(text) > max_text_length:
        # Create a summary for MCQ generation
        summary_prompt = f"Summarize the following content in 500 words to create MCQs from:\n\n{text}"
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
    
    # Difficulty-specific instructions
    difficulty_instructions = {
        "Easy": "Create basic, straightforward questions that test fundamental understanding. Use simple language and obvious answer choices.",
        "Medium": "Create moderately challenging questions that test comprehension and application. Include some analysis and reasoning.",
        "Hard": "Create advanced questions that test deep understanding, critical thinking, and complex concepts. Include synthesis and evaluation."
    }

    # Concept-specific instructions
    if concept.lower() == "entire pdf":
        concept_instruction = "Use the contents in the entire PDF to generate MCQs based on the difficulty level."
    else:
        concept_instruction = f"Focus deeply ONLY on: **{concept}** (ignore unrelated content). All questions must be about this specific concept."
    
    # Generate MCQs
    prompt = f"""Create {num_questions} multiple choice questions from the following content.
                    Difficulty Level: {difficulty}
                    {difficulty_instructions.get(difficulty, difficulty_instructions["Medium"])}
                    Content Scope: 
                    {concept_instruction}

Each question should have exactly 4 options (A, B, C, D) with only one correct answer.
Format each MCQ as:
Q: [question]
A) [option A]
B) [option B] 
C) [option C]
D) [option D]
Correct Answer: [A/B/C/D]

Content:
{text}

MCQs:"""
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": groq_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(groq_api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        mcq_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        print("[DEBUG] Raw LLM MCQ response:\n", mcq_text)
        # Parse the MCQs
        mcqs = parse_mcqs(mcq_text)
        print(f"[DEBUG] Parsed MCQ list: {mcqs}")
        # If parsing failed, create simple MCQs
        if not mcqs:
            print("[DEBUG] Fallback: Creating simple MCQs")
            mcqs = create_simple_mcqs(text, num_questions, groq_api_key, groq_api_url, groq_model)
        return mcqs[:num_questions]
    except Exception as e:
        print(f"[DEBUG] Exception in generate_mcqs: {e}")
        return [{"question": f"Error generating MCQs: {e}", "options": ["A) Try again", "B) Try again", "C) Try again", "D) Try again"], "correct_answer": "A"}]

def create_simple_mcqs(text, num_questions, groq_api_key, groq_api_url, groq_model):
    """
    Create simple MCQs when parsing fails.
    """
    prompt = f"""From this content, create {num_questions} simple multiple choice questions.
Each question should have 4 options (A, B, C, D) with only one correct answer.
Format: "Q: [question] A) [option] B) [option] C) [option] D) [option] Correct Answer: [A/B/C/D]"

Content: {text[:2000]}

Simple MCQs:"""
    
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
        
        # Simple parsing
        mcqs = []
        lines = result.split('\n')
        current_q = None
        current_options = []
        current_correct = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                # Save previous MCQ if complete
                if current_q and len(current_options) == 4 and current_correct:
                    mcqs.append({
                        "question": current_q,
                        "options": current_options,
                        "correct_answer": current_correct
                    })
                
                # Start new MCQ
                current_q = line[2:].strip()
                current_options = []
                current_correct = None
            elif line.startswith(('A)', 'A.')):
                current_options.append(line[2:].strip())
            elif line.startswith(('B)', 'B.')):
                current_options.append(line[2:].strip())
            elif line.startswith(('C)', 'C.')):
                current_options.append(line[2:].strip())
            elif line.startswith(('D)', 'D.')):
                current_options.append(line[2:].strip())
            elif line.startswith('Correct Answer:'):
                current_correct = line.split(':')[1].strip().upper()
        
        # Add last MCQ
        if current_q and len(current_options) == 4 and current_correct:
            mcqs.append({
                "question": current_q,
                "options": current_options,
                "correct_answer": current_correct
            })
        
        return mcqs[:num_questions]
        
    except Exception as e:
        return [{"question": "What is the main topic?", "options": ["A) Various topics", "B) One topic", "C) No topic", "D) Multiple topics"], "correct_answer": "A"}]

def parse_mcqs(text):
    """
    Parse MCQ text into structured format.
    """
    mcqs = []
    
    # Split by question markers
    question_blocks = re.split(r'(?:^|\n)\s*Q[:\.]?\s*', text, flags=re.IGNORECASE)
    
    for block in question_blocks:
        block = block.strip()
        if not block:
            continue
            
        # Extract question and options
        lines = block.split('\n')
        if len(lines) < 5:  # Need question + 4 options + correct answer
            continue
            
        question = lines[0].strip()
        if not question:
            continue
            
        options = []
        correct_answer = None
        
        for line in lines[1:]:
            line = line.strip()
            if line.startswith(('A)', 'A.')):
                options.append(line[2:].strip())
            elif line.startswith(('B)', 'B.')):
                options.append(line[2:].strip())
            elif line.startswith(('C)', 'C.')):
                options.append(line[2:].strip())
            elif line.startswith(('D)', 'D.')):
                options.append(line[2:].strip())
            elif line.startswith('Correct Answer:'):
                correct = line.split(':')[1].strip().upper()
                if correct in ['A', 'B', 'C', 'D']:
                    correct_answer = correct
        
        if len(options) == 4 and correct_answer:
            mcqs.append({
                "question": question,
                "options": options,
                "correct_answer": correct_answer
            })
    
    return mcqs

def create_simple_mcqs(text, num_questions, ollama_base_url, ollama_model):
    """
    Create simple MCQs when parsing fails.
    """
    prompt = f"""From this content, create {num_questions} simple multiple choice questions.
Each question should have 4 options (A, B, C, D) with only one correct answer.
Format: "Q: [question] A) [option] B) [option] C) [option] D) [option] Correct Answer: [A/B/C/D]"

Content: {text[:2000]}

Simple MCQs:"""
    
    url = f"{ollama_base_url}/api/generate"
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 1024,
            "stop": ["\n\n\n\n"]
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        result = data.get("response", "").strip()
        
        # Simple parsing
        mcqs = []
        lines = result.split('\n')
        current_q = None
        current_options = []
        current_correct = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                # Save previous MCQ if complete
                if current_q and len(current_options) == 4 and current_correct:
                    mcqs.append({
                        "question": current_q,
                        "options": current_options,
                        "correct_answer": current_correct
                    })
                
                # Start new MCQ
                current_q = line[2:].strip()
                current_options = []
                current_correct = None
            elif line.startswith(('A)', 'A.')):
                current_options.append(line[2:].strip())
            elif line.startswith(('B)', 'B.')):
                current_options.append(line[2:].strip())
            elif line.startswith(('C)', 'C.')):
                current_options.append(line[2:].strip())
            elif line.startswith(('D)', 'D.')):
                current_options.append(line[2:].strip())
            elif line.startswith('Correct Answer:'):
                current_correct = line.split(':')[1].strip().upper()
        
        # Add last MCQ
        if current_q and len(current_options) == 4 and current_correct:
            mcqs.append({
                "question": current_q,
                "options": current_options,
                "correct_answer": current_correct
            })
        
        return mcqs[:num_questions]
        
    except Exception as e:
        return [{"question": "What is the main topic?", "options": ["A) Various topics", "B) One topic", "C) No topic", "D) Multiple topics"], "correct_answer": "A"}]

def calculate_score(user_answers, mcqs):
    """
    Calculate the score based on user answers.
    """
    if not user_answers or not mcqs:
        return 0, 0
    
    correct = 0
    total = len(mcqs)
    
    for i, mcq in enumerate(mcqs):
        if i < len(user_answers) and user_answers[i] == mcq['correct_answer']:
            correct += 1
    
    return correct, total 

def create_mcqs_pdf(mcqs_list, filename="mcqs.pdf"):
    """Create a PDF file from MCQs"""
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
    story.append(Paragraph("📋 Multiple Choice Questions", title_style))
    story.append(Spacer(1, 20))
    
    # Add each MCQ
    for i, mcq in enumerate(mcqs_list, 1):
        # Question
        question_text = f"<b>Question {i}:</b> {mcq['question']}"
        story.append(Paragraph(question_text, styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Options
        options_text = ""
        for j, option in enumerate(mcq['options']):
            option_letter = chr(65 + j)  # A, B, C, D
            options_text += f"{option_letter}) {option}<br/>"
        story.append(Paragraph(options_text, styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Correct Answer (robust mapping)
        answer_letter = str(mcq['correct_answer']).strip()[0] if mcq['correct_answer'] else 'A'
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        correct_index = answer_map.get(answer_letter, 0)
        correct_text = f"<b>Correct Answer:</b> {answer_letter}) {mcq['options'][correct_index]}"
        story.append(Paragraph(correct_text, styles['Normal']))
        story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_mcqs_csv(mcqs_list):
    """Create a CSV file from MCQs"""
    data = []
    for i, mcq in enumerate(mcqs_list, 1):
        options_text = " | ".join([f"{chr(65+j)}) {option}" for j, option in enumerate(mcq['options'])])
        data.append({
            'Question Number': i,
            'Question': mcq['question'],
            'Options': options_text,
            'Correct Answer': f"{mcq['correct_answer']}) {mcq['options'][ord(mcq['correct_answer']) - 65]}"
        })
    
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue() 
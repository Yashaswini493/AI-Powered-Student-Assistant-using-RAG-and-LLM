import os
import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import time
import requests
import re 
import summary
import quest_ans
import flashcards
import mcq_generator
import exp_5
import insights
from styles_main import FEATURE_CARDS_CSS
import base64
from pathlib import Path
import uuid
from dotenv import load_dotenv

load_dotenv() 

def apply_page_style(feature_index=None):
    """Applies colored tab styling matching the feature card colors"""
    tab_colors = {
        0: ("#E7E3FF", "#5E4FA2"),  # Lavender (bg, active text)
        1: ("#D4E6FF", "#1A73E8"),  # Pastel Blue
        2: ("#DFFFD6", "#0B8043"),  # Mint Green
        3: ("#D7F0FF", "#039BE5"),  # Sky Blue
        4: ("#FFDCE5", "#D81B60"),  # Blush Pink
        5: ("#FFF2D9", "#F09300")   # Pastel Yellow
    }
    
    # Base CSS injection
    st.markdown(FEATURE_CARDS_CSS, unsafe_allow_html=True)
    
    # Tab color styling
    if feature_index in tab_colors:
        bg_color, text_color = tab_colors[feature_index]
        tab_style = f"""
        <style>
            /* Colored tabs */
            .stTabs [data-baseweb="tab"] {{
                background-color: {bg_color} !important;
                color: #333 !important;
                border-radius: 8px !important;
                margin-right: 8px !important;
                padding: 10px 20px !important;
                transition: all 0.3s ease !important;
            }}
            
            .stTabs [data-baseweb="tab"]:hover {{
                transform: translateY(-2px);
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            
            .stTabs [aria-selected="true"] {{
                background-color: {bg_color} !important; /* light grey */
                color: {text_color} !important;
                font-weight: 600 !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-bottom: 3px solid {text_color} !important;
            }}
            
            /* Tab content border */
            .stTab [data-baseweb="tab-panel"] {{
                border-left: 1px solid #eee;
                border-right: 1px solid #eee;
                border-bottom: 1px solid #eee;
                border-radius: 0 0 8px 8px;
                padding: 20px;
            }}
        </style>
        """
        st.markdown(FEATURE_CARDS_CSS, unsafe_allow_html=True)

# --- Page Config (must be first) ---
st.set_page_config(
    page_title="AI-Powered Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)
apply_page_style()

# Convert ANY local image to work in Streamlit
def get_img_base64(path):
    path = Path(path).expanduser().absolute()  # Handles ~/ paths
    return base64.b64encode(path.read_bytes()).decode()

bg_base64 = get_img_base64("bg_img.png")  # Or full path like "~/Downloads/bg_img.png"

st.markdown(
    f"""
    <style>
        [data-testid="stAppViewContainer"] > .main {{
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Make content area semi-transparent */
        .st-emotion-cache-1v0mbdj {{
            background-color: rgba(255, 255, 255, 0.9) !important;
            border-radius: 10px;
            padding: 2% !important;
        }}
        
        /* Fix header colors */
        h1, h2, h3, h4, h5, h6 {{
            color: #000000 !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
) 

# --- Config ---
GROK_API_KEY = os.getenv("GROK_API_KEY")  # You'll need to set this
GROK_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Update if different
GROK_MODEL = "qwen/qwen3-32b"
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
TOP_K = 3

def grok_api_call(prompt, system_prompt=None):
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": GROK_MODEL,
        "messages": messages,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(GROK_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error calling Grok API: {str(e)}")
        return None

# --- Helper Functions (from chunk-exp.py) ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()+"\n"
    return text

def split_into_paragraphs(text):
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

def embed_chunks(chunks, embedder):
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    return np.array(embeddings).astype("float32")

def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index



# --- Streamlit UI ---
st.markdown("""
<div style="
    text-align: center;
    background-color: rgba(255, 255, 255, 0.5);
    border-radius: 8px;
    border: 1px solid rgba(150, 150, 150, 0.3);
    padding: 1rem;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: black; !important
    display: inline-block;
    display: flex !important;
    justify-content: center !important;
    gap: 8px !important;
    margin-bottom: 1.5rem !important;
    padding: 5px 30px !important;
">
    <h1 style="margin: 0;">🧠 AI-Powered Study Assistant</h1>
    <h6 style="margin: 0; font-weight: normal;">
        Upload a PDF and explore it using various intelligent tools
    </h6>
</div>
""", unsafe_allow_html=True)


# --- PDF Upload Block ---
# --- Transparent File Uploader Style with Black Text ---
st.markdown("""
<style>
[data-testid="stFileUploader"] {
    background-color: rgba(255, 255, 255, 0.6) !important; /* Transparent white */
    border-radius: 8px !important;
    border: 2px dashed rgba(150, 150, 150, 0.5) !important;
    padding: 1rem !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: black !important;
}

[data-testid="stFileUploader"] section {
    background-color: transparent !important;
    color: black !important;
}

[data-testid="stFileUploader"] label {
    color: black !important;
    font-weight: 500;
}

[data-testid="stFileUploader"] div[role="button"] {
    background-color: rgba(255, 255, 255, 0.4) !important;
    border: 1px solid rgba(150, 150, 150, 0.3) !important;
    border-radius: 5px !important;
    color: black !important;
}

/* Target the "Limit 200MB per file" small text */
[data-testid="stFileUploader"] small {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)


# --- File Upload Logic ---
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.chunks = None
    st.session_state.embedder = None
    st.session_state.text = None
    st.session_state.filename = None

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting and processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = split_into_paragraphs(text)
        embedder = load_embedder()
        embeddings = embed_chunks(chunks, embedder)
        index = build_faiss_index(embeddings)
        st.session_state.faiss_index = index
        st.session_state.chunks = chunks
        st.session_state.embedder = embedder
        st.session_state.text = text
        st.session_state.filename = uploaded_file.name
    st.success("✅ PDF uploaded and processed successfully!")



# --- Info Box ---
if st.session_state.text:
    num_words = len(st.session_state.text.split())
    num_chunks = len(st.session_state.chunks)
    container = st.container()
    with container:
        with st.expander("ℹ️ PDF Info", expanded=True):
            st.write(f"**File:** {st.session_state.filename}")
            st.write(f"**Word count:** {num_words}")
            #st.write(f"**Chunk count:** {num_chunks}")

# --- Tabs Navigation ---
tabs = st.tabs([
    "Summarization 📜",
    "Question Answering ❓",
    "Flashcards 📇",
    "MCQ Generator ☑️",
    "Explain Like I'm 5 👶",
    "Out-of-PDF Insights 💡"
])

# --- Summarization Tab ---
with tabs[0]:
    # Create a container and apply styling directly to it
    container = st.container()
    # Apply the bento box styling to this specific container
    container.markdown(
        """
        <style>
            /* Main container styling with hover effect */
            div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
                background-color: rgba(252, 252, 252, 0.8) !important;
                border-radius: 8px !important;
                padding: 1rem !important;
                border: 2px solid rgba(252, 252, 252, 0.8) !important;
                box-shadow: 4px 6px 0px #d2cfe3 !important;
                margin-bottom: 2rem !important;
                transition: all 0.3s ease !important;
            }
            
            /* Hover state - becomes fully opaque */
            div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
                background-color: rgba(252, 252, 252, 1) !important;
                border: 2px solid rgba(252, 252, 252, 1) !important;
                box-shadow: 4px 6px 0px #d2cfe3, 0 0 15px rgba(0,0,0,0.1) !important;
            }
            
            /* Form styling with hover effect */
            div[data-testid="stForm"] {
                border: 2px solid #333 !important;
                border-radius: 8px !important;
                padding: 1rem !important;
                margin: 1rem 0 !important;
                background-color: rgba(252, 252, 252, 0) !important;
                transition: all 0.3s ease !important;
            }
            
            /* Form hover state */
            div[data-testid="stForm"]:hover {
                background-color: rgba(252, 252, 252, 0.9) !important;
            }
            
            /* Input and button styling (unchanged) */
            div[data-testid="stForm"] .stNumberInput input {
                border: 2px solid #8FA6E0 !important;
                background-color: rgba(143, 166, 224, 0.1) !important;
                color: #fff !important;
            }
            div[data-testid="stForm"] button {
                background-color: #8FA6E0 !important;
                border: none !important;
            }
            div[data-testid="stForm"] button:hover {
                background-color: #7A93D1 !important;
                transform: translateY(-2px);
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    with container:
        col1, col2 = st.columns([0.9, 0.05])
        with col1:
            st.header("Summarization")
        with col2:
            if st.session_state.get("summary_history"):
                if st.button("🗑️", key="clear_all", help="Clear all history"):
                    st.session_state.summary_history.clear()
                    st.rerun()
        
        st.info("Generate a concise summary of your PDF.")
        
        if st.session_state.text:
            if 'summary_history' not in st.session_state:
                st.session_state.summary_history = []
            
            for idx, summary_item in enumerate(st.session_state.summary_history):
                col_hist, col_del = st.columns([0.95, 0.05])
                with col_hist:
                    st.markdown(f"**Original length:** {summary_item['original_length']} words")
                    st.markdown(f"**Requested summary length:** {summary_item['requested_words']} words")
                    st.markdown(f"**Summary:**\n\n{summary_item['summary_text']}")
                    st.markdown(
                    """
                    <style>
                    hr {
                        border: none;
                        border-top: 2px solid #a8a8a8 !important;
                        margin-top: 1rem;
                        margin-bottom: 1rem;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                    st.divider()
                with col_del:
                    if st.button("❌", key=f"del_{idx}", help="Delete this entry"):
                        del st.session_state.summary_history[idx]
                        st.rerun()

            if st.session_state.summary_history:
                col1, col2 = st.columns(2)
                with col1:
                    pdf_buffer = summary.create_summary_pdf(st.session_state.summary_history)
                    st.download_button(
                        label="📄 Download as PDF",
                        data=pdf_buffer.getvalue(),
                        file_name="Summary.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                with col2:
                    csv_data = summary.create_summary_csv(st.session_state.summary_history)
                    st.download_button(
                        label="📊 Download as CSV",
                        data=csv_data,
                        file_name="Summary.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                        
            # This form will now have dark gray (#a8a8a8) bento styling
            with st.form(key='summary_form'):
                col1, col2 = st.columns(2)
                with col1:
                    num_words = st.number_input(
                    "How many words should the summary be?", 
                    min_value=50, 
                    max_value=1000, 
                    value=200, 
                    step=10,
                    key=f"words_{len(st.session_state.summary_history)}"
                )
                with col2:
                    concept_input = st.text_input(
                        "📌 Focus on:",
                        value="Entire PDF",
                        placeholder="e.g., Transformers"
                    )
                    concept_mode = "single" if concept_input != "Entire PDF" else "entire"
                    if concept_input.strip().lower() == "entire pdf":
                        concept_mode = "entire"
                    else:
                        concept_mode = "single"
                        concept_name = concept_input
                submit_button = st.form_submit_button("Generate Summary")
                
                if submit_button:
                    with st.spinner("Generating summary..."):
                        summary_text = summary.summarize_pdf(
                            st.session_state.text,
                            num_words,
                            concept_input if concept_mode == "single" else "Entire PDF",
                            GROK_API_KEY,
                            GROK_API_URL,
                            GROK_MODEL
                        )
                        st.session_state.summary_history.append({
                            'original_length': len(st.session_state.text.split()),
                            'requested_words': num_words,
                            'summary_text': summary_text
                        })
                        st.rerun()
        else:
            st.warning("Please upload and process a PDF first.")

# --- Question Answering Tab ---
with tabs[1]:
    container = st.container()
    with container:
        # Header with clear all option
        col1, col2 = st.columns([0.9, 0.05])
        with col1:
            st.header("Question Answering")
        with col2:
            if st.session_state.get("qa_history"):
                if st.button("🗑️", key="clear_all_qa", help="Clear all history"):
                    st.session_state.qa_history.clear()
                    st.rerun()

        st.info("Ask questions about your PDF. Uses RAG for context-aware answers.")

        if st.session_state.faiss_index:
            if 'qa_history' not in st.session_state:
                st.session_state.qa_history = []

            for idx, qa in enumerate(st.session_state.qa_history):
                col_hist, col_del = st.columns([0.95, 0.05])
                with col_hist:
                    st.markdown(f"**Q: {qa['question']}**")
                    st.markdown(f"**A:** {qa['answer']}")
                    st.markdown(
                        """
                        <style>
                        hr {
                            border: none;
                            border-top: 2px solid #a8a8a8 !important;
                            margin-top: 1rem;
                            margin-bottom: 1rem;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    st.divider()
                with col_del:
                    if st.button("❌", key=f"del_qa_{idx}", help="Delete this entry"):
                        del st.session_state.qa_history[idx]
                        st.rerun()

            # Download buttons
            if st.session_state.qa_history:
                col1, col2 = st.columns(2)
                with col1:
                        pdf_buffer = quest_ans.create_qa_pdf(st.session_state.qa_history)
                        st.download_button(
                            label="📄 Download as PDF",
                            data=pdf_buffer.getvalue(),
                            file_name="question_answer.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                with col2:
                        csv_data = quest_ans.create_qa_csv(st.session_state.qa_history)
                        st.download_button(
                            label="📊 Download as CSV",
                            data=csv_data,
                            file_name="question_answer.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

            with st.form(key='qa_form'):
                # Modified text area to match summarization tab style
                user_query = st.text_input(  # Changed from text_area to text_input
                    "Ask a question about the PDF:", 
                    key=f"question_{len(st.session_state.qa_history)}"
                )
                
                # Custom CSS to make the input match summarization style
                st.markdown(
                    """
                    <style>
                        div[data-testid="stForm"] .stTextInput input {
                            height: 40px !important;
                            padding: 10px !important;
                        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                submit_button = st.form_submit_button("Ask a Question")
                
                if submit_button and user_query:
                    with st.spinner("Retrieving answer..."):
                        answer = quest_ans.answer_question(
                            user_query,
                            st.session_state.embedder,
                            st.session_state.faiss_index,
                            st.session_state.chunks,
                            GROK_API_KEY,
                            GROK_API_URL,
                            GROK_MODEL
                        )
                        st.session_state.qa_history.append({
                            'question': user_query,
                            'answer': answer
                        })
                        st.rerun()
        else:
            st.warning("Please upload and process a PDF first.")

# --- Flashcards Tab ---
with tabs[2]:
    container = st.container()
    
    with container:
        # Header + Clear All option
        col1, col2 = st.columns([0.9, 0.05])
        with col1:
            st.header("Flashcards")
        with col2:
            if st.session_state.get("flashcard_history"):
                if st.button("🗑️", key="clear_all_flashcards", help="Clear all history"):
                    st.session_state.flashcard_history.clear()
                    st.rerun()

        st.info("Generate flashcards from your PDF for active recall.")
        
        # Initialize session state for flashcard history
        if 'flashcard_history' not in st.session_state:
            st.session_state.flashcard_history = []
        
        if st.session_state.text:
            # Display previous flashcard generations
            for idx, generation in enumerate(st.session_state.flashcard_history, 1):
                col_hist, col_del = st.columns([0.95, 0.05])
                with col_hist:
                    st.markdown(f"**Generation {idx}** (Requested: {generation['requested_cards']} cards)")
                    for i, card in enumerate(generation['cards'], 1):
                        with st.expander(f"Card {i}: {card['question'][:50]}...", expanded=False):
                            # Add grey border styling
                            st.markdown(
                                """
                                <style>
                                    div[data-testid="stExpander"] {
                                        border: 1px solid #CCCCCC !important;
                                        border-radius: 8px !important;
                                        padding: 12px !important;
                                    }
                                    div[data-testid="stExpander"]:hover {
                                        border-color: #999999 !important;
                                    }
                                    div[data-testid="stExpander"] > div:first-child {
                                        background-color: #F8F9FA !important;
                                        padding: 8px 12px !important;
                                    }
                                </style>
                                """,
                                unsafe_allow_html=True
                            )
                            st.markdown(f"**Question:** {card['question']}")
                            st.markdown(f"**Answer:** {card['answer']}")
                with col_del:
                    if st.button("❌", key=f"del_flash_{idx}", help="Delete this generation"):
                        del st.session_state.flashcard_history[idx-1]  # idx is 1-based
                        st.rerun()
                
                # Divider between generations
                st.markdown("<hr style='border-top: 2px solid #a8a8a8; margin: 1.5rem 0;'>", unsafe_allow_html=True)

            st.markdown("""
            <style>
                /* Grey download buttons */
                div.stDownloadButton > button {
                    background-color: #7bc7ab !important;
                    color: white !important;
                    border: 1px solid #7bc7ab !important;
                }
                div.stDownloadButton > button:hover {
                    background-color: #3fab83 !important;
                    border: 1px solid #3fab83 !important;
                    color: white !important;
                    transform: translateY(-2px);
                }
                div.stDownloadButton > button:focus {
                    box-shadow: rgb(128 128 128 / 50%) 0 0 0 0.2rem;
                }
            </style>
            """, unsafe_allow_html=True)

            # Your existing download buttons code
            if st.session_state.flashcard_history:
                #st.subheader("Download Options")
                all_cards = [card for gen in st.session_state.flashcard_history for card in gen['cards']]
                
                col1, col2 = st.columns(2)
                with col1:
                    pdf_buffer = flashcards.create_flashcards_pdf(all_cards)
                    st.download_button(
                        label="📄 Download All as PDF",
                        data=pdf_buffer.getvalue(),
                        file_name="all_flashcards.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                with col2:
                    csv_data = flashcards.create_flashcards_csv(all_cards)
                    st.download_button(
                        label="📊 Download All as CSV",
                        data=csv_data,
                        file_name="all_flashcards.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            # Current flashcard generation form
            with st.form(key='flashcard_form'):
                col1, col2 = st.columns(2)
                with col1:
                    num_cards = st.number_input(
                        "How many flashcards do you want to generate?",
                        min_value=3,
                        max_value=20,
                        value=5,
                        step=1,
                        key=f"num_cards_{len(st.session_state.flashcard_history)}"
                    )
                with col2:
                    concept_input = st.text_input(
                        "📌 Focus on:",
                        value="Entire PDF",
                        placeholder="e.g., Transformers"
                    )
                    concept_mode = "single" if concept_input != "Entire PDF" else "entire"
                    if concept_input.strip().lower() == "entire pdf":
                        concept_mode = "entire"
                    else:
                        concept_mode = "single"
                        concept_name = concept_input
                
                submit_button = st.form_submit_button("Generate Flashcards")
                
                if submit_button:
                    with st.spinner("Creating flashcards..."):
                        flashcard_list = flashcards.generate_flashcards(
                            st.session_state.text,
                            num_cards,
                            concept_input if concept_mode == "single" else "Entire PDF",
                            GROK_API_KEY,
                            GROK_API_URL,
                            GROK_MODEL
                        )
                        
                        # Handle potential parsing issues
                        if len(flashcard_list) == 1:
                            qtext = flashcard_list[0]['question']
                            atext = flashcard_list[0]['answer']
                            if any(x in qtext for x in ['\nQ:', '\nQuestion:', 'Flashcard']) or any(x in atext for x in ['\nQ:', '\nQuestion:', 'Flashcard']):
                                text = qtext + '\nA: ' + atext
                                parsed = flashcards.parse_flashcards(text)
                                if parsed:
                                    flashcard_list = parsed
                        
                        if flashcard_list:
                            st.session_state.flashcard_history.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'requested_cards': num_cards,
                                'actual_cards': len(flashcard_list),
                                'cards': flashcard_list
                            })
                            st.rerun()
                        else:
                            st.error("Failed to generate flashcards. Please try again.")
        else:
            st.warning("Please upload and process a PDF first.")

# --- MCQ Generator Tab ---
with tabs[3]:
    container = st.container()
    with container:
        # Header with clear all option
        col1, col2 = st.columns([0.9, 0.05])
        with col1:
            st.header("MCQ Generator")
        with col2:
            if st.session_state.get("mcqs"):
                if st.button("🗑️", key="clear_all_mcqs", help="Clear all MCQs"):
                    st.session_state.pop("mcqs", None)
                    st.session_state.pop("user_answers", None)
                    st.rerun()

        st.info("Generate multiple-choice questions from your PDF.")

        # Only show the form if PDF text exists
        # Only show the form if PDF text exists
        if 'text' in st.session_state and st.session_state.text:
            # Display history/questions section above the form
            
            if 'mcqs' in st.session_state and st.session_state.mcqs:
                st.subheader("📝 Answer the questions:")
                
                # Add custom styling for questions and buttons
                st.markdown(
                                """
                                <style>
                                    div[data-testid="stExpander"] {
                                        border: 1px solid #CCCCCC !important;
                                        border-radius: 8px !important;
                                        padding: 12px !important;
                                    }
                                    div[data-testid="stExpander"]:hover {
                                        border-color: #999999 !important;
                                    }
                                    div[data-testid="stExpander"] > div:first-child {
                                        background-color: #F8F9FA !important;
                                        padding: 8px 12px !important;
                                    }
                                </style>
                                """,
                                unsafe_allow_html=True
                            )

                for i, mcq in enumerate(st.session_state.mcqs):
                    # Create a container for each question with custom styling
                    with st.container():
                        st.markdown(f'<div class="question-container">', unsafe_allow_html=True)
                        
                        col_mcq, col_del = st.columns([0.95, 0.05])
                        with col_mcq:
                            st.markdown(f"**Question {mcq['question']}**")
                            option_labels = [f"{chr(65+j)}) {option}" for j, option in enumerate(mcq['options'])]
                            selected = st.radio(
                                f"Select your answer for Question {i+1}:",
                                option_labels,
                                key=f"mcq_{i}",
                                index=None
                            )
                            if selected:
                                st.session_state.user_answers[i] = chr(65 + option_labels.index(selected))
                        with col_del:
                            if st.button("❌", key=f"del_mcq_{i}", help="Delete this question"):
                                del st.session_state.mcqs[i]
                                del st.session_state.user_answers[i]
                                st.rerun()
                        
                        st.markdown('</div>', unsafe_allow_html=True)

                # Add submit button after all questions
                st.markdown("""
                <style>
                    /* Submit button styling */
                    div.stButton > button:first-child {
                        background-color: #8FA6E0 !important;
                        color: white !important;
                        border: none !important;
                        padding: 0.5rem 1rem !important;
                        border-radius: 4px !important;
                        font-weight: 500 !important;
                        transition: all 0.2s ease !important;
                    }
                    
                    div.stButton > button:first-child:hover {
                        background-color: #7A93D1 !important;
                        transform: translateY(-1px);
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
                    }
                    
                    div.stButton > button:first-child:active {
                        transform: translateY(0) !important;
                    }
                    
                    div.stButton > button:first-child:focus:not(:active) {
                        box-shadow: 0 0 0 0.2rem rgba(74, 143, 231, 0.5) !important;
                    }
                </style>
                """, unsafe_allow_html=True)
                if st.button("✅ Submit Answers", use_container_width=True, key="submit-quiz"):
                    if None in st.session_state.user_answers:
                        st.warning("Please answer all questions before submitting.")
                    else:
                        correct, total = mcq_generator.calculate_score(st.session_state.user_answers, st.session_state.mcqs)
                        percentage = (correct / total) * 100
                        
                        st.subheader("🎯 Quiz Results")
                        st.metric("Score", f"{correct}/{total}", f"{percentage:.1f}%")
                        
                        if percentage >= 80:
                            st.success("🎉 Excellent! Great job!")
                        elif percentage >= 60:
                            st.info("👍 Good work! Keep studying!")
                        else:
                            st.warning("📚 Keep practicing! Review the material.")
                        
                        with st.expander("📖 Review Correct Answers"):
                            st.markdown(
                                """
                                <style>
                                    div[data-testid="stExpander"] {
                                        border: 1px solid #CCCCCC !important;
                                        border-radius: 8px !important;
                                        padding: 12px !important;
                                    }
                                    div[data-testid="stExpander"]:hover {
                                        border-color: #999999 !important;
                                    }
                                    div[data-testid="stExpander"] > div:first-child {
                                        background-color: #F8F9FA !important;
                                        padding: 8px 12px !important;
                                    }
                                </style>
                                """,
                                unsafe_allow_html=True
                            )
                            for i, mcq in enumerate(st.session_state.mcqs):
                                user_ans = st.session_state.user_answers[i]
                                correct_ans = mcq['correct_answer']
                                is_correct = user_ans == correct_ans
                                
                                status = "✅" if is_correct else "❌"
                                st.markdown(f"{status} **Question {i+1}:** {mcq['question']}")
                                answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                                user_index = answer_map.get(user_ans, 0)
                                st.markdown(f"Your answer: {user_ans}) {mcq['options'][user_index]}")
                                if not is_correct:
                                    correct_index = answer_map.get(correct_ans, 0)
                                    st.markdown(f"Correct answer: {correct_ans}) {mcq['options'][correct_index]}")
                                st.divider()

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    pdf_buffer = mcq_generator.create_mcqs_pdf(st.session_state.mcqs)
                    st.download_button(
                        label="📄 Download as PDF",
                        data=pdf_buffer.getvalue(),
                        file_name="mcqs.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                with col2:
                    csv_data = mcq_generator.create_mcqs_csv(st.session_state.mcqs)
                    st.download_button(
                        label="📊 Download as CSV",
                        data=csv_data,
                        file_name="mcqs.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            # Input form for new MCQs
            with st.form(key='mcq_form'):
                col1, col2 = st.columns(2)
                with col1:
                    num_questions = st.number_input(
                        "How many MCQs do you want?", 
                        min_value=3, 
                        max_value=15, 
                        value=5, 
                        step=1
                    )
                with col2:
                    concept_input = st.text_input(
                        "📌 Focus on:",
                        value="Entire PDF",
                        placeholder="e.g., Transformers"
                    )
                    concept_mode = "single" if concept_input != "Entire PDF" else "entire"
                    if concept_input.strip().lower() == "entire pdf":
                        concept_mode = "entire"
                    else:
                        concept_mode = "single"
                        concept_name = concept_input
                
                target = st.radio(
                    "What's your goal?",
                    ["🎯 Quick Review", "📚 Thorough Understanding", "🧠 Master the Topic"],
                    help="Quick Review: Basic concepts | Thorough Understanding: Analysis & application | Master the Topic: Critical thinking & synthesis",
                    horizontal=True,
                    label_visibility="visible"
                )
                difficulty = {
                    "🎯 Quick Review": "Easy",
                    "📚 Thorough Understanding": "Medium", 
                    "🧠 Master the Topic": "Hard"
                }[target]

                submit_button = st.form_submit_button("Generate MCQs")
            
                if submit_button:
                    with st.spinner("Creating MCQs..."):
                        mcq_list = mcq_generator.generate_mcqs(
                            st.session_state.text,
                            num_questions,
                            difficulty,
                            concept_input if concept_mode == "single" else "Entire PDF",
                            GROK_API_KEY,
                            GROK_API_URL,
                            GROK_MODEL
                        )
                        if mcq_list:
                            st.session_state.mcqs = mcq_list
                            st.session_state.user_answers = [None] * len(mcq_list)
                            st.success(f"✅ Generated {len(mcq_list)} MCQs for {target}!")
                            st.rerun()
                        else:
                            st.error("Failed to generate MCQs. Please try again.")
        else:
            st.warning("Please upload and process a PDF first to generate MCQs.")
            
# --- Explain Like I'm 5 Tab ---
with tabs[4]:
    container = st.container()
    with container:
        # Header with Clear All option (matching Summarization style)
        col1, col2 = st.columns([0.9, 0.05])
        with col1:
            st.header("Explain Like I'm 5 (ELI5)")
        with col2:
            if st.session_state.get("eli5_history"):
                if st.button("🗑️", key="clear_all_eli5", help="Clear all history"):
                    st.session_state.eli5_history.clear()
                    st.rerun()

        st.info("Paste a complex answer or concept to get a simple explanation.")

        # History section
        if st.session_state.faiss_index:
            if 'eli5_history' not in st.session_state:
                st.session_state.eli5_history = []

            for idx, eli5_item in enumerate(st.session_state.eli5_history):
                col_hist, col_del = st.columns([0.95, 0.05])
                with col_hist:
                    st.markdown(f"**Q: {eli5_item['question']}**")
                    st.markdown(f"**A:** {eli5_item['answer']}")
                    st.markdown(
                        """
                        <style>
                        hr {
                            border: none;
                            border-top: 2px solid #a8a8a8 !important;
                            margin-top: 1rem;
                            margin-bottom: 1rem;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    st.divider()
                with col_del:
                    if st.button("❌", key=f"del_eli5_{idx}", help="Delete this entry"):
                        del st.session_state.eli5_history[idx]
                        st.rerun()
            
            if st.session_state.eli5_history:
                col1, col2 = st.columns(2)
                with col1:
                    pdf_buffer = exp_5.create_eli5_pdf(st.session_state.eli5_history)
                    st.download_button(
                        label="📄 Download as PDF",
                        data=pdf_buffer.getvalue(),
                        file_name="eli5.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                with col2:
                    csv_data = exp_5.create_eli5_csv(st.session_state.eli5_history)
                    st.download_button(
                        label="📊 Download as CSV",
                        data=csv_data,
                        file_name="eli5.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            # Input form
            with st.form(key='eli5_form'):
                if st.session_state.faiss_index:
                    user_query_5 = st.text_input(
                        "Ask a question about the PDF and get simplified answers:"
                    )
                    if user_query_5:
                        eli5_query = f"Explain like I'm 5 years old: {user_query_5}"
                submit_button = st.form_submit_button("Ask a Question")

                if submit_button and 'eli5_query' in locals():
                    with st.spinner("Retrieving answer..."):
                        answer_5 = exp_5.answer_question(
                            eli5_query,
                            st.session_state.embedder,
                            st.session_state.faiss_index,
                            st.session_state.chunks,
                            GROK_API_KEY,
                            GROK_API_URL,
                            GROK_MODEL
                        )
                        st.markdown(f"**Answer:** {answer_5}")
                        st.session_state.eli5_history.append({
                            'question': user_query_5,
                            'answer': answer_5
                        })
                        st.rerun()
        else:
            st.warning("Please upload and process a PDF first.")

# --- Out-of-PDF Insights Tab ---
with tabs[5]:
    container = st.container()
    with container:
        # Two-column header with "Clear All"
        col1, col2 = st.columns([0.9, 0.05])
        with col1:
            st.header("Out-of-PDF Insights")
        with col2:
            if st.session_state.get("insights_history"):
                if st.button("🗑️", key="clear_insights", help="Delete all insights"):
                    st.session_state.insights_history = []
                    st.rerun()

        st.info("Ask advanced or application-level questions for LLM-powered insights.")

        if st.session_state.faiss_index:
            if st.session_state.text:
                if 'insights_history' not in st.session_state:
                    st.session_state.insights_history = []

                # Display past insights
                for idx, qa in enumerate(st.session_state.insights_history):
                    col_hist, col_del = st.columns([0.95, 0.05])
                    with col_hist:
                        st.markdown(f"**Q: {qa['question']}**")
                        st.markdown(f"**A:** {qa['answer']}")
                        st.markdown(
                            """
                            <style>
                            hr {
                                border: none;
                                border-top: 2px solid #a8a8a8 !important;
                                margin-top: 1rem;
                                margin-bottom: 1rem;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
                        st.divider()
                    with col_del:
                        if st.button("❌", key=f"del_insight_{qa['question'][:10]}_{idx}", help="Delete this entry"):
                            del st.session_state.insights_history[idx]
                            st.rerun()

                if st.session_state.insights_history:
                    col1, col2 = st.columns(2)
                    with col1:
                        pdf_buffer = insights.create_insights_pdf(st.session_state.insights_history)
                        st.download_button(
                            label="📄 Download as PDF",
                            data=pdf_buffer.getvalue(),
                            file_name="insights.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    with col2:
                        csv_data = insights.create_insights_csv(st.session_state.insights_history)
                        st.download_button(
                            label="📊 Download as CSV",
                            data=csv_data,
                            file_name="insights.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                # Input form for new insights
                with st.form(key='insights_form'):
                    insight_query = st.text_input(
                        "🔍 What insights would you like to uncover from your PDF?",
                        key="insight_query"
                    )
                    submit_button = st.form_submit_button("Ask Insights")

                    if submit_button and insight_query:
                        with st.spinner("Retrieving insightful answer..."):
                            answer_insights = insights.answer_question(
                                insight_query,
                                st.session_state.embedder,
                                st.session_state.faiss_index,
                                st.session_state.chunks,
                                GROK_API_KEY,
                                GROK_API_URL,
                                GROK_MODEL
                            )
                            st.session_state.insights_history.append({
                                'question': insight_query,
                                'answer': answer_insights
                            })
                            st.rerun()

        else:
            st.warning("Please upload and process a PDF first.")

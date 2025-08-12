# styles.py
FEATURE_CARDS_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    /* Base styles */
    .stApp {
        font-family: 'Poppins', sans-serif !important;
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }

    /* Base tab styling */
    div[data-baseweb="tab-list"] {
        gap: 8px !important;
        background: transparent !important;
    }

    /* Center-align the entire tab container */
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        justify-content: center !important;
        gap: 8px !important;
        margin-bottom: 1.5rem !important;
        padding: 5px 30px !important;
    }

    /* Make tabs wider and consistent in size */
    .stTabs [data-baseweb="tab"] {
        flex: 1 !important;          /* Allow tabs to grow equally */
        min-width: 120px !important; /* Minimum width */
        max-width: 400px !important; /* Maximum width */
        padding: 20px 20px !important;
        text-align: center !important;
        margin: 0 !important;
    }

    
    /* Individual tabs - applying your color scheme */
    button[data-baseweb="tab"] {
        background-color: #E7E3FF !important;  /* Lavender */
        color: #5E4FA2 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        margin: 0 4px !important;
        border: none !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        font-size: 16px !important;
    }

    button[data-baseweb="tab"]:nth-child(1) {
        background-color: #E7E3FF !important;  /* Lavender */
        color: #1A73E8 !important;
    }
    
    button[data-baseweb="tab"]:nth-child(2) {
        background-color: #D4E6FF !important;  /* Pastel Blue */
        color: #5E4FA2 !important;
    }
    
    button[data-baseweb="tab"]:nth-child(3) {
        background-color: #DFFFD6 !important;  /* Mint Green */
        color: #0B8043 !important;
    }
    
    button[data-baseweb="tab"]:nth-child(4) {
        background-color: #D7F0FF !important;  /* Sky Blue */
        color: #039BE5 !important;
    }
    
    button[data-baseweb="tab"]:nth-child(5) {
        background-color: #FFDCE5 !important;  /* Blush Pink */
        color: #D81B60 !important;
    }
    
    button[data-baseweb="tab"]:nth-child(6) {
        background-color: #FFF2D9 !important;  /* Pastel Yellow */
        color: #F09300 !important;
    }
    
    /* Active tab styling - OVERRIDE STREAMLIT DEFAULT BLUE */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #E7E3FF !important;  /* Lavender */
        color: #5E4FA2 !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transform: translateY(-1px) !important;
        border-bottom: 3px solid #5E4FA2 !important;
        font-size: 17px !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"]:nth-child(2) {
        background-color: #D4E6FF !important;  /* Pastel Blue */
        color: #1A73E8 !important;
        border-bottom-color: #1A73E8 !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"]:nth-child(3) {
        background-color: #DFFFD6 !important;  /* Mint Green */
        color: #0B8043 !important;
        border-bottom-color: #0B8043 !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"]:nth-child(4) {
        background-color: #D7F0FF !important;  /* Sky Blue */
        color: #039BE5 !important;
        border-bottom-color: #039BE5 !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"]:nth-child(5) {
        background-color: #FFDCE5 !important;  /* Blush Pink */
        color: #D81B60 !important;
        border-bottom-color: #D81B60 !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"]:nth-child(6) {
        background-color: #FFF2D9 !important;  /* Pastel Yellow */
        color: #F09300 !important;
        border-bottom-color: #F09300 !important;
    }
    
    /* Hover effects */
    button[data-baseweb="tab"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        font-size: 17px !important;
    }

    /* Force black text for all main elements */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp p,
    .stApp .stMarkdown,
    .stApp .stExpander,
    .stApp .stFileUploader label,
    .stApp .stCaption {
        color: #000000 !important;
    }

    /* Title styling */
    .stApp h1 {
        text-align: center !important;
        margin-bottom: 1.5rem !important;
        font-weight: 700 !important;
    }

    /* Caption styling */
    .stApp .stMarkdown h6 {
        text-align: center !important;
        margin-bottom: 2rem !important;
        font-weight: 400 !important;
        
    }

    /* PDF uploader styling */
    .stFileUploader {
        margin-bottom: 2rem !important;
    }

    /* Info box styling */
    .stExpander {
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        margin-bottom: 1.5rem !important;
    }
    .stExpander .stMarkdown {
        font-size: 14px !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        margin-bottom: 1.5rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 12px 20px !important;
        border-radius: 8px !important;
        background-color: #e9ecef !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #dee2e6 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0d6efd !important;
        color: white !important;
    }

    /* Form elements */
    .stNumberInput, .stTextArea, .stTextInput {
        margin-bottom: 1rem !important;
    }

    /* Button styling */
    .stButton button {
        background-color: #0d6efd !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    .stButton button:hover {
        background-color: #0b5ed7 !important;
        transform: translateY(-1px);
    }

    /* Spinner styling */
    .stSpinner > div {
        border-color: #0d6efd transparent transparent transparent !important;
    }

    /* Alert boxes */
    .stAlert {
        border-radius: 8px !important;
        padding: 16px !important;
    }

    /* Divider styling */
    .stDivider {
        margin: 1.5rem 0 !important;
    }

    /* Radio button styling */
    .stRadio [role="radiogroup"] {
        gap: 12px !important;
        margin-bottom: 1rem !important;
    }

    .stRadio [role="radio"] {
        padding: 8px 16px !important;
        border-radius: 8px !important;
        border: 1px solid #dee2e6 !important;
    }

    /* Download buttons */
    .stDownloadButton button {
        width: 100% !important;
        margin-top: 8px !important;
    }

    /* MCQ options styling */
    [data-testid="stMarkdownContainer"] p {
        margin-bottom: 4px !important;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: wrap !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            flex: 1 0 45% !important;
            margin-bottom: 8px !important;
        }
    }

    /* Remove default Streamlit button styling */
    button[kind="secondary"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    button[kind="secondary"]:hover {
        background: transparent !important;
    }
</style>
"""
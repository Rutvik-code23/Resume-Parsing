import streamlit as st
import pickle
import re
import PyPDF2
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Resume Parser & Classifier",
    page_icon="📄",
    layout="wide"
)

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    """Load the trained model, vectorizer, and encoder"""
    try:
        with open('resume_classifier_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('resume_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('category_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, vectorizer, encoder
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()

model, vectorizer, encoder = load_model()

# Clean resume function (same as training)
def cleanResume(resumeText):
    """Clean and preprocess resume text"""
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)  # Remove URLs
    resumeText = re.sub(r'RT|CC', ' ', resumeText)  # Remove RT and CC
    resumeText = re.sub(r'#\S+', ' ', resumeText)  # Remove hashtags
    resumeText = re.sub(r'@\S+', ' ', resumeText)  # Remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', resumeText)  # Remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)  # Remove non-ASCII characters
    resumeText = re.sub(r'\s+', ' ', resumeText)  # Remove extra white spaces
    return resumeText

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def predict_category(resume_text):
    """Predict the category of the resume"""
    # Clean the resume text
    cleaned_text = cleanResume(resume_text)
    
    # Vectorize the text
    text_features = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction_encoded = model.predict(text_features)[0]
    prediction = encoder.inverse_transform([prediction_encoded])[0]
    
    # Get prediction probabilities
    probabilities = model.predict_proba(text_features)[0]
    
    # Get top 3 categories
    top_indices = probabilities.argsort()[-3:][::-1]
    top_categories = encoder.inverse_transform(top_indices)
    top_probabilities = probabilities[top_indices]
    
    return prediction, top_categories, top_probabilities

# Main app
st.title("📄 Resume Parser & Category Classifier")
st.markdown("Upload a resume PDF to predict its category automatically!")

# Sidebar for model info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application classifies resumes into **25 different categories**:
    
    - Data Science, HR, Advocate, Arts
    - Web Designing, Mechanical Engineer
    - Sales, Health and fitness
    - Civil Engineer, Java Developer
    - Business Analyst, SAP Developer
    - Automation Testing, Testing
    - Electrical Engineering
    - Operations Manager
    - Python Developer, DevOps Engineer
    - Network Security Engineer
    - PMO, Database, Hadoop
    - ETL Developer, DotNet Developer
    - Blockchain
    """)
    
    st.info("💡 **Tip:** The model uses TF-IDF features and K-Nearest Neighbors classifier with 99% accuracy!")

# File upload
uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=['pdf'],
    help="Select a PDF file containing the resume"
)

if uploaded_file is not None:
    # Extract text from PDF
    with st.spinner("Extracting text from PDF..."):
        resume_text = extract_text_from_pdf(uploaded_file)
    
    if resume_text:
        # Display extracted text
        with st.expander("📝 View Extracted Resume Text", expanded=False):
            st.text(resume_text[:2000] + "..." if len(resume_text) > 2000 else resume_text)
        
        # Clean and display cleaned text
        cleaned_text = cleanResume(resume_text)
        with st.expander("🧹 View Cleaned Text", expanded=False):
            st.text(cleaned_text[:2000] + "..." if len(cleaned_text) > 2000 else cleaned_text)
        
        # Predict category
        if st.button("🔍 Predict Category", type="primary", use_container_width=True):
            with st.spinner("Analyzing resume..."):
                prediction, top_categories, top_probabilities = predict_category(resume_text)
            
            # Main prediction result
            st.success(f"### 🎯 Predicted Category: **{prediction}**")
            
            # Top 3 categories with confidence scores
            st.subheader("📊 Top 3 Category Predictions")
            
            # Create columns for better display
            cols = st.columns(3)
            for idx, (category, prob) in enumerate(zip(top_categories, top_probabilities)):
                with cols[idx]:
                    st.metric(
                        label=category,
                        value=f"{prob*100:.2f}%"
                    )
                    # Progress bar
                    st.progress(float(prob))
            
            # Visualization
            st.subheader("📈 Probability Distribution")
            import pandas as pd
            prob_df = pd.DataFrame({
                'Category': top_categories,
                'Probability': top_probabilities
            })
            st.bar_chart(prob_df.set_index('Category'))
            
            # Resume statistics
            with st.expander("📊 Resume Statistics"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Characters", f"{len(resume_text):,}")
                with col2:
                    st.metric("Total Words", f"{len(resume_text.split()):,}")
                with col3:
                    st.metric("Cleaned Words", f"{len(cleaned_text.split()):,}")
else:
    st.info("👆 Please upload a PDF file to get started")
    st.markdown("---")
    
    # Example section
    st.subheader("🔍 What This App Does")
    st.markdown("""
    1. **Extracts text** from your resume PDF
    2. **Cleans and preprocesses** the text (removes URLs, special characters, etc.)
    3. **Analyzes** the content using TF-IDF features
    4. **Classifies** the resume into one of 25 professional categories
    5. **Shows confidence scores** for the top predictions
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Powered by Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)


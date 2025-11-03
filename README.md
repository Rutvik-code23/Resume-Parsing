# Resume Parser & Classifier

A machine learning-based web application that automatically parses and classifies resumes into 25 different professional categories using Streamlit.

## 🎯 Features

- **PDF Resume Upload**: Upload any resume in PDF format
- **Text Extraction**: Automatically extracts text from uploaded resumes
- **Intelligent Classification**: Classifies resumes into 25 professional categories
- **Confidence Scores**: Shows prediction probabilities for top categories
- **Beautiful UI**: Modern and user-friendly Streamlit interface
- **High Accuracy**: 99% accuracy on test data using K-Nearest Neighbors classifier

## 📋 Categories Supported

The model can classify resumes into the following 25 categories:

1. Data Science
2. HR
3. Advocate
4. Arts
5. Web Designing
6. Mechanical Engineer
7. Sales
8. Health and fitness
9. Civil Engineer
10. Java Developer
11. Business Analyst
12. SAP Developer
13. Automation Testing
14. Electrical Engineering
15. Operations Manager
16. Python Developer
17. DevOps Engineer
18. Network Security Engineer
19. PMO
20. Database
21. Hadoop
22. ETL Developer
23. DotNet Developer
24. Blockchain
25. Testing

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Resume-Parsing
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if you want to retrain or model files are missing)
   ```bash
   python train_model.py
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   
   The app will automatically open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
Resume-Parsing/
├── app.py                        # Streamlit application
├── train_model.py                # Model training script
├── requirements.txt              # Python dependencies
├── UpdatedResumeDataSet.csv      # Training dataset
├── Resume Parsing.ipynb          # Jupyter notebook (original analysis)
├── resume_classifier_model.pkl   # Trained model (generated)
├── resume_vectorizer.pkl         # TF-IDF vectorizer (generated)
├── category_encoder.pkl          # Label encoder (generated)
└── README.md                     # This file
```

## 🔍 How It Works

1. **Text Extraction**: The app extracts text from uploaded PDF resumes using PyPDF2
2. **Text Cleaning**: The extracted text is cleaned to remove URLs, special characters, hashtags, and mentions
3. **Feature Extraction**: TF-IDF vectorization is applied to convert text into numerical features
4. **Classification**: The trained KNN classifier predicts the most appropriate category
5. **Results Display**: Shows the predicted category along with confidence scores

## 🧠 Model Details

- **Algorithm**: K-Nearest Neighbors (KNN) with OneVsRest strategy
- **Feature Extraction**: TF-IDF with 1500 max features
- **Accuracy**: 99% on test set
- **Training Data**: 769 samples
- **Test Data**: 193 samples

## 📊 Model Performance

- **Training Accuracy**: 99%
- **Test Accuracy**: 99%
- **Average Precision**: 0.99
- **Average Recall**: 0.99
- **Average F1-Score**: 0.99

## 💻 Usage

1. Open the Streamlit app in your browser
2. Click on "Upload Resume PDF" and select a PDF file
3. Click "Predict Category" button
4. View the predicted category and confidence scores
5. Explore extracted and cleaned text if needed

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Library**: scikit-learn
- **Text Processing**: NLTK
- **PDF Processing**: PyPDF2
- **Data Processing**: pandas, numpy

## 📝 Notes

- The app expects clean PDF files for best results
- Text extraction quality depends on PDF structure
- Model performance may vary with resumes that don't match training data patterns

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

Resume Parsing Project - NLP Semester 5

## 🙏 Acknowledgments

- Dataset: UpdatedResumeDataSet.csv
- ML Framework: scikit-learn
- UI Framework: Streamlit
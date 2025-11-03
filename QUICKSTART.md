# Quick Start Guide

## 🚀 Running the App in 3 Simple Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Streamlit App
```bash
streamlit run app.py
```

### Step 3: Upload and Predict
1. The app will open in your browser automatically
2. Click "Upload Resume PDF" button
3. Select your resume PDF file
4. Click "Predict Category" button
5. View the results!

## 📋 What You Need

- ✅ Python 3.8 or higher
- ✅ All `.pkl` model files (already generated)
  - `resume_classifier_model.pkl`
  - `resume_vectorizer.pkl`
  - `category_encoder.pkl`
- ✅ A PDF resume to test with

## 🎯 Example Usage

The app will:
1. Extract text from your PDF resume
2. Clean and preprocess the text
3. Classify it into one of 25 categories
4. Show confidence scores for top predictions

## ⚠️ Troubleshooting

**If model files are missing:**
```bash
python train_model.py
```

**If Streamlit doesn't open automatically:**
Open your browser and go to: `http://localhost:8501`

**If you get import errors:**
```bash
pip install --upgrade -r requirements.txt
```

## 📝 Testing with Sample PDF

Try the included sample: `Rutvik Prajapati - Resume.pdf`

Happy parsing! 🎉


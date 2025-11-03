import numpy as np
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Load the dataset
print("Loading dataset...")
resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv')

# Clean resume function
def cleanResume(resumeText):
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)  # To remove the urls
    resumeText = re.sub(r'RT|CC', ' ', resumeText)  # Remove RT and CC
    resumeText = re.sub(r'#\S+', ' ', resumeText)  # Remove hashtags
    resumeText = re.sub(r'@\S+', ' ', resumeText)  # Removing mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub(r'\s+', ' ', resumeText)  # remove extra white space
    return resumeText

# Clean the resume text
print("Cleaning resume text...")
resumeDataSet['Cleaned_Resume'] = resumeDataSet['Resume'].apply(lambda x: cleanResume(x))

# Encode categories
print("Encoding categories...")
le = LabelEncoder()
resumeDataSet['Category_Encoded'] = le.fit_transform(resumeDataSet['Category'])

# Prepare features and target
requiredText = resumeDataSet['Cleaned_Resume'].values
requiredTarget = resumeDataSet['Category_Encoded'].values

# Create TF-IDF vectorizer
print("Creating TF-IDF features...")
wordvectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=1500)
wordvectorizer.fit(requiredText)
WordFeatures = wordvectorizer.transform(requiredText)

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Train the model
print("Training model...")
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

# Evaluate the model
prediction = clf.predict(X_test)
print('\nAccuracy of the KNeighborsClassifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of the KNeighborsClassifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
print('\nClassification report:\n{}'.format(metrics.classification_report(y_test, prediction)))

# Save the model and vectorizer
print("\nSaving model and vectorizer...")
with open('resume_classifier_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('resume_vectorizer.pkl', 'wb') as f:
    pickle.dump(wordvectorizer, f)

with open('category_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model saved successfully!")
print(f"\nCategories: {list(le.classes_)}")


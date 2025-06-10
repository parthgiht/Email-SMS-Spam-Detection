# 📧 Email/SMS Spam Detection
This project implements a machine learning-based Email/SMS Spam Classifier to detect whether a given message is spam or not. The classifier uses Natural Language Processing (NLP) techniques to preprocess text data and a Multinomial Naive Bayes model for classification. The project includes a Jupyter Notebook for model training and a Streamlit web application for user interaction.

### 📋 Table of Contents

- Overview
- Features
- Dataset
- Project Structure
- Installation
- Usage
- Model Details
- Web Application
- Results
- Contributing

### 🎯 Overview
The Email/SMS Spam Classifier processes text messages to classify them as Spam or Not Spam. The pipeline includes data cleaning, exploratory data analysis (EDA), text preprocessing, model training, and evaluation. The final model is deployed as an interactive web application using Streamlit, allowing users to input messages and receive predictions.

## Features

- 🔡 **Text Preprocessing**:- Converts text to lowercase, tokenizes it, removes special characters, stop words, and punctuation, and applies stemming.
  
- 🤖 **Model Training**:- Evaluates multiple Naive Bayes models (**Gaussian**, **Multinomial**, **Bernoulli**) with **CountVectorizer** and **TfidfVectorizer**.
  
- 🔍 **Model Selection**:- Uses TfidfVectorizer with Multinomial Naive Bayes for optimal performance.
  
- 🌐 **Web Interface**:- A Streamlit app for real-time spam detection.
  
- 💾 **Model Persistence**:- Saves the trained model and vectorizer using pickle for deployment.

## 📄 Dataset

The dataset (`SMS spam data.csv`) contains 5,572 SMS messages labeled as ham (`not spam`) or `spam`. It includes:

- v1:- Label (`ham` or `spam`).
  
- v2:- Text of the message.
  
- Additional unnamed columns (dropped during preprocessing).

The dataset is cleaned by removing unnecessary columns and preprocessing the text for model training.

## 📁 Project Structure
```
├── 💻 Email SMS spam detection.ipynb  # Google colab for data analysis and model training
├── 🌐 Streamlit_Main.py               # Streamlit app for spam classification
├── 🔢 vectorizer.pkl                  # Saved TfidfVectorizer sklearn library
├── 🤖 model.pkl                       # Saved Multinomial Naive Bayes model
├── 📄 SMS spam data.csv               # Dataset 
└── 📝 README.md                       # Project documentation
```


## Installation

1. Clone the repository:-
```
git clone https://github.com/your-username/email-sms-spam-classifier.git
cd email-sms-spam-classifier
```

2. Install the required dependencies:-
```
streamlit
numpy
pandas
nltk
scikit-learn
```


3. Download the NLTK data:-
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. Place the dataset (SMS spam data.csv) in the project directory or update the file path in the notebook.


## Usage
🤖 **Training the Model**

1. Open Email SMS spam detection.ipynb in Jupyter Notebook or Google Colab.
2. Run the notebook cells to:
      - Load and clean the dataset.
      - Preprocess the text data.
      - Train and evaluate Naive Bayes models.
      - Save the TfidfVectorizer (`vectorizer.pkl`) and Multinomial Naive Bayes model (`model.pkl`).


## 📝 Model Details

- Text Preprocessing:
    - Converts text to lowercase.
    - Tokenizes using NLTK's word_tokenize.
    - Removes special characters, stop words, and punctuation.
    - Applies Porter Stemming.


- Feature Extraction:
    - Uses TfidfVectorizer to convert text into numerical features.


- Model Evaluation:
    - Compared `Gaussian`, `Multinomial`, and `Bernoulli` Naive Bayes models.
    - `TfidfVectorizer` with `MultinomialNB` was selected due to its high accuracy (95.94%) and perfect precision (1.0) on the test set.
    - Results with TfidfVectorizer:
    - **GaussianNB**:-  Accuracy: **87.62%**, Precision: **0.523**
    - **MultinomialNB**:-  Accuracy: **95.94%**, Precision: **1.0**
    - **BernoulliNB**:-  Accuracy: **97.00%**, Precision: **0.973**


## Results
The Multinomial Naive Bayes model with TfidfVectorizer achieved:

- 📊 Accuracy: **95.94%**
- 🛡️ Precision: **100% (no false positives for spam).**
- 📈 Confusion Matrix:
```
  [[896   0]
  [ 42  96]]
```

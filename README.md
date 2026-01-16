# Email Spam Detection with Machine Learning

### Project Overview

This project is a machine learning-based security tool designed to classify email messages as **Spam** (unwanted/dangerous) or **Ham** (legitimate). Using Natural Language Processing (NLP) techniques, the system identifies patterns common in phishing and scam emails to protect users from potential threats.

### Key Features

* **Automated Data Cleaning:** Handles encoding issues and removes redundant data columns.
* **Text Vectorization:** Implements **TF-IDF (Term Frequency-Inverse Document Frequency)** to transform raw text into meaningful numerical features.
* **Probabilistic Classification:** Utilizes the **Multinomial Naive Bayes** algorithm, optimized for high-dimensional text datasets.
* **Interactive Interface:** Includes a real-time prediction loop allowing users to input custom messages for instant classification.

### Tech Stack

* **Language:** Python 3.x
* **Data Handling:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

### Machine Learning Pipeline

1. **Exploratory Data Analysis (EDA):** Visualizing the distribution of Spam vs. Ham in the dataset.
2. **Preprocessing:** Label encoding (Ham=0, Spam=1) and text tokenization.
3. **Vectorization:** Calculating TF-IDF weights to emphasize rare, high-information words.
4. **Training:** Fitting the Naive Bayes model on the training subset.
5. **Evaluation:** Measuring performance using a Confusion Matrix and Accuracy Score.

### Results

* **Accuracy:** Achieved over **96%** accuracy on test data.
* **Robustness:** Highly effective at filtering common spam keywords like "winner," "claim," and "free."

### How to Use

1. **Clone the repository:**
```bash
git clone https://github.com/rohankapase/email-spam-detection-using-ML.git


2. **Install dependencies:**
```bash
pip install pandas scikit-learn matplotlib seaborn

```


3. **Run the application:**
```bash
python spam_detection.py
```



### Dataset

The project uses the `spam.csv` dataset, which contains 5,572 records of SMS/Email messages labeled as 'ham' or 'spam'.

---


### **Conclusion**

**Email Spam Detection with Machine Learning**
Developed a text classification system to filter junk mail using Python. The project implements **TF-IDF Vectorization** and the **Multinomial Naive Bayes** algorithm to analyze word patterns. It features a real-time prediction interface that accurately distinguishes between legitimate communication and harmful spam or phishing attempts.

### **Key Highlights:**

* **Technology:** Python, Scikit-learn, Pandas.
* **Core Logic:** Uses probabilistic modeling to identify spam keywords and frequencies.
* **Accuracy:** Achieved high precision (96%+) in identifying "Ham" (safe) vs "Spam" messages.

---

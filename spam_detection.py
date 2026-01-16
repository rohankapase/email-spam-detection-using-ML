import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. LOAD DATASET
# 'latin-1' encoding is used because text data often contains special symbols
df = pd.read_csv('spam.csv', encoding='latin-1')

# 2. DATA CLEANING
# Remove unnecessary columns that are empty (Unnamed 2, 3, 4)
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# Rename columns to meaningful names
df.columns = ['Label', 'Message']

# Convert text labels to numbers: 'ham' becomes 0 and 'spam' becomes 1
df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})

# 3. SPLIT DATA
# Split data into Training set (80%) and Testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Label'], test_size=0.2, random_state=42)

# 4. TEXT TO NUMBERS (Vectorization)
# TF-IDF converts text into numerical features based on word importance
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# 5. MODEL TRAINING
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)



# 6. EVALUATION
y_pred = model.predict(X_test_tfidf)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 7. VISUALIZING RESULTS
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Spam Detection Confusion Matrix')
plt.show()



# 8. LIVE USER INPUT PREDICTION
print("\n--- Email Spam Detector System ---")
while True:
    user_msg = input("\nEnter your message (or type 'exit' to quit): ")
    
    if user_msg.lower() == 'exit':
        print("Exiting...")
        break
    
    # Transform the user input to numerical format
    data = tfidf.transform([user_msg])
    
    # Make prediction
    prediction = model.predict(data)
    
    # Display Result
    if prediction[0] == 1:
        print("RESULT: This is a SPAM message! 🚩")
    else:
        print("RESULT: This is a HAM (Safe) message. ✅")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report


# Step 1: Load data into a pandas DataFrame
data = pd.read_csv("/data1/aakash/Codemix/Aakash_02/1Humor_Codemix.csv")
data=data.dropna()


# Step 2: Preprocess text data and extract n-gram features
X = data['Sentence']
y = data['Tag']

# Convert the sentences to n-gram features
ngram_vectorizer = CountVectorizer(ngram_range=(1, 3))
X_ngrams = ngram_vectorizer.fit_transform(X)

# Step 3: Split the data into train, test, and validation sets
X_train, X_remaining, y_train, y_remaining = train_test_split(X_ngrams, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42, stratify=y_remaining)


# Step 4: Train the SVM classifier on the training set
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)


# Step 5: Evaluate the classifier on the test and validation sets
y_pred_test = svm_classifier.predict(X_test)
y_pred_val = svm_classifier.predict(X_val)

accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_val = accuracy_score(y_val, y_pred_val)

print("Test Set Accuracy:", accuracy_test)
print("Validation Set Accuracy:", accuracy_val)


# Step 6: Print the classification report
print("\nClassification Report for Test Set:")
print(classification_report(y_test, y_pred_test))


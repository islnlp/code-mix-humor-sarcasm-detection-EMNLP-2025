import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Step 1: Load the original dataset and additional dataset into pandas DataFrames
original_data = pd.read_csv("/data1/aakash/Codemix/Aakash_02/1Humor_Codemix.csv")
original_data=original_data.dropna()
eng = pd.read_csv("/data1/aakash/Codemix/Aakash_02/1Humor_English(new).csv")


english_data = eng[:3000]



# Step 2: Split the original dataset into train, test, and validation sets
X_original = original_data['Sentence']
y_original = original_data['Tag']

X_train_original, X_remaining_original, y_train_original, y_remaining_original = train_test_split(X_original, y_original, test_size=0.3, random_state=42, stratify=y_original)
X_test_original, X_val_original, y_test_original, y_val_original = train_test_split(X_remaining_original, y_remaining_original, test_size=0.5, random_state=42, stratify=y_remaining_original)


# Step 3: Combine the train set from the original dataset with the additional dataset
combined_train = pd.concat([X_train_original, english_data['Sentence']])
combined_train_labels = pd.concat([y_train_original, english_data['Tag']])


# Step 4: Preprocess the combined train set and additional test and validation sets (extract n-gram features)
ngram_vectorizer = CountVectorizer(ngram_range=(1, 3))
X_train_combined = ngram_vectorizer.fit_transform(combined_train)
X_test_original_ngrams = ngram_vectorizer.transform(X_test_original)
X_val_original_ngrams = ngram_vectorizer.transform(X_val_original)


# Step 5: Train the classifier on the combined train set
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_combined, combined_train_labels)


# Step 6: Evaluate the classifier on the test and validation sets from the original dataset
y_pred_test_original = nb_classifier.predict(X_test_original_ngrams)
y_pred_val_original = nb_classifier.predict(X_val_original_ngrams)

accuracy_test_original = accuracy_score(y_test_original, y_pred_test_original)
accuracy_val_original = accuracy_score(y_val_original, y_pred_val_original)

print("Test Set Accuracy on Original Data:", accuracy_test_original)
print("Validation Set Accuracy on Original Data:", accuracy_val_original)


# Step 7: Print the classification report for the test set on the combined dataset
print("\nClassification Report for Test Set on Combined Dataset:")
print(classification_report(y_test_original, y_pred_test_original))


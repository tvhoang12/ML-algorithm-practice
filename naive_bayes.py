import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#read data from CSV file
data = pd.read_csv('spam_ham_dataset.csv')

#delete duplicate rows
data = data.drop_duplicates()

# change data to DataFrame
df = pd.DataFrame(data)

# change to vectorize beacause Naive Bayes needs numerical input
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

#initialize and train the model
model = MultinomialNB() # model Naive Bayes
model.fit(X, y)

#test model accuracy
new_messages = [
    "i am staying at company",
    "collect your free prize now"
]

#transform new messages to vectorized form
X_new = vectorizer.transform(new_messages)

#predict
predictions = model.predict(X_new)

print(f"Tin nhắn 1: '{new_messages[0]}' -> Dự đoán: {predictions[0]}")
print(f"Tin nhắn 2: '{new_messages[1]}' -> Dự đoán: {predictions[1]}")

# get prediction probabilities
probabilities = model.predict_proba(X_new)
print(f"\nXác suất (ham, spam) cho tin 1: {probabilities[0]}")
print(f"Xác suất (ham, spam) cho tin 2: {probabilities[1]}")
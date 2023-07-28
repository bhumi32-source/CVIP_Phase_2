import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1') 

# Feature extraction - TF-IDF representation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['v2'])

# Target labels (1 for spam, 0 for ham)
y = data['v1'].map({'spam': 1, 'ham': 0})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

labels = ['Accuracy']
values = [accuracy]

plt.bar(labels, values)
plt.ylim(0, 1.0)  # Set the y-axis limit from 0 to 1.0 (as accuracy ranges from 0 to 1)
plt.ylabel('Accuracy')
plt.title('Email Spam Detection Model Accuracy')
plt.show()

# Class Distribution Pie Chart
class_counts = data['v1'].value_counts()
labels = class_counts.index
sizes = class_counts.values
colors = ['#ff9999', '#66b3ff']  # Custom colors for the pie chart
explode = (0.1, 0)  # Explode the first slice (spam) for emphasis

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, colors=colors, explode=explode, startangle=140)
plt.title('Class Distribution', fontsize=14, fontweight='bold')
plt.axis('equal')  # Equal aspect ratio ensures a circular pie chart

# Bar Chart for Precision and Recall
report = classification_report(y_test, y_pred, output_dict=True)
metrics_df = pd.DataFrame(report).transpose()
ax = metrics_df.plot(kind='bar', y=['precision', 'recall'], ylim=(0, 1.0), color=['#66b3ff', '#ff9999'])
plt.title('Precision and Recall Scores', fontsize=14, fontweight='bold')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(rotation=0, fontsize=11)  # Rotate x-axis labels and set font size

# Add text labels above the bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                xytext=(0, 10), textcoords='offset points', fontsize=10)

plt.tight_layout()  # Adjust plot to prevent overlapping
plt.show()
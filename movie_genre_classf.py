import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load the dataset
# Replace these paths with the paths to your 4 datasets
train_path1 = r'C:\Users\PRINTER WORLD\Desktop\Genre Classification Dataset\description.csv'
train_path2 = r'C:\Users\PRINTER WORLD\Desktop\Genre Classification Dataset\test_data.csv'
train_path3 = r'C:\Users\PRINTER WORLD\Desktop\Genre Classification Dataset\test_data_solution.csv'
train_path4 = r'C:\Users\PRINTER WORLD\Desktop\Genre Classification Dataset\train_data.csv'

# Function to load and combine the datasets
def load_datasets():
    # Adjust as needed if formats differ across datasets
    df1 = pd.read_csv(train_path1, delimiter=":::", engine="python", names=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    df2 = pd.read_csv(train_path2, delimiter=":::", engine="python", names=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    df3 = pd.read_csv(train_path3, delimiter=":::", engine="python", names=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    df4 = pd.read_csv(train_path4, delimiter=":::", engine="python", names=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    
    # Concatenate all datasets
    train_data = pd.concat([df1, df2, df3, df4], ignore_index=True)
    return train_data

# Prepare the data
train_data = load_datasets()

# Check for missing values in DESCRIPTION
print("Missing values in DESCRIPTION before handling:", train_data['DESCRIPTION'].isnull().sum())

# Drop rows with missing DESCRIPTION
train_data = train_data.dropna(subset=['DESCRIPTION'])

# Alternatively, you could fill missing values with an empty string:
# train_data['DESCRIPTION'] = train_data['DESCRIPTION'].fillna('')

# After handling missing values
print("Missing values in DESCRIPTION after handling:", train_data['DESCRIPTION'].isnull().sum())

X = train_data['DESCRIPTION']  # Features
y = train_data['GENRE']        # Labels

# Text vectorization with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Choose and train a classifier (e.g., Logistic Regression)
model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# Use model.predict(new_data) for test predictions where new_data is test data transformed with vectorizer
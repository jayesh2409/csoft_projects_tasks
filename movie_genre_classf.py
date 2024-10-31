import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# paths to the dataset files
train_path1 = r'C:\Users\PRINTER WORLD\Desktop\Genre Classification Dataset\description.csv'
train_path2 = r'C:\Users\PRINTER WORLD\Desktop\Genre Classification Dataset\test_data.csv'
train_path3 = r'C:\Users\PRINTER WORLD\Desktop\Genre Classification Dataset\test_data_solution.csv'
train_path4 = r'C:\Users\PRINTER WORLD\Desktop\Genre Classification Dataset\train_data.csv'
 
def load_datasets():
    df1 = pd.read_csv(train_path1, delimiter=":::", engine="python", names=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    df2 = pd.read_csv(train_path2, delimiter=":::", engine="python", names=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    df3 = pd.read_csv(train_path3, delimiter=":::", engine="python", names=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    df4 = pd.read_csv(train_path4, delimiter=":::", engine="python", names=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    
    #combining all dataframes 
    combined_data = pd.concat([df1, df2, df3, df4], ignore_index=True)
    return combined_data

# Load the data
train_data = load_datasets()

print("Missing values in DESCRIPTION before handling:", train_data['DESCRIPTION'].isnull().sum())

train_data = train_data.dropna(subset=['DESCRIPTION'])

print("All good! Missing values in DESCRIPTION after handling:", train_data['DESCRIPTION'].isnull().sum())

# features (X) and Labels (y)
X = train_data['DESCRIPTION']  
y = train_data['GENRE']        

vectorizer = TfidfVectorizer(max_features=5000)  # keep manageable mx 5000 features
X_tfidf = vectorizer.fit_transform(X)

# split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = LogisticRegression()

# train model
model.fit(X_train, y_train)

#making predictions and evaluate
y_pred = model.predict(X_val)
print("We got an accuracy of:", accuracy_score(y_val, y_pred))
print("How we did on each genre:\n", classification_report(y_val, y_pred))

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse


# Preprocessing text removing punctuations and making all lower case


def preprocess(data, col, lang):
    data[col] = data[col].str.replace(r'<[^<>]*>', '', regex=True)

    data[col] = data[col].str.lower()
    if lang == 'en':
        data[col] = data[col].str.replace(r"n\'t", " not", regex=True)
        data[col] = data[col].str.replace(r"\'t", " not", regex=True)

    data[col] = data[col].str.replace(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', regex=True)
    data[col] = data[col].str.replace(r'[^\w\s\?]', ' ', regex=True)
    data[col] = data[col].str.replace(r'([\;\:\|•«\n])', ' ', regex=True)

    return data


# Creating vectorizer, Removing stopwords, creating embeddings for the headlines and returning sparse arrays
def tv(train, test):
    stopwords_list = stopwords.words("english")
    tfidf = TfidfVectorizer(stop_words=stopwords_list, ngram_range=(1, 2))
    tfidf_text_train = tfidf.fit_transform(train['headlines'])
    tfidf_text_test = tfidf.transform(test['headlines'])

    X_train_ef = train.drop(columns='headlines')
    X_test_ef = test.drop(columns='headlines')

    train = sparse.hstack([X_train_ef, tfidf_text_train]).tocsr()
    X_test = sparse.hstack([X_test_ef, tfidf_text_test]).tocsr()

    return train, X_test


def creating_model() -> object:
    # opening clickbait file and putting it as array
    with open('clickbait_yes', encoding="utf8") as f:
        lines = [line.rstrip() for line in f]

    # Generating a dataframe from the list
    df = pd.DataFrame(lines, columns=['headlines'])
    # Adding a class as 1 to denote its a clickbait
    df['class'] = 1

    with open('clickbait_no', encoding="utf8") as f:
        lines = [line.rstrip() for line in f]

        # Generating a dataframe from the list
    df1 = pd.DataFrame(lines, columns=['headlines'])
    # Adding a class as 0 to denote its a non-clickbait
    df1['class'] = 0

    # Generating a dataframe with labeled headlines for clickbait and non click bait
    df = df.append(df1)

    # for the test data
    X_test_data = pd.read_csv('clickbait_hold_X.csv', names=['headlines'])

    X_test = preprocess(X_test_data, "headlines", "en")

    # Cleaning and preprocessing the dataframe headlines columns
    df = preprocess(df, "headlines", "en")

    # Creating test and validation set

    X = df.drop(columns=['class']).copy()
    Y = df['class']
    X_train, X_test = tv(X, X_test)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y, test_size=0.8, random_state=150)

    # Implementing a TFIDF Vectorizer
    # X_train, X_test = tv(X_train, X_test)
    # X_train, X_valid = tv(X_train, X_valid)

    # Creating a Naive Bayes model
    nb_classifier = MultinomialNB(alpha=.05)
    # Fitting a model
    model = nb_classifier.fit(X_train, Y_train)

    # Prediction on the generated model

    # for test data
    nb_test_predict_test = nb_classifier.predict(X_test)

    nb_test_predict_valid = nb_classifier.predict(X_valid)

    print(f'The F1 score on a validation_set  is {f1_score(Y_valid, nb_test_predict_valid)}')

    return nb_test_predict_test


if __name__ == '__main__':
    prediction = creating_model()
    print(f'The prediction in the test data is {prediction} . This variable prediction can be used '
          f'to check the score')

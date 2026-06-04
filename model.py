import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def train_model(df):

    df = df.copy()

    numeric_columns = [
        "Danceability",
        "Energy",
        "Loudness",
        "Speechiness",
        "Acousticness",
        "Liveness",
        "Tempo",
        "Duration (ms)",
        "Valence"
    ]

    df = df[numeric_columns + ["Popularity"]]

    df = df.dropna()

    threshold = df["Popularity"].median()

    df["Target"] = (
        df["Popularity"] >= threshold
    ).astype(int)

    X = df[numeric_columns]

    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    ) * 100

    report = classification_report(
        y_test,
        predictions
    )

    return accuracy, report

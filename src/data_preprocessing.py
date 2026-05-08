import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):

    df = pd.read_csv(path)

    return df


def preprocess_data(df):

    # Remove missing values
    df = df.dropna()

    # Features and target
    X = df.drop("median_house_value", axis=1)

    y = df["median_house_value"]

    # Convert categorical data into numerical
    X = pd.get_dummies(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test
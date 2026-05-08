from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

def train_model(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    print(f"\nMean Absolute Error: {mae}")

    print(f"R2 Score: {r2}")

    # Actual vs Predicted Graph
    plt.figure(figsize=(8,5))

    plt.scatter(y_test, predictions)

    plt.xlabel("Actual Prices")

    plt.ylabel("Predicted Prices")

    plt.title("Actual vs Predicted House Prices")

    plt.show()


def save_model(model, path):

    joblib.dump(model, path)

    print("\nModel saved successfully.")
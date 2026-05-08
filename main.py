from src.data_preprocessing import load_data, preprocess_data

from src.train_model import (
    train_model,
    evaluate_model,
    save_model
)

from src.visualization import (
    plot_price_distribution,
    plot_heatmap,
    plot_income_vs_price
)

DATA_PATH = "data/housing.csv"

MODEL_PATH = "models/house_price_model.pkl"


def main():

    print("Loading dataset...")

    df = load_data(DATA_PATH)

    print(df.head())

    # Graphs
    print("\nGenerating graphs...")

    plot_price_distribution(df)

    plot_heatmap(df)

    plot_income_vs_price(df)

    # Preprocessing
    print("\nPreprocessing data...")

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Model Training
    print("\nTraining model...")

    model = train_model(X_train, y_train)

    # Evaluation
    print("\nEvaluating model...")

    evaluate_model(model, X_test, y_test)

    # Save Model
    save_model(model, MODEL_PATH)

    print("\nProject completed successfully.")


if __name__ == "__main__":

    main()
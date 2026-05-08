import matplotlib.pyplot as plt
import seaborn as sns

def plot_price_distribution(df):

    plt.figure(figsize=(8,5))

    sns.histplot(df["median_house_value"], bins=30)

    plt.title("House Price Distribution")

    plt.xlabel("House Price")

    plt.ylabel("Count")

    plt.show()


def plot_heatmap(df):

    plt.figure(figsize=(12,8))

    correlation = df.corr(numeric_only=True)

    sns.heatmap(correlation, annot=False)

    plt.title("Correlation Heatmap")

    plt.show()


def plot_income_vs_price(df):

    plt.figure(figsize=(8,5))

    sns.scatterplot(
        x=df["median_income"],
        y=df["median_house_value"]
    )

    plt.title("Median Income vs House Price")

    plt.xlabel("Median Income")

    plt.ylabel("House Price")

    plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distributions(df):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], ax=axes[i], kde=True, color='steelblue')
        axes[i].set_title(col)
    plt.suptitle("Distribution des variables", fontsize=16)
    plt.tight_layout()
    plt.savefig("distributions.png")
    plt.show()
    print("✅ distributions.png sauvegardé")

def plot_correlation(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Heatmap de corrélation")
    plt.tight_layout()
    plt.savefig("correlation.png")
    plt.show()
    print("✅ correlation.png sauvegardé")

def plot_boxplots(df):
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.boxplot(x='Outcome', y=col, data=df, ax=axes[i], hue='Outcome', palette='Set2', legend=False)
        axes[i].set_title(col)
    plt.suptitle("Boxplots par rapport au diabète", fontsize=16)
    plt.tight_layout()
    plt.savefig("boxplots.png")
    plt.show()
    print("✅ boxplots.png sauvegardé")
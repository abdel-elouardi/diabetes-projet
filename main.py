from src.cleaning import load_data, clean_data, check_duplicates, remove_outliers, get_info
from src.visualization import plot_distributions, plot_correlation, plot_boxplots
from src.model import prepare_data, train_model, evaluate_model, save_model

def main():
    print("🚀 Démarrage du pipeline diabetes\n")

    # Étape 1 - Chargement des données
    print("📂 Chargement des données...")
    df = load_data('data/diabetes.csv')

    # Étape 2 - Nettoyage
    print("\n🧹 Nettoyage des données...")
    df = clean_data(df)
    df = check_duplicates(df)
    df = remove_outliers(df)

    # Étape 3 - Informations
    print("\n📊 Informations sur les données...")
    get_info(df)

    # Étape 4 - Visualisations
    print("\n🎨 Génération des graphiques...")
    plot_distributions(df)
    plot_correlation(df)
    plot_boxplots(df)

    # Étape 5 - Modèle
    print("\n🤖 Entraînement du modèle...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    model = train_model(X_train, y_train)

    # Étape 6 - Évaluation
    print("\n📈 Évaluation du modèle...")
    evaluate_model(model, X_test, y_test)

    # Étape 7 - Sauvegarde
    print("\n💾 Sauvegarde du modèle...")
    save_model(model)

    print("\n✅ Pipeline terminé avec succès !")

if __name__ == "__main__":
    main()
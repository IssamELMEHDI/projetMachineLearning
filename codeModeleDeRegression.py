#1-Importation des bibliothèques nécessaires :
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns 
import openpyxl

sns.set()

#2-Chargement des données:
# Charger les données dans un dataframe pandas
df = pd.read_csv('./Salary_Data.csv')

# Tracer le graphique
df.plot(figsize=(20,5))
plt.title("Salaire vs. Expérience",size=20)
plt.xlabel("Années d'expérience",size=20)
plt.ylabel("Salaire",size=20)
plt.show()

# Séparer les variables indépendantes (X) et la variable dépendante (y)
X = df[['YearsExperience']]
y = df['Salary']

#3-Séparation des données d'entraînement et de test :
# Séparer les données en données d'entraînement (70%) et données de test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#4-Entraînement du modèle :
# Initialiser le modèle de régression linéaire
regressor = LinearRegression()

# Entraîner le modèle sur les données d'entraînement
regressor.fit(X_train, y_train)

#5-Évaluation du modèle :
# Faire des prédictions sur les données de test
y_pred = regressor.predict(X_test)

# Calculer l'erreur quadratique moyenne et le coefficient de détermination (R²)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Erreur quadratique moyenne :', mse)
print('Coefficient de détermination (R²) :', r2)


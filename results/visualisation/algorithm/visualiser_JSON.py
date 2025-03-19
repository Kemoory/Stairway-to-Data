import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données JSON (parce que sans données, pas de chocolat)
with open('results/visualisation/algorithm/evaluation_results.json', 'r') as file:
    data = json.load(file)

# Convertir le JSON en DataFrame (le DataFrame, c'est la crème de la data)
df = pd.DataFrame(data)

# Configurer la figure matplotlib (on prépare la toile pour notre chef-d'œuvre)
fig = plt.figure(figsize=(14, 10))

# Créer une liste pour stocker les poignées (handles) et étiquettes (labels) de la légende
handles, labels = None, None

# Tracer le MAE (Mean Absolute Error, ou "Mon Amour d'Erreur")
ax1 = plt.subplot(2, 3, 1)
sns.barplot(x='preprocessing', y='MAE', hue='model', data=df, ax=ax1)
plt.title('MAE par Prétraitement et Modèle')
plt.xticks(rotation=45)
# Retirer la légende de ce sous-graphe (on la garde pour plus tard, comme un dessert)
ax1.get_legend().remove()
# Sauvegarder les poignées et étiquettes pour une utilisation ultérieure
handles, labels = ax1.get_legend_handles_labels()

# Tracer le MSE (Mean Squared Error, ou "Mon Super Échec")
ax2 = plt.subplot(2, 3, 2)
sns.barplot(x='preprocessing', y='MSE', hue='model', data=df, ax=ax2)
plt.title('MSE par Prétraitement et Modèle')
plt.xticks(rotation=45)
# Retirer la légende (on reste discret pour l'instant)
ax2.get_legend().remove()

# Tracer le RMSE (Root Mean Squared Error, ou "Racine de Mes Soucis Élevés")
ax3 = plt.subplot(2, 3, 3)
sns.barplot(x='preprocessing', y='RMSE', hue='model', data=df, ax=ax3)
plt.title('RMSE par Prétraitement et Modèle')
plt.xticks(rotation=45)
# Retirer la légende (on garde le suspense)
ax3.get_legend().remove()

# Tracer le score R2 (ou "Roi des Résultats")
ax4 = plt.subplot(2, 3, 4)
sns.barplot(x='preprocessing', y='R2_score', hue='model', data=df, ax=ax4)
plt.title('Score R2 par Prétraitement et Modèle')
plt.xticks(rotation=45)
# Retirer la légende (on la mettra ailleurs, promis)
ax4.get_legend().remove()

# Tracer l'Erreur Relative (ou "Erreur qui relativise tout")
ax5 = plt.subplot(2, 3, 5)
sns.barplot(x='preprocessing', y='Relative Error', hue='model', data=df, ax=ax5)
plt.title('Erreur Relative par Prétraitement et Modèle')
plt.xticks(rotation=45)
# Retirer la légende (on la garde pour la fin, comme un feu d'artifice)
ax5.get_legend().remove()

# Créer une légende unique dans l'espace vide en bas à droite (position 6)
ax6 = plt.subplot(2, 3, 6)
# Cacher l'axe pour ce sous-graphe (parce que personne ne veut voir ça ici)
ax6.axis('off')
# Ajouter la légende à ce sous-graphe vide (le grand final)
ax6.legend(handles, labels, loc='center')

# Ajuster la mise en page (parce qu'on aime quand c'est bien rangé)
plt.tight_layout()
plt.show()
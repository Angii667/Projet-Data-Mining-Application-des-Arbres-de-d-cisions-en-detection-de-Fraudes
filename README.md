# Détection de Fraude Financière — Data Mining End-to-End

## Description du projet
Ce projet a pour objectif de construire un modèle de détection de fraude financière en suivant un pipeline complet de Data Science / Data Mining. Il s'agit d'un Travail Pratique réalisé dans le cadre d'un cours, visant à maîtriser l'ensemble des étapes : exploration des données, prétraitement, feature engineering, gestion du déséquilibre des classes, modélisation avec un arbre de décision, évaluation et interprétation des résultats.

Le notebook original (`Projet_Data_Mining_BILAKE_Tchaa_Mèwè_Angelo.ipynb`) présente une analyse détaillée et un code bien structuré, avec des visualisations riches et des commentaires pédagogiques.

## Objectif
L'objectif métier est de détecter les transactions frauduleuses afin de minimiser les pertes financières tout en limitant les fausses alertes (frictions clients).  
D'un point de vue modélisation, l'enjeu est de traiter un **fort déséquilibre de classes** (~5% de fraudes) et de privilégier le rappel (recall) et le F1-score plutôt que l'exactitude brute.

## Jeu de données
- **Source** : `Fraud Detection Dataset.csv` (51 000 transactions / 12 colonnes)
- **Variables** :
  - Identifiants : `Transaction_ID`, `User_ID`
  - Numériques : `Transaction_Amount`, `Time_of_Transaction`, `Previous_Fraudulent_Transactions`, `Account_Age`, `Number_of_Transactions_Last_24H`
  - Catégorielles : `Transaction_Type`, `Device_Used`, `Location`, `Payment_Method`
- **Cible** : `Fraudulent` (0 = légitime, 1 = frauduleux)

## Démarche suivie
Le projet suit un pipeline structuré en plusieurs grandes étapes :

1. **Analyse Exploratoire des Données (EDA)**  
   - Distribution de la cible (déséquilibre, ratio ~4,92% de fraude)  
   - Histogrammes et boxplots des variables numériques par classe  
   - Taux de fraude par catégorie (type de transaction, appareil, localisation, méthode de paiement)  
   - Analyse temporelle (fraudes par tranche horaire, tendances nocturnes)  
   - Matrice de corrélation  
   *Des insights clairs sont dégagés : les achats en ligne, les appareils mobiles, certaines villes (Chicago, Los Angeles) et les paiements UPI sont plus risqués ; les transactions nocturnes montrent un taux de fraude plus élevé.*

2. **Prétraitement & Nettoyage**  
   - Suppression des identifiants  
   - Traitement des valeurs aberrantes par la méthode IQR (capping)  
   - Imputation des valeurs manquantes (médiane pour les numériques, mode ou catégorie explicite pour les catégorielles)  
   - Encodage des variables catégorielles avec `LabelEncoder` (adapté aux arbres de décision)

3. **Feature Engineering**  
   Création de 6 nouvelles features métier :
   - `is_high_amount` : montant > P90  
   - `amount_per_account_age` : ratio montant / âge du compte  
   - `tx_velocity_flag` : nombre de transactions en 24h > P80  
   - `risk_score` : score composite pondéré (historique fraude, montant élevé, vélocité)  
   - `is_night_transaction` : transaction nocturne (00h-06h)  
   - `avg_amount_per_tx` : montant moyen par transaction sur 24h  
   *L'analyse de l'information mutuelle et des corrélations montre un pouvoir discriminant modéré pour certaines features, notamment `is_night_transaction`.*

4. **Gestion du Déséquilibre des Classes**  
   - Démonstration du problème du modèle naïf (prédiction "tout légitime" → accuracy trompeuse)  
   - Application de **SMOTE** (synthetic minority oversampling) pour rééquilibrer l'ensemble d'entraînement (ratio 0.45)

5. **Modélisation**  
   - Premier modèle : `DecisionTreeClassifier` sans contraintes → **overfitting massif** (F1 train 100%, test 5.86%)  
   - Diagnostic et préparation pour le tuning des hyperparamètres (non réalisé dans le notebook, mais identifié comme étape suivante)

6. **Évaluation**  
   Une fonction d'évaluation complète a été définie, produisant :
   - Accuracy, précision, rappel, F1-score  
   - AUC-ROC, AUC-PR (référence pour données déséquilibrées)  
   - Matrice de confusion avec interprétation métier (vrais/faux positifs/négatifs)  
   - Rapport de classification détaillé

## Résultats notables
- **Importance du déséquilibre** : L'accuracy sans traitement est trompeuse (95% pour un modèle naïf).
- **Feature discriminante** : `is_night_transaction` se détache visuellement comme une variable à fort pouvoir séparateur.
- **Risque d'overfitting** : L'arbre non contraint mémorise les données d'entraînement, rendant nécessaire une régularisation (profondeur max, min_samples_leaf, etc.).
- **Qualité de l'analyse** : L'EDA est très fouillée, avec des visualisations claires et des commentaires métier pertinents.

## Points forts
- **Pipeline complet** : toutes les étapes d'un projet de data mining sont couvertes, de la compréhension des données à l'évaluation.
- **Gestion réaliste du déséquilibre** : utilisation de SMOTE et focalisation sur les métriques adaptées (F1, AUC-PR, recall).
- **Feature engineering métier** : création de variables interprétables (score de risque, transactions nocturnes) qui font sens pour un analyste fraude.
- **Visualisations de qualité** : histogrammes, boxplots, cartes de corrélation, matrices de confusion annotées.
- **Pédagogie** : chaque section est précédée d'un texte explicatif sur l'objectif et l'importance de l'analyse.
- **Reproductibilité** : graine aléatoire fixée, chemins explicites, environnement documenté.
- **Modularité** : fonctions d'évaluation réutilisables pour comparer plusieurs modèles.

## Faiblesses et limites
- **Modèle unique** : seul l'arbre de décision est exploré. L'absence de comparaison avec d'autres classifieurs (Random Forest, XGBoost, régression logistique pénalisée) limite les conclusions.
- **Tuning insuffisant** : l'arbre n'a pas été réglé ; le notebook s'arrête après avoir diagnostiqué l'overfitting sans montrer l'optimisation des hyperparamètres.
- **Feature engineering limité** : les variables créées ont un pouvoir discriminant modeste (faible information mutuelle). Des interactions ou des transformations plus poussées pourraient être bénéfiques.
- **Validation croisée absente** : l'évaluation repose sur un seul split train/test, ce qui peut donner une estimation biaisée de la performance.
- **Pas de démonstration de déploiement** : bien que mentionné dans la structure, le déploiement n'est pas traité.
- **Encodage simplifié** : l'utilisation de `LabelEncoder` pour des variables catégorielles peut introduire une notion d'ordre factice ; un encodage one-hot serait préférable pour des modèles non arborescents.

## Pistes d'amélioration
1. **Comparer plusieurs modèles** : Random Forest, Gradient Boosting (XGBoost, LightGBM), régression logistique avec régularisation et pondération des classes.
2. **Tuning approfondi** : utiliser `GridSearchCV` ou `RandomizedSearchCV` pour optimiser les hyperparamètres de l'arbre (profondeur, min_samples_split, min_samples_leaf, critère).
3. **Enrichir le feature engineering** :
   - Créer des interactions (ex : montant × fréquence)
   - Encodage des catégories par leur taux de fraude (target encoding)
   - Features temporelles plus fines (jour de la semaine, week-end)
   - Détection de dépassement de seuils personnalisés par utilisateur
4. **Validation croisée stratifiée** pour une estimation robuste de la performance.
5. **Analyse des erreurs** : étudier les faux négatifs pour comprendre les fraudes manquées et améliorer le ciblage.
6. **Interprétabilité avancée** : afficher les règles de décision de l'arbre (`export_text`, `plot_tree`), calculer les SHAP values pour comprendre l'impact des features.
7. **Equilibrer le jeu de données** de manière plus fine, par exemple en combinant SMOTE et sous-échantillonnage, ou en testant des pondérations de classes adaptées.
8. **Mettre en place un suivi de dérive** : le modèle devrait être monitoré en production pour détecter une évolution des patterns de fraude.

## Conclusion
Ce notebook constitue un excellent exemple de projet data mining appliqué à la fraude financière. Il démontre une bonne compréhension des enjeux métier et des techniques de data science. Les visualisations et l'analyse exploratoire sont particulièrement soignées. Le projet gagnerait à être complété par une phase de tuning et de comparaison de modèles, ainsi que par un enrichissement du feature engineering, pour aboutir à un modèle performant et prêt pour une mise en production.

---

## Auteur
**BILAKE Tchaa Mèwè Angelo**  
*Projet réalisé dans le cadre d’un TP de Data Mining.*

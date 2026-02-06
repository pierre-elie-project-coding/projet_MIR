# Fiche Récapitulative : Implémentation XGBoost pour Prédiction de Paramètres et États

## 📋 Objectif Global

Prédire à partir des données brutes `read_data` :
1. **Paramètres** (`read_par`) : caractéristiques globales du signal
2. **États** (`values_par`) : séquence temporelle d'états discrets

---

## 🏗️ Architecture Générale

### Fichier Principal
[src/prediction.py](src/prediction.py)

### Classes Principales

#### 1. `DataProcessor`
- **Rôle** : Chargement et préparation des données
- **Méthodes** :
  - `load_all_data(stop)` : Charge `read_data`, `read_par`, et `values_par`
  - `extract_global_features(read_data_list)` : Features statistiques du signal
  - `prepare_window_data(df, window_size)` : Fenêtres glissantes pour états

#### 2. `XGBoostPredictor`
- **Rôle** : Entraînement et prédiction des modèles XGBoost
- **Attributs** :
  - `param_models` : Dictionnaire de régresseurs (un par paramètre)
  - `state_model` : Classificateur pour les états
  - `state_classes`, `state_to_idx`, `idx_to_state` : Mapping des classes
- **Méthodes** :
  - `train_param_models(X, y_df)` : Entraîne les régresseurs
  - `train_state_model(X, y)` : Entraîne le classificateur
  - `predict_params(X)` : Prédit tous les paramètres
  - `predict_states(X)` : Prédit les états
  - `smooth_piecewise_constant(predictions)` : Lissage post-traitement

---

## 📊 Modèle 1 : Prédiction des Paramètres

### Type de Problème
**Régression multi-output** (un modèle par paramètre)

### Features d'Entrée
Statistiques globales calculées sur `read_data` :

| Feature | Description |
|---------|-------------|
| `mean` | Moyenne du signal |
| `std` | Écart-type |
| `min` | Valeur minimale |
| `max` | Valeur maximale |
| `q25` | 1er quartile |
| `q50` | Médiane |
| `q75` | 3e quartile |
| `len` | Longueur du signal |

### Targets (Sorties)
Paramètres dans `read_par` :
- `maxv`, `minv`, `pulselen`, `inct`, `dect`
- `speed` (moyenne de la liste si variable-length)
- `val_background`, `speed_th`, `resolution`

### Modèle XGBoost
```python
XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100
)
```

### Traitement Spécial : Speed
- **Problème** : `speed` peut être une liste de longueur variable `[]`, `[2650.0]`, `[2400.0, 2400.0, 2450.0]`
- **Solution** : Prendre la **moyenne** des valeurs (0 si liste vide)

```python
if isinstance(sample_val, list):
    y_params[col] = y_params_raw[col].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
```

### Nettoyage des Données
- Suppression des lignes avec valeurs `NaN` dans features ou targets
- Validation : `~y_params.isna().any(axis=1) & ~X_params.isna().any(axis=1)`

### Évaluation
- **Métrique** : RMSE (Root Mean Squared Error) par paramètre
- **Split** : 80% train, 20% test

---

## 🎯 Modèle 2 : Prédiction des États

### Type de Problème
**Classification multi-classe** (7 classes)

### Classes Possibles
États discrets détectés automatiquement : `[-3, -2, -1, 0, 1, 2, 3]`

> **Important** : Les classes sont découvertes dynamiquement depuis les données d'entraînement avec `np.unique(y)` pour gérer la variabilité.

---

## 🪟 Concept : Fenêtre Glissante (Sliding Window)

### Qu'est-ce qu'une fenêtre glissante ?

Une **fenêtre glissante** est une technique qui consiste à extraire des "morceaux" successifs du signal en déplaçant une fenêtre de taille fixe le long de la séquence.

### Pourquoi l'utiliser ?

**Sans fenêtre** : Le modèle ne verrait qu'une seule valeur → difficile de deviner l'état

**Avec fenêtre** : Le modèle voit le **contexte local** (montée, descente, plateau) → peut mieux prédire l'état

### Implémentation dans le code

```python
def prepare_window_data(self, df, window_size=10):
    sig_arr = np.array(signal)
    pad_width = window_size // 2  # 5 si window_size=10
    
    # Padding pour gérer les bords
    sig_padded = np.pad(sig_arr, pad_width, mode='edge')
    
    for i in range(len(states)):
        # Fenêtre centrée sur i
        window = sig_padded[i : i + window_size]
        X_windows.append(window)
        y_windows.append(states[i])
```

### Exemple Concret

**Signal** : `[0.05, 0.06, 0.08, 0.12, 0.25, 0.30, 0.28, 0.15, 0.10, 0.07, ...]` (500 points)

**Avec fenêtre de taille 10 centrée** :

```
Position 0:  [0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.08, 0.12, 0.25, 0.30]
              └────── padding (edge) ─────┘  └──── contexte local ────┘

Position 5:  [0.05, 0.06, 0.08, 0.12, 0.25, 0.30, 0.28, 0.15, 0.10, 0.07]
              └──────────────── contexte autour de position 5 ──────────┘

Position 10: [0.12, 0.25, 0.30, 0.28, 0.15, 0.10, 0.07, 0.06, 0.05, 0.04]
              └──────────────── contexte autour de position 10 ─────────┘
```

**Schéma visuel** :

```
Signal complet (500 points):
┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│ │ │ │ │█│ │ │ │ │ │ │ │ │ │ │  ← Position i=4
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

Fenêtre centrée (taille 10):
    ┌───────────────────┐
    │ │ │ │ │█│ │ │ │ │ │          ← 10 valeurs autour de i=4
    └───────────────────┘
    
La fenêtre "glisse" à la position suivante:
      ┌───────────────────┐
      │ │ │ │ │ │█│ │ │ │ │        ← 10 valeurs autour de i=5
      └───────────────────┘
```

### Résultat du windowing

- **Entrée** : 1 signal de 500 points
- **Sortie** : 500 fenêtres de 10 points chacune
- **Pour XGBoost** : 500 échantillons d'entraînement

### Modèle XGBoost
```python
XGBClassifier(
    objective='multi:softmax',
    num_class=len(state_classes),  # Détecté automatiquement
    n_estimators=150,
    max_depth=5
)
```

### Encodage des Classes
1. **Avant entraînement** : États réels → Indices 0-based
   ```python
   unique_states = np.unique(y)  # Ex: [-2, -1, 0, 1, 2, 3]
   state_to_idx = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5}
   ```
2. **Après prédiction** : Indices → États réels (décodage inverse)
   ```python
   y_pred_idx = self.state_model.predict(X)  # [0, 1, 2, ...]
   return [self.idx_to_state[int(idx)] for idx in y_pred_idx]  # [-2, -1, 0, ...]
   ```

---

## 🧹 Post-Traitement : Lissage Constant par Morceaux

### Problème

Le classificateur prédit un état à chaque position, mais les prédictions peuvent être bruitées :

```
Prédictions brutes XGBoost:
[-1, -1, -1,  2, -1, -1, -1, -1, -1,  2,  2,  2,  1,  2,  2, ...]
             ↑                    ↑           ↑
          Outliers / Bruit ponctuel

Vrais états (plateaux constants):
[-1, -1, -1, -1, -1, -1, -1, -1, -1,  2,  2,  2,  2,  2,  2, ...]
```

### Algorithme `smooth_piecewise_constant`

#### Étape 1️⃣ : Filtre Médian (Débruitage)

```python
from scipy.ndimage import median_filter
smoothed = median_filter(predictions, size=5)
```

**Action** : Remplace chaque valeur par la **médiane** de ses 5 voisins

```
Avant:  [-1, -1, -1,  2, -1, -1, -1, -1, -1,  2,  2,  2,  1,  2,  2]
                     ↓                                   ↓
Après:  [-1, -1, -1, -1, -1, -1, -1, -1, -1,  2,  2,  2,  2,  2,  2]
                  Outliers supprimés ✓
```

#### Étape 2️⃣ : Détection des Points de Changement

**Principe** : Parcourir le signal lissé avec un pas de `window_size // 2` et comparer les moyennes avant/après

```python
window_size = 20  # Paramétrable (défaut: 20, appelé avec 30)
threshold = 0.3   # Paramétrable (défaut: 0.3, appelé avec 0.5)

for i in range(window_size, n, window_size // 2):  # Pas de 10 si window=20
    window_before = smoothed[max(0, i-window_size):i]
    window_after = smoothed[i:min(n, i+window_size)]
    
    mean_before = np.mean(window_before)
    mean_after = np.mean(window_after)
    
    # Changement significatif détecté ?
    if abs(mean_after - mean_before) > threshold:
        segments.append((current_start, i, np.median(smoothed[current_start:i])))
        current_start = i
```

**Visualisation de la détection** :

```
Signal après filtre médian:
┌────────────────────────────────────────────────────┐
│-1 -1 -1 -1 -1 -1│ 2  2  2  2  2  2│ 1  1  1  1  1  │
└────────────────────────────────────────────────────┘
Position i=50 →   ↑
                Point de changement potentiel

Fenêtre avant [30:50]:        Fenêtre après [50:70]:
┌─────────────────┐            ┌─────────────────┐
│-1 -1 -1 -1 -1 -1│            │ 2  2  2  2  2  2│
└─────────────────┘            └─────────────────┘
mean = -1.0                    mean = 2.0

|2.0 - (-1.0)| = 3.0 > 0.5 ✓ → Point de changement détecté !
```

#### Étape 3️⃣ : Création des Segments Constants

Pour chaque segment, appliquer la **médiane** de toutes les valeurs du segment :

```python
segments = [
    (0, 50, -1.0),      # Segment 1: positions 0-50, valeur = médiane(smoothed[0:50])
    (50, 120, 2.0),     # Segment 2: positions 50-120, valeur = médiane(smoothed[50:120])
    (120, 200, 1.0)     # Segment 3: positions 120-200, valeur = médiane(smoothed[120:200])
]

result = np.zeros_like(predictions)
for start, end, value in segments:
    result[start:end] = value  # Toutes les positions du segment = valeur constante
```

**Résultat final** :

```
Prédictions lissées (piecewise constant):
┌──────────────────────────────────────────────────────┐
│ -1 -1 -1 -1 -1 -1 -1 -1 -1 ... -1 (50 fois)         │  Plateau 1
│  2  2  2  2  2  2  2  2  2 ...  2 (70 fois)         │  Plateau 2
│  1  1  1  1  1  1  1  1  1 ...  1 (80 fois)         │  Plateau 3
└──────────────────────────────────────────────────────┘
```

### Paramètres Ajustables

Dans le code, la fonction est appelée avec :
```python
pred_s_smooth = predictor.smooth_piecewise_constant(pred_s, window_size=30, threshold=0.5)
```

#### `window_size` (utilisé: 30)
- **Petit** (10-20) : Détecte changements courts → segments plus courts, sensible aux variations
- **Grand** (40-60) : Ignore petites variations → segments plus longs, plus stable

#### `threshold` (utilisé: 0.5)
- **Petit** (0.2-0.3) : Très sensible → détecte même petits changements → plus de segments
- **Grand** (0.8-1.0) : Peu sensible → ne détecte que grands changements → moins de segments

### Exemple Complet de Transformation

```python
# 1. Prédictions brutes du classificateur
pred_raw = [0, 0, -1, 0, 0, 0, 1, 0, 0, 2, 2, 1, 2, 2, 2, 3, 2, 2, 3, 3, 3, ...]

# 2. Après filtre médian (size=5)
after_median = [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, ...]

# 3. Segments détectés (window_size=30, threshold=0.5)
segments = [(0, 95, 0), (95, 180, 2), (180, 250, 3)]

# 4. Résultat final lissé
pred_smooth = [0, 0, 0, ..., 0 (×95), 2, 2, 2, ..., 2 (×85), 3, 3, 3, ..., 3 (×70)]
                └─ Plateau 1 ─┘      └─ Plateau 2 ─┘       └─ Plateau 3 ─┘
```

### Pourquoi ça marche ?

Les **vrais états** ont une structure en plateaux (changements peu fréquents). Le lissage :
1. **Supprime le bruit ponctuel** (filtre médian)
2. **Identifie les vrais changements de régime** (comparaison de moyennes)
3. **Force la constance dans chaque segment** (médiane par segment)

**Résultat** : Prédictions qui reproduisent la structure en plateaux des vrais états ! ✨

---

## 📈 Résultats Typiques

### Paramètres (RMSE)
- Excellente précision : `pulselen`, `resolution` (RMSE ≈ 0.0)
- Bonne précision : `maxv`, `minv`, `val_background` (RMSE < 0.1)
- Variable : `speed`, `inct`, `dect` (dépend de la variabilité)

### États (Accuracy)
- Dépend de la qualité du signal et de la taille d'entraînement
- **Le lissage améliore significativement la lisibilité visuelle** sans changer l'accuracy brute

---

## 🎨 Visualisation

### Graphiques Générés
1. **Signal brut** (`read_data`)
2. **États** :
   - **Vert (ligne pleine)** : Vrais états (ground truth)
   - **Orange (pointillé transparent)** : Prédictions brutes du classificateur
   - **Rouge (tirets épais)** : Prédictions lissées (piecewise constant)
3. **Paramètres** : Comparaison True vs Pred en format dictionnaire

---

## 🔧 Pipeline Complet

### 1. Chargement des Données
```python
processor = DataProcessor(base_path="data")
df = processor.load_all_data(stop=200)  # stop=None pour toutes les données
```

### 2. Entraînement Paramètres
```python
# Features globales (8 statistiques)
X_params = processor.extract_global_features(df['read_data'])

# Targets avec gestion des listes
y_params_raw = pd.DataFrame(df['read_par'].tolist())
y_params = pd.DataFrame()
for col in y_params_raw.columns:
    if isinstance(y_params_raw[col].iloc[0], list):
        y_params[col] = y_params_raw[col].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    else:
        y_params[col] = y_params_raw[col]

# Nettoyage NaN
valid_mask = ~y_params.isna().any(axis=1) & ~X_params.isna().any(axis=1)
X_params = X_params[valid_mask].reset_index(drop=True)
y_params = y_params[valid_mask].reset_index(drop=True)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_params, y_params, test_size=0.2)

# Entraînement (un modèle par paramètre)
predictor = XGBoostPredictor()
predictor.train_param_models(X_train, y_train)
```

### 3. Entraînement États
```python
# Fenêtres glissantes (taille 10, centrées)
subset_df = df.iloc[:20]
window_size = 10
X_states, y_states = processor.prepare_window_data(subset_df, window_size)

# Train/Test split
X_st_train, X_st_test, y_st_train, y_st_test = train_test_split(X_states, y_states, test_size=0.2)

# Entraînement (classification multi-classe)
predictor.train_state_model(X_st_train, y_st_train)
```

### 4. Prédiction et Visualisation
```python
# Prédiction paramètres
feat = processor.extract_global_features([signal])
pred_params = predictor.predict_params(feat)

# Prédiction états (recréer les fenêtres)
sig_padded = np.pad(signal, window_size//2, mode='edge')
windows = [sig_padded[i:i+window_size] for i in range(len(signal))]
pred_states = predictor.predict_states(np.array(windows))

# Lissage post-traitement
pred_states_smooth = predictor.smooth_piecewise_constant(
    pred_states, 
    window_size=30, 
    threshold=0.5
)
```

---

## 📝 Points Clés de l'Implémentation

### Défis Résolus

1. **Listes de longueur variable** 
   - Problème : `speed` peut être `[]`, `[v1]`, ou `[v1, v2, v3]`
   - Solution : Agrégation par moyenne

2. **Valeurs NaN**
   - Problème : Certaines entrées manquantes dans `read_par`
   - Solution : Filtrage strict avant entraînement

3. **Classes dynamiques**
   - Problème : Classes d'états peuvent varier selon le dataset
   - Solution : Découverte automatique avec `np.unique(y)`

4. **Encodage XGBoost**
   - Problème : XGBoost exige classes 0-based
   - Solution : Mapping bidirectionnel `state_to_idx` / `idx_to_state`

5. **Prédictions bruitées**
   - Problème : États prédits oscillent au lieu de former des plateaux
   - Solution : Post-traitement en 3 étapes (médian → détection → segments)

### Choix de Design

| Aspect | Choix | Justification |
|--------|-------|---------------|
| **Paramètres** | Régression | Valeurs continues |
| **États** | Classification | Ensemble discret fini |
| **Features (params)** | Statistiques globales | Capture propriétés du signal entier |
| **Features (états)** | Fenêtres locales | États dépendent du contexte temporel local |
| **Fenêtrage** | Centrée avec padding | Contexte symétrique autour de chaque point |
| **Lissage** | Piecewise constant | Reproduit structure en plateaux des vrais états |

---

## 🚀 Utilisation

### Script Standalone
```bash
python3 src/prediction.py
```
Génère `prediction_result2.png` avec visualisations.

### Dans un Notebook
Voir `DataBio.ipynb` pour code de boucle sur plusieurs IDs avec comparaison graphique et textuelle.

---

## 🔮 Améliorations Possibles

1. **Features avancées** : 
   - Transformée de Fourier (fréquences dominantes)
   - Autocorrélation
   - Détection de pics
   - Dérivées temporelles

2. **Modèles alternatifs** :
   - CatBoost ou LightGBM
   - Réseau de neurones (LSTM/GRU pour séquences)
   - Transformers pour modélisation temporelle

3. **Hyperparamètre tuning** :
   - Grid search sur `n_estimators`, `max_depth`, `learning_rate`
   - Validation croisée temporelle
   - Bayesian optimization

4. **Lissage adaptatif** :
   - Algorithme de Viterbi pour séquences d'états optimales
   - HMM (Hidden Markov Model)
   - Dynamic programming pour transitions

5. **Ensembling** :
   - Combiner XGBoost + LightGBM + CatBoost
   - Stacking avec meta-learner
   - Voting classifier

---

## 📚 Bibliothèques Utilisées
- **XGBoost** : Gradient boosting optimisé
- **scikit-learn** : Preprocessing, métriques, train/test split
- **pandas/numpy** : Manipulation et calcul sur données
- **matplotlib** : Visualisation des résultats
- **scipy** : Filtres de signal (median_filter)


Prédire à partir des données brutes `read_data` :
1. **Paramètres** (`read_par`) : caractéristiques globales du signal
2. **États** (`values_par`) : séquence temporelle d'états discrets

---

## 🏗️ Architecture Générale

### Fichier Principal
[src/prediction.py](src/prediction.py)

### Classes Principales

#### 1. `DataProcessor`
- **Rôle** : Chargement et préparation des données
- **Méthodes** :
  - `load_all_data(stop)` : Charge `read_data`, `read_par`, et `values_par`
  - `extract_global_features(read_data_list)` : Features statistiques du signal
  - `prepare_window_data(df, window_size)` : Fenêtres glissantes pour états

#### 2. `XGBoostPredictor`
- **Rôle** : Entraînement et prédiction des modèles XGBoost
- **Attributs** :
  - `param_models` : Dictionnaire de régresseurs (un par paramètre)
  - `state_model` : Classificateur pour les états
  - `state_classes`, `state_to_idx`, `idx_to_state` : Mapping des classes
- **Méthodes** :
  - `train_param_models(X, y_df)` : Entraîne les régresseurs
  - `train_state_model(X, y)` : Entraîne le classificateur
  - `predict_params(X)` : Prédit tous les paramètres
  - `predict_states(X)` : Prédit les états
  - `smooth_piecewise_constant(predictions)` : Lissage post-traitement

---

## 📊 Modèle 1 : Prédiction des Paramètres

### Type de Problème
**Régression multi-output** (un modèle par paramètre)

### Features d'Entrée
Statistiques globales calculées sur `read_data` :

| Feature | Description |
|---------|-------------|
| `mean` | Moyenne du signal |
| `std` | Écart-type |
| `min` | Valeur minimale |
| `max` | Valeur maximale |
| `q25` | 1er quartile |
| `q50` | Médiane |
| `q75` | 3e quartile |
| `len` | Longueur du signal |

### Targets (Sorties)
Paramètres dans `read_par` :
- `maxv`, `minv`, `pulselen`, `inct`, `dect`
- `speed` (moyenne de la liste si variable-length)
- `val_background`, `speed_th`, `resolution`

### Modèle XGBoost
```python
XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100
)
```

### Traitement Spécial : Speed
- Problème : `speed` peut être une liste de longueur variable `[]`, `[2650.0]`, `[2400.0, 2400.0, 2450.0]`
- Solution : Prendre la **moyenne** des valeurs (0 si liste vide)

### Nettoyage des Données
- Suppression des lignes avec valeurs `NaN` dans features ou targets
- Validation : `~y_params.isna().any(axis=1) & ~X_params.isna().any(axis=1)`

### Évaluation
- **Métrique** : RMSE (Root Mean Squared Error) par paramètre
- **Split** : 80% train, 20% test

---

## 🎯 Modèle 2 : Prédiction des États

### Type de Problème
**Classification multi-classe** (7 classes)

### Classes Possibles
États discrets détectés automatiquement : `[-3, -2, -1, 0, 1, 2, 3]`

> **Important** : Les classes sont découvertes dynamiquement depuis les données d'entraînement avec `np.unique(y)` pour gérer la variabilité.

### Features d'Entrée
**Fenêtre glissante** (sliding window) sur `read_data` :
- Taille de fenêtre : 10 points
- Type : Fenêtre centrée avec padding (`mode='edge'`)
- Pour chaque position `i`, fenêtre = `signal[i-5 : i+5]`

#### Exemple
Signal de longueur 500 → 500 fenêtres de 10 valeurs → 500 échantillons d'entraînement

### Modèle XGBoost
```python
XGBClassifier(
    objective='multi:softmax',
    num_class=len(state_classes),  # Détecté automatiquement
    n_estimators=150,
    max_depth=5
)
```

### Encodage des Classes
1. **Avant entraînement** : États réels → Indices 0-based
   - `-2 → 0`, `-1 → 1`, `0 → 2`, `1 → 3`, `2 → 4`, `3 → 5`
2. **Après prédiction** : Indices → États réels (décodage inverse)

### Post-Traitement : Lissage Constant par Morceaux

#### Méthode `smooth_piecewise_constant`
1. **Filtre médian** (taille 5) : débruitage initial
2. **Détection des points de changement** :
   - Fenêtre glissante (taille paramétrable, défaut 20)
   - Comparaison des moyennes avant/après
   - Seuil de changement significatif (défaut 0.3)
3. **Création de segments constants** :
   - Chaque segment prend la médiane des valeurs du segment

#### Paramètres Ajustables
- `window_size=30` : Taille fenêtre de détection
- `threshold=0.5` : Seuil de détection de changement

> **Astuce** : Augmenter `window_size` et `threshold` → plateaux plus longs et lisses

### Évaluation
- **Métrique** : Accuracy (pourcentage de prédictions correctes)
- **Split** : 80% train, 20% test

---

## 🔧 Pipeline Complet

### 1. Chargement des Données
```python
processor = DataProcessor(base_path="data")
df = processor.load_all_data(stop=200)  # stop=None pour toutes les données
```

### 2. Entraînement Paramètres
```python
# Features globales
X_params = processor.extract_global_features(df['read_data'])

# Targets avec gestion des listes
y_params_raw = pd.DataFrame(df['read_par'].tolist())
y_params = pd.DataFrame()
for col in y_params_raw.columns:
    if isinstance(y_params_raw[col].iloc[0], list):
        y_params[col] = y_params_raw[col].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    else:
        y_params[col] = y_params_raw[col]

# Nettoyage NaN
valid_mask = ~y_params.isna().any(axis=1) & ~X_params.isna().any(axis=1)
X_params = X_params[valid_mask].reset_index(drop=True)
y_params = y_params[valid_mask].reset_index(drop=True)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_params, y_params, test_size=0.2)

# Entraînement
predictor = XGBoostPredictor()
predictor.train_param_models(X_train, y_train)
```

### 3. Entraînement États
```python
# Fenêtres glissantes
subset_df = df.iloc[:20]
window_size = 10
X_states, y_states = processor.prepare_window_data(subset_df, window_size)

# Train/Test split
X_st_train, X_st_test, y_st_train, y_st_test = train_test_split(X_states, y_states, test_size=0.2)

# Entraînement
predictor.train_state_model(X_st_train, y_st_train)
```

### 4. Prédiction et Visualisation
```python
# Prédiction paramètres
feat = processor.extract_global_features([signal])
pred_params = predictor.predict_params(feat)

# Prédiction états
sig_padded = np.pad(signal, window_size//2, mode='edge')
windows = [sig_padded[i:i+window_size] for i in range(len(signal))]
pred_states = predictor.predict_states(np.array(windows))
pred_states_smooth = predictor.smooth_piecewise_constant(pred_states)
```

---

## 📈 Résultats Typiques

### Paramètres (RMSE)
- Excellente précision : `pulselen`, `resolution` (RMSE ≈ 0.0)
- Bonne précision : `maxv`, `minv`, `val_background` (RMSE < 0.1)
- Variable : `speed`, `inct`, `dect` (dépend de la variabilité)

### États (Accuracy)
- Dépend de la qualité du signal et de la taille d'entraînement
- Lissage améliore significativement la lisibilité visuelle

---

## 🎨 Visualisation

### Graphiques Générés
1. **Signal brut** (`read_data`)
2. **États** :
   - Vert : Vrais états
   - Orange pointillé : Prédictions brutes
   - Rouge tirets : Prédictions lissées (piecewise constant)
3. **Paramètres** : Comparaison True vs Pred en format dictionnaire

---

## 📝 Points Clés de l'Implémentation

### Défis Résolus

1. **Listes de longueur variable** 
   - Problème : `speed` peut être `[]`, `[v1]`, ou `[v1, v2, v3]`
   - Solution : Agrégation par moyenne

2. **Valeurs NaN**
   - Problème : Certaines entrées manquantes dans `read_par`
   - Solution : Filtrage strict avant entraînement

3. **Classes dynamiques**
   - Problème : Classes d'états peuvent varier selon le dataset
   - Solution : Découverte automatique avec `np.unique(y)`

4. **Encodage XGBoost**
   - Problème : XGBoost exige classes 0-based
   - Solution : Mapping bidirectionnel `state_to_idx` / `idx_to_state`

5. **Prédictions bruitées**
   - Problème : États prédits oscillent au lieu de former des plateaux
   - Solution : Post-traitement avec détection de changements

### Choix de Design

| Aspect | Choix | Justification |
|--------|-------|---------------|
| **Paramètres** | Régression | Valeurs continues |
| **États** | Classification | Ensemble discret fini |
| **Features (params)** | Statistiques globales | Capture propriétés du signal entier |
| **Features (états)** | Fenêtres locales | États dépendent du contexte local |
| **Lissage** | Piecewise constant | Reproduit structure en plateaux des vrais états |

---

## 🚀 Utilisation

### Script Standalone
```bash
python3 src/prediction.py
```
Génère `prediction_result.png` avec visualisations.

### Dans un Notebook
Voir `DataBio.ipynb` pour code de boucle sur plusieurs IDs avec comparaison graphique et textuelle.

---

## 🔮 Améliorations Possibles

1. **Features avancées** : 
   - Transformée de Fourier (fréquences dominantes)
   - Autocorrélation
   - Détection de pics

2. **Modèles alternatifs** :
   - CatBoost ou LightGBM
   - Réseau de neurones (LSTM pour les états)

3. **Hyperparamètre tuning** :
   - Grid search sur `n_estimators`, `max_depth`, `learning_rate`
   - Validation croisée

4. **Lissage adaptatif** :
   - Viterbi algorithm pour séquences d'états
   - HMM (Hidden Markov Model)

5. **Ensembling** :
   - Combiner plusieurs modèles
   - Stacking avec meta-learner

---

## 📚 Bibliothèque Utilisée
- **XGBoost** : Gradient boosting optimisé
- **scikit-learn** : Preprocessing et métriques
- **pandas/numpy** : Manipulation de données
- **matplotlib** : Visualisation
- **scipy** : Filtres (median_filter)

# Importando bibliotecas, módulos e recursos
import optuna
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from optuna import create_study, Trial
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate


# Constantes
SEMENTE_ALEATORIA = 0
TAMANHO_DIVISAO_TESTE = 0.1
NUM_FOLDS = 10
NUM_TENTATIVAS_OTIMIZACAO = 1000
MAIN_METRIC = "roc_auc"
METRICS = ["roc_auc", "accuracy"]


# Importando dados tratado e utilizando binarizador no target
df = pd.read_csv("/home/diogo23039/work/GF/2024_1/data/atributos_engenheirados.csv.zip", sep = ",")

features_columns = []
for column in df.columns:
    features_columns.append(column)
features_columns.remove('GF')

FEATURES = features_columns
TARGET = ['GF']


#FEATURES = df.columns[:-1]
#TARGET = df.columns[-1]

df = df.reindex(FEATURES + TARGET, axis=1)

binarizador = LabelBinarizer()
binarizador.fit(df[TARGET])

nomes_das_features = binarizador.classes_

print(binarizador.transform(df[TARGET]))

df[TARGET] = binarizador.transform(df[TARGET])
df


# Split de dados
indices = df.index
indices_treino, indices_teste = train_test_split(
    indices, test_size=TAMANHO_DIVISAO_TESTE, random_state=SEMENTE_ALEATORIA, stratify=df['GF']
)

df_treino = df.loc[indices_treino]
df_teste = df.loc[indices_teste]

X_treino = df_treino.reindex(FEATURES, axis=1).values
y_treino = df_treino.reindex(TARGET, axis=1).values.ravel()

X_teste = df_teste.reindex(FEATURES, axis=1).values
y_teste = df_teste.reindex(TARGET, axis=1).values.ravel()

X = df.reindex(FEATURES, axis=1).values
y = df.reindex(TARGET, axis=1).values.ravel()

def cria_instancia_modelo(trial):
    """Cria uma instância do modelo.

    Args:
      trial: objeto tipo Trial do optuna.

    Returns:
      Uma instância do modelo desejado.

    """
    parametros = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 500, log=True),
        "criterion": trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        ),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "n_jobs": -1,
        "bootstrap": True,
        "random_state": SEMENTE_ALEATORIA,
    }

    has_max_depth = trial.suggest_categorical(
        "has_max_depth", [True, False]
    )
    if has_max_depth:
        parametros["max_depth"] = trial.suggest_int(
            "max_depth", 1, 100, log=True
        )
    else:
        parametros["max_depth"] = None

    max_features_is_float = trial.suggest_categorical(
        "max_features_is_float", [True, False]
    )
    if max_features_is_float:
        parametros["max_features"] = trial.suggest_float(
            "max_features_float", 1e-5, 1, log=True
        )
    else:
        parametros["max_features"] = trial.suggest_categorical(
            "max_features_categorical", ["sqrt", "log2", None]
        )

    has_max_leaf_nodes = trial.suggest_categorical(
        "has_max_leaf_nodes", [True, False]
    )
    if has_max_leaf_nodes:
        parametros["max_leaf_nodes"] = trial.suggest_int(
            "max_leaf_nodes", 10, 1000, log=True
        )
    else:
        parametros["max_leaf_nodes"] = None

    min_impurity_decrease_is_float = trial.suggest_categorical(
        "min_impurity_decrease_is_float", [True, False]
    )
    if min_impurity_decrease_is_float:
        parametros["min_impurity_decrease"] = trial.suggest_float(
            "min_impurity_decrease", 1e-5, 1, log=True
        )
    else:
        parametros["min_impurity_decrease"] = 0

    has_class_weight = trial.suggest_categorical(
        "has_class_weight", [True, False]
    )
    if has_class_weight:
        parametros["class_weight"] = trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample"]
        )
    else:
        parametros["class_weight"] = None

    has_ccp_alpha = trial.suggest_categorical(
        "has_ccp_alpha", [True, False]
    )
    if has_ccp_alpha:
        parametros["ccp_alpha"] = trial.suggest_float("ccp_alpha", 1e-5, 1, log=True)
    else:
        parametros["ccp_alpha"] = 0


    max_samples_is_float = trial.suggest_categorical(
        "max_samples_is_float", [True, False]
    )
    if max_samples_is_float:
        parametros["max_samples"] = trial.suggest_float("max_samples", 1e-5, 1, log=True)
    else:
        parametros["max_samples"] = None

    model = RandomForestClassifier(**parametros)

    return model

def funcao_objetivo(
    trial,
    X,
    y,
    num_folds=NUM_FOLDS,
):
    """Função objetivo do optuna

    Referencia:
      https://medium.com/@walter_sperat/using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451
    """

    modelo = cria_instancia_modelo(trial)

    metricas = cross_val_score(
        modelo,
        X,
        y,
        scoring=MAIN_METRIC,
        cv=NUM_FOLDS,
    )

    return np.min([np.mean(metricas), np.median([metricas])])


objeto_de_estudo = create_study(direction="maximize")

def funcao_objetivo_parcial(trial):
    return funcao_objetivo(trial, X_treino, y_treino)

for _ in range(NUM_TENTATIVAS_OTIMIZACAO):
    objeto_de_estudo.optimize(funcao_objetivo_parcial, n_trials=1)
    objeto_de_estudo.trials_dataframe().to_csv('engenharia_de_atributos/model/tentativas.csv')

melhor_trial_rf = objeto_de_estudo.best_trial

print(f"Número do melhor trial: {melhor_trial_rf.number}")
print(f"Parâmetros do melhor trial: {melhor_trial_rf.params}")

# Uma instância do modelo baseline é criada.
modelo_rf_ot = cria_instancia_modelo(melhor_trial_rf)

modelo_rf_ot.fit(X_treino, y_treino)
y_previsao_rf_ot = modelo_rf_ot.predict(X_teste)

roc_auc_rf_ot = roc_auc_score(y_teste, y_previsao_rf_ot)
print("Métrica de ROC AUC para modelo otimizado:", roc_auc_rf_ot)

accuracy_rf_ot = accuracy_score(y_teste, y_previsao_rf_ot)
print("Métrica de acurácia para modelo otimizado:", accuracy_rf_ot)

joblib.dump(modelo_rf_ot, "engenharia_de_atributos/model/best_model_rf.pkl")
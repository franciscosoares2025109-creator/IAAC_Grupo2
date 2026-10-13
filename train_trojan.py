import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.data_preprocessing import prepare_pipeline_data

# 1. Preparar os dados
DATASET_PATH = "Trojan_Detection.csv"
X_train, X_test, y_train, y_test, feature_names, scaler, label_encoder = prepare_pipeline_data(DATASET_PATH)

# 2. Definir os modelos a comparar
models = {
    "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

best_model = None
best_f1 = 0.0
best_name = ""

# 3. Treinar e avaliar
print("=== Treino de Classificadores para Trojan Detection ===")
for name, model in models.items():
    print(f"\nA treinar {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"[{name}] Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

# 4. Guardar o melhor modelo e transformadores na pasta 'models/'
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, f"models/trojan_best_model_{best_name}.pkl", compress=3)
joblib.dump(scaler, "models/trojan_scaler.pkl")
joblib.dump(label_encoder, "models/trojan_label_encoder.pkl")

print(f"\nMelhor modelo ({best_name}) guardado com sucesso em 'models/'!")
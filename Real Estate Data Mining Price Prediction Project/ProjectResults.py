# Kütüphanelerin Yüklenmesi

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import warnings
# Gereksiz uyarılar çıktıda gözükmemesi için kullanıldı
warnings.filterwarnings("ignore")

# Metriklerin Hesaplanması


def evaluate_model(y_true, y_pred, y_proba, y_train, y_train_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Recall": round(recall_score(y_true, y_pred), 3),
        "F1 Score": round(f1_score(y_true, y_pred), 3),
        "ROC AUC": round(roc_auc_score(y_true, y_proba), 3),
        "Overfitting (Train - Test Acc)": round(accuracy_score(y_train, y_train_pred) - accuracy_score(y_true, y_pred), 5)
    }


# Modellerin Tanımlanması
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

# Tüm sütunların sonuçta gösterilmesi için kullanıldı
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Modellerin Temel Olarak Çalıştırılması
"""
Veri ön işleme sonrasında modeller ilk halleriyle çalıştırıldı.
"""

print("\n--- 1. Başlangıç: Tüm Modellerin Karşılaştırması ---")
df = pd.read_excel(r"C:/Users/kaanb/Desktop/log_price_eklenmis_veri.xlsx")
X = df.drop(columns=["price", "log_price"])
y = (df["log_price"] >= df["log_price"].median()).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results_base = []
for name, model in base_models.items():
    if name == "KNN":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_train_pred = model.predict(X_train_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_train_pred = model.predict(X_train)
    m = evaluate_model(y_test, y_pred, y_proba, y_train, y_train_pred)
    m["Model"] = name
    results_base.append(m)
results_base_df = pd.DataFrame(results_base).set_index("Model")
print(results_base_df[["Accuracy", "Recall", "F1 Score",
      "ROC AUC", "Overfitting (Train - Test Acc)"]])

# 2. GridSearchCV ile Hiperparametre Optimizasyonu
"""
Decision Tree algoritmasındaki overfittingi çözmek ve Random forest ile XGBoost algoritmalarındaki parametreleri optimize etmek için hiperparametre yaptık.
Diğer modellerin parametre sayılarının az olması bu yüzden sonucu önemli ölçüde etkilemeyeceği için hiperparametreye gerek görülmemiştir."""

print("\n--- 2. Hiperparametre Optimizasyonu (GridSearchCV) ---")
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
dt_params = {
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

rf_grid = GridSearchCV(RandomForestClassifier(
    random_state=42), rf_params, cv=5, scoring="f1", n_jobs=-1)
xgb_grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                        random_state=42), xgb_params, cv=5, scoring="f1", n_jobs=-1)
dt_grid = GridSearchCV(DecisionTreeClassifier(
    random_state=42), dt_params, cv=5, scoring="f1", n_jobs=-1)

rf_grid.fit(X_train, y_train)
xgb_grid.fit(X_train, y_train)
dt_grid.fit(X_train, y_train)

results_tuned = []
tuned_models = {
    "Random Forest (Tuned)": rf_grid.best_estimator_,
    "XGBoost (Tuned)": xgb_grid.best_estimator_,
    "Decision Tree (Tuned)": dt_grid.best_estimator_
}
for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_train_pred = model.predict(X_train)
    m = evaluate_model(y_test, y_pred, y_proba, y_train, y_train_pred)
    m["Model"] = name
    results_tuned.append(m)
results_tuned_df = pd.DataFrame(results_tuned).set_index("Model")
print(results_tuned_df[["Accuracy", "Recall", "F1 Score",
      "ROC AUC", "Overfitting (Train - Test Acc)"]])

# 3. Özellik Ekleme Sonrası Performans Karşılaştırması
"""
Model performanslarına önemli ölçüde etki edeceği düşünüldüğü için metrekare başına fiyat adında yeni bir özellik oluşturulmuştur. 
"""
print("\n--- 3. Özellik Mühendisliği (m2 başı fiyat) Sonrası Karşılaştırma ---")
df = pd.read_excel(
    r"C:\Users\musta\OneDrive\Masaüstü\proje\log_price_eklenmis_veri.xlsx")
df["price_per_sqm"] = df["price"] / df["netSqm"].replace(0, 1)
X = df.drop(columns=["price", "log_price"])
y = (df["log_price"] >= df["log_price"].median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results_feat = []
for name, model in base_models.items():
    if name == "KNN":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_train_pred = model.predict(X_train_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_train_pred = model.predict(X_train)
    m = evaluate_model(y_test, y_pred, y_proba, y_train, y_train_pred)
    m["Model"] = name
    results_feat.append(m)
results_feat_df = pd.DataFrame(results_feat).set_index("Model")
print(results_feat_df[["Accuracy", "Recall", "F1 Score",
      "ROC AUC", "Overfitting (Train - Test Acc)"]])

# ROC Eğrisi


def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        if name == "KNN":
            y_proba = model.predict_proba(scaler.transform(X_test))[:, 1]
        else:
            y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Eğrileri')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


plot_roc_curves(base_models, X_test, y_test)

# Confusion Matrix


def plot_all_confusion_matrices(models, X_test, y_test, scaler=None, class_names=["Düşük Fiyat", "Yüksek Fiyat"]):
    plt.figure(figsize=(18, 12))

    for i, (name, model) in enumerate(models.items()):
        plt.subplot(2, 3, i + 1)

        if name == "KNN" and scaler is not None:
            y_pred = model.predict(scaler.transform(X_test))
        else:
            y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Tahmin Edilen")
        plt.ylabel("Gerçek")

        accuracy = round(accuracy_score(y_test, y_pred), 3)
        precision = round(precision_score(y_test, y_pred), 3)
        recall = round(recall_score(y_test, y_pred), 3)
        f1 = round(f1_score(y_test, y_pred), 3)
        metrics_text = f"Acc: {accuracy}  Prec: {precision}  Rec: {recall}  F1: {f1}"
        plt.text(0.5, -0.3, metrics_text, fontsize=9,
                 ha='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()


plot_all_confusion_matrices(base_models, X_test, y_test, scaler=scaler)


# F1 Score Gelişimi

# Daha önce elde edilen F1 skorlarını kullanarak veri çerçevesi oluştur
f1_full_df = pd.DataFrame({
    "Base": results_base_df["F1 Score"],
    "Tuned": results_tuned_df["F1 Score"],
    "Feature Engineered": results_feat_df["F1 Score"]
})

# Hiperparametre uygulanmayan modeller için 'Tuned' kolonuna 'Base' değerlerini ata
for model in f1_full_df.index:
    if "Tuned" not in model:
        f1_full_df.loc[model, "Tuned"] = f1_full_df.loc[model, "Base"]

# Model adlarını sadeleştir
f1_full_df.index = f1_full_df.index.str.replace(" \(Tuned\)", "", regex=True)

# Aynı isimli modelleri gruplandırarak ortalama al
f1_mean_df = f1_full_df.groupby(f1_full_df.index).mean()

# Grafik çizimi
fig, ax = plt.subplots(figsize=(10, 6))
f1_mean_df.plot(kind="bar", ax=ax)
ax.set_title("Tüm Modellerin F1 Score Gelişimi", fontsize=14)
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1)
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 1. Model isimlerini sadeleştir (Tuned versiyonlar için)
results_tuned_df_clean = results_tuned_df.copy()
results_tuned_df_clean.index = results_tuned_df_clean.index.str.replace(
    r" \(Tuned\)", "", regex=True)

# 2. Ortak modelleri belirle
common_models = results_base_df.index.intersection(
    results_tuned_df_clean.index)

# 3. Overfitting değerlerini al
overfit_base = results_base_df.loc[common_models,
                                   "Overfitting (Train - Test Acc)"]
overfit_tuned = results_tuned_df_clean.loc[common_models,
                                           "Overfitting (Train - Test Acc)"]

# 4. Grafik için hazırlık
x = np.arange(len(common_models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars_base = ax.bar(x - width/2, overfit_base, width, label='Başlangıç Modeli')
bars_tuned = ax.bar(x + width/2, overfit_tuned, width,
                    label='Optimize Edilmiş Model')

# 5. Grafik ayarları
ax.set_ylabel('Overfitting (Train - Test Accuracy)')
ax.set_title('Model Başına Overfitting Değerlerinin Karşılaştırılması')
ax.set_xticks(x)
ax.set_xticklabels(common_models, rotation=45, ha='right')
ax.legend()
ax.grid(True, axis='y')

# 6. Barların üstüne değer yazdırma fonksiyonu


def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # yukarı doğru 3 birim boşluk
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(bars_base)
autolabel(bars_tuned)

plt.tight_layout()
plt.show()

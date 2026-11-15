import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_save():
    # 1. Load dataset Iris
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Split data untuk evaluasi singkat
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Latih model Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Akurasi Model pada Data Uji: {acc:.2%}")

    # 4. Latih ulang pada seluruh data untuk produksi & simpan dengan joblib
    model.fit(X, y)
    artifact = {
        "model": model,
        "target_names": list(iris.target_names),
        "feature_names": list(iris.feature_names),
    }
    joblib.dump(artifact, "model_iris.joblib")
    print("Model berhasil disimpan sebagai 'model_iris.joblib'")

if __name__ == "__main__":
    train_and_save()

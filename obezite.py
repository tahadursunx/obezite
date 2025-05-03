import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Veri setini yükle
file_path = "ObesityDataSet_raw_and_data_sinthetic (1).csv"
df = pd.read_csv(file_path)

# Kategorik sütunları seç
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Label Encoding (hedef değişken hariç)
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Özellikler (X) ve hedef değişken (y)
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğit
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Modeli kaydet
joblib.dump(rf_model, "random_forest_model.pkl")

# Label encoder'ları da kaydet (ön işleme için gerekli)
joblib.dump(label_encoders, "label_encoders.pkl")

# Doğruluğu hesapla ve yazdır (isteğe bağlı)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"Random Forest Doğruluğu: {rf_accuracy:.4f}")
print("Random Forest modeli ve Label Encoder'lar kaydedildi.")

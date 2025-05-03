import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np

# Model ve encoder'ları yükle
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Türkçe -> İngilizce dönüşümleri
turkish_to_english = {
    "Cinsiyet": {"Erkek": "Male", "Kadın": "Female"},
    "Ailede Obezite": {"Evet": "yes", "Hayır": "no"},
    "Yüksek Kalorili Yiyecek": {"Evet": "yes", "Hayır": "no"},
    "Atıştırma": {"Hayır": "no", "Bazen": "Sometimes", "Sık Sık": "Frequently", "Always": "Always"},
    "Sigara": {"Evet": "yes", "Hayır": "no"},
    "Kalori Takibi": {"Evet": "yes", "Hayır": "no"},
    "Kalorili İçecek": {"Hayır": "no", "Bazen": "Sometimes", "Sık Sık": "Frequently", "Always": "Always"},
    "Ulaşım": {
        "Toplu Taşıma": "Public_Transportation",
        "Yürüyüş": "Walking",
        "Otomobil": "Automobile",
        "Motosiklet": "Motorbike",
        "Bisiklet": "Bike"
    }
}

# Giriş alanları
entries = {}

# Arayüz başlat
root = tk.Tk()
root.title("Obezite Tahmin Uygulaması")
root.geometry("500x800") # Pencere yüksekliğini artırdık

# Etiket ve giriş oluşturma fonksiyonu (hizalama için geliştirildi)
def create_input(label_text, var_type="entry", options=None):
    frame = tk.Frame(root)
    frame.pack(pady=5, fill="x", padx=10) # fill="x" ile yatayda genişleme sağladık
    label = tk.Label(frame, text=label_text + ":", width=25, anchor="w") # Sabit genişlik ve sola hizalama
    label.pack(side="left")
    if var_type == "entry":
        ent = tk.Entry(frame)
        ent.pack(side="right", expand=True, fill="x") # Sağa hizalama ve genişleme
        entries[label_text] = ent
    elif var_type == "combo":
        combo = ttk.Combobox(frame, values=options, state="readonly")
        combo.pack(side="right", expand=True, fill="x") # Sağa hizalama ve genişleme
        entries[label_text] = combo

# Giriş alanlarını oluştur
create_input("Cinsiyet", "combo", ["Kadın", "Erkek"])
create_input("Yaş")
create_input("Boy (cm)")
create_input("Kilo (kg)")
create_input("Ailede Obezite", "combo", ["Evet", "Hayır"])
create_input("Yüksek Kalorili Yiyecek", "combo", ["Evet", "Hayır"])
create_input("Sebze Tüketimi (0-3 arası)")
create_input("Ana Öğün Sayısı")
create_input("Atıştırma", "combo", ["Hayır", "Bazen", "Sık Sık", "Her Zaman"])
create_input("Sigara", "combo", ["Evet", "Hayır"])
create_input("Su Tüketimi (Litre)")
create_input("Kalori Takibi", "combo", ["Evet", "Hayır"])
create_input("Fiziksel Aktivite (0-3 arası)")
create_input("Ekran Süresi (saat)")
create_input("Kalorili İçecek", "combo", ["Hayır", "Bazen", "Sık Sık", "Her Zaman"])
create_input("Ulaşım", "combo", ["Toplu Taşıma", "Yürüyüş", "Otomobil", "Motosiklet", "Bisiklet"])

# Sonuç etiketi
result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), fg="blue")
result_label.pack(pady=10)

# Tahmin fonksiyonu (sonucu ekranda gösterir)
def predict():
    try:
        input_data = []
        for label_text in entries:
            val = entries[label_text].get()
            if label_text in turkish_to_english:  # Kategorik
                val_eng = turkish_to_english[label_text][val]
                encoder_key = list(label_encoders.keys())[list(turkish_to_english.keys()).index(label_text)]
                val_encoded = label_encoders[encoder_key].transform([val_eng])[0]
                input_data.append(val_encoded)
            else:  # Sayısal
                input_data.append(float(val))

        input_array = np.array(input_data).reshape(1, -1)
        pred = model.predict(input_array)[0]
        tahmin = label_encoders["NObeyesdad"].inverse_transform([pred])[0]

        # BMI hesaplama
        boy_cm = float(entries["Boy (cm)"].get())
        kilo = float(entries["Kilo (kg)"].get())
        boy_m = boy_cm / 100
        bmi = round(kilo / (boy_m ** 2), 2)

        # Öneri
        if bmi < 18.5:
            advice = "Zayıf: Beslenme düzeninizi gözden geçirin, bir diyetisyene başvurun."
        elif 18.5 <= bmi < 25:
            advice = "Normal: Harika! Dengeli beslenmeye ve hareket etmeye devam edin."
        elif 25 <= bmi < 30:
            advice = "Fazla Kilolu: Fiziksel aktiviteyi artırın, porsiyon kontrolü yapın."
        elif 30 <= bmi < 35:
            advice = "Obez (Sınıf I): Sağlıklı alışkanlıklar edinin, düzenli egzersiz yapın."
        elif 35 <= bmi < 40:
            advice = "Obez (Sınıf II): Profesyonel destek almanız faydalı olabilir."
        else:
            advice = "Aşırı Obez (Sınıf III): Sağlık açısından risklidir, tıbbi destek alın."

        result_label.config(text=f"Tahmin Edilen Obezite Durumu:\n{tahmin}\n"
                                 f"Vücut Kitle İndeksiniz: {bmi}\n"
                                 f"Öneri: {advice}")
    except Exception as e:
        messagebox.showerror("Hata", f"Bir hata oluştu:\n{e}")
        result_label.config(text="Tahmin yapılamadı.")


# Buton
tk.Button(root, text="Tahmin Et", command=predict, bg="green", fg="white", font=("Arial", 12)).pack(pady=20)

root.mainloop()
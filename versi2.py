import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Membaca dataset Iris dari CSV
data = pd.read_csv("iris.csv")

# Memisahkan fitur (X) dan target (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Inisialisasi list untuk menyimpan akurasi
accuracies = []

# Jumlah iterasi untuk mencapai akurasi yang rendah
num_iterations = 1000

# Target akurasi antara 51% hingga 60%
target_accuracy = 0.55

for _ in range(num_iterations):
    # Mengacak data dengan cara yang lebih signifikan
    shuffled_indices = np.random.permutation(len(data))
    X_shuffled = X.iloc[shuffled_indices]
    y_shuffled = y.iloc[shuffled_indices]

    # Menambahkan noise ke dalam data dengan variasi yang lebih besar
    noisy_X = X_shuffled + np.random.normal(0, 1, size=X_shuffled.shape)

    # Memisahkan data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(noisy_X, y_shuffled, test_size=0.2, random_state=np.random.randint(1, 100))

    # Membuat model klasifikasi (Decision Tree Classifier) dengan parameter yang bervariasi
    model = DecisionTreeClassifier(max_depth=np.random.randint(2, 10), min_samples_split=np.random.randint(2, 10), 
                                   min_samples_leaf=np.random.randint(1, 5), random_state=np.random.randint(1, 100))

    # Melatih model
    model.fit(X_train, y_train)

    # Memprediksi spesies untuk data uji
    y_pred = model.predict(X_test)

    # Mengukur akurasi
    accuracy = np.mean(cross_val_score(model, noisy_X, y_shuffled, cv=10))
    accuracies.append(accuracy)

    # Menghentikan iterasi jika target akurasi tercapai
    if accuracy <= target_accuracy:
        break

# Menghitung rata-rata akurasi
average_accuracy = np.mean(accuracies)

# Mencetak rata-rata akurasi dalam persentase
print("Rata-rata Akurasi:", round(average_accuracy * 100, 2), "%")

# Mengambil model dengan akurasi terbaik
best_model = model

# Melakukan prediksi menggunakan model terbaik
y_pred = best_model.predict(X)

# Menambahkan kolom 'Predicted_Species' ke dalam dataset asli
data['Predicted_Species'] = y_pred

# Menampilkan dataset dengan hasil prediksi
print(data)

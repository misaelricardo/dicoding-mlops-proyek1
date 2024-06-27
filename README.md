# Submission 1: Machine Learning Pipeline - Red Wine Quality Predictor
Nama: Misael Bistok Ricardo

Username dicoding: misaelricardo

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) |
| Masalah | Prediksi kualitas wine merah berdasarkan sifat fisikokimianya. Kualitas wine merupakan faktor penting dalam industri wine, dan dapat dipengaruhi oleh berbagai variabel fisikokimia seperti kadar alkohol, keasaman, dan kandungan gula. Prediksi ini membantu produsen wine untuk menentukan kualitas produk mereka dan mengambil keputusan produksi yang lebih baik. |
| Solusi machine learning | Model machine learning yang dapat memprediksi kualitas wine berdasarkan dataset yang diberikan. Solusi ini mencakup preprocessing data, pelatihan model, dan evaluasi model untuk memastikan performa yang optimal. |
| Metode pengolahan | Splitting Data: Membagi dataset menjadi data training dan data testing. |
| Arsitektur model | Model neural network dengan arsitektur sebagai berikut: <br> - Input Layer: Menangani 11 fitur fisikokimia dari wine. <br> - Dense Layer 1: 128 unit dengan aktivasi ReLU dan dropout sebesar 20%. <br> - Dense Layer 2: 64 unit dengan aktivasi ReLU dan dropout sebesar 20%. <br> - Dense Layer 3: 64 unit dengan aktivasi ReLU dan dropout sebesar 20%. <br> - Output Layer: 1 unit untuk prediksi kualitas wine. |
| Metrik evaluasi | - Mean Absolute Error (MAE): Mengukur rata-rata kesalahan absolut antara nilai prediksi dan nilai aktual. <br> - Mean Squared Error (MSE): Mengukur rata-rata kesalahan kuadrat antara nilai prediksi dan nilai aktual. <br> - Root Mean Squared Error (RMSE): Mengukur akar dari rata-rata kesalahan kuadrat antara nilai prediksi dan nilai aktual. |
| Performa model | Model yang dibuat memiliki performa sebagai berikut: <br> - Mean Absolute Error (MAE): 0.491 <br> - Mean Squared Error (MSE): 0.389 <br> - Root Mean Squared Error (RMSE): 0.624 |

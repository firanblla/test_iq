from flask import Flask, render_template, request
import joblib
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__, static_folder='static')

# Load model yang sudah dilatih
model = joblib.load('./models/iq_prediction_model.pkl')

# Fungsi untuk menentukan keterangan berdasarkan Outcome
def interpret_iq(outcome):
    if outcome == 3:
        return "Di Atas Rata-Rata"
    elif outcome == 2:
        return "Rata-Rata"
    elif outcome == 1:
        return "Di Bawah Rata-Rata"
    else:
        return "Tidak Diketahui"

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil skor mentah dari form
        raw_score = int(request.form['raw_score'])

        # Lakukan prediksi
        predicted_iq = model.predict(np.array([[raw_score]]))[0]

        # Tentukan outcome berdasarkan rentang yang telah ditentukan
        if 138 >= predicted_iq >= 110:  # Di Atas Rata-Rata
            outcome = 3
        elif 108 >= predicted_iq >= 90:  # Rata-Rata
            outcome = 2
        elif 89 >= predicted_iq >= 56:  # Di Bawah Rata-Rata
            outcome = 1
        else:
            outcome = 0  # Jika di luar rentang yang diharapkan

        # Interpretasi hasil
        keterangan = interpret_iq(outcome)

        return render_template('result.html', raw_score=raw_score, predicted_iq=predicted_iq, keterangan=keterangan)
    except ValueError:
        return "Invalid input. Please enter a numeric value."

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
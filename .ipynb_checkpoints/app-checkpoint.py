from flask import Flask, request, render_template, jsonify, url_for
import pandas as pd
import numpy as np
import joblib
import os
import locale

# Memuat model dan scaler yang telah disimpan
rf_reg = joblib.load(r'D:\Prediksi\Python\Random_Forest_Regressor\web\model\random_forest_model.pkl')
scaler = joblib.load(r'D:\Prediksi\Python\Random_Forest_Regressor\web\model\scaler.pkl')
columns = joblib.load(r'D:\Prediksi\Python/Random_Forest_Regressor\web\model\columns.pkl')

# Set locale untuk format mata uang Indonesia
locale.setlocale(locale.LC_ALL, 'id_ID.UTF-8')
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Halaman untuk memperbarui model
@app.route('/pembaruan', methods=['GET', 'POST'])
def pembaruan():
    if request.method == 'POST':
        model_file = request.files['model']
        scaler_file = request.files['scaler']
        columns_file = request.files['columns']

        # Memastikan ekstensi file yang diupload sesuai
        if model_file and allowed_file(model_file.filename):
            model_file.save(os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
        if scaler_file and allowed_file(scaler_file.filename):
            scaler_file.save(os.path.join(MODEL_DIR, 'scaler.pkl'))
        if columns_file and allowed_file(columns_file.filename):
            columns_file.save(os.path.join(MODEL_DIR, 'columns.pkl'))

        # Memuat model yang diperbarui
        global rf_reg, scaler, columns
        try:
            rf_reg = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
            scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
            columns = joblib.load(os.path.join(MODEL_DIR, 'columns.pkl'))
            print("Model, Scaler, and Columns updated successfully.")
        except Exception as e:
            print(f"Error updating model or related files: {e}")
        
        return redirect(url_for('pembaruan', message="Model berhasil diperbarui"))

    return render_template('pembaruan.html')


# API endpoint untuk prediksi harga rumah
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil data dari form input
        kt = float(request.form['KT'])
        km = float(request.form['KM'])
        lb = float(request.form['LB'])
        lt = float(request.form['LT'])
        usia = float(request.form['Usia'])
        lokasi = request.form['Lokasi']
        fasilitas = request.form.getlist('Fasilitas')

        # Membuat DataFrame input untuk prediksi
        input_data = {
            'KT': [kt],
            'KM': [km],
            'LB': [lb],
            'LT': [lt],
            'Usia': [usia],
            'Lokasi': [lokasi],  
            'Fasilitas': [', '.join(fasilitas)]  
        }
        
        df_input = pd.DataFrame(input_data)

        # Lakukan One-Hot Encoding untuk kolom kategorikal
        df_input_encoded = pd.get_dummies(df_input, drop_first=True)

        # Pastikan input memiliki kolom yang sama dengan data pelatihan (dengan One-Hot Encoding yang sama)
        df_input_encoded = df_input_encoded.reindex(columns=columns, fill_value=0)

        # Melakukan scaling pada fitur input
        input_scaled = scaler.transform(df_input_encoded)

        # Prediksi menggunakan model Random Forest
        prediction = rf_reg.predict(input_scaled)

        # Menggunakan np.exp untuk kembali ke harga asli (skala log)
        predicted_price = np.exp(prediction[0])

        # Format harga dalam mata uang IDR
        formatted_price = locale.currency(predicted_price, grouping=True)

        # Mengarahkan ke halaman prediksi untuk menampilkan hasil
        return render_template('prediksi.html', prediction=formatted_price)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Datos recibidos:", data)  # Depuración

        mag = float(data['mag'])
        depth = float(data['depth'])
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])

        features = np.array([[mag, depth, longitude, latitude]])
        probability = model.predict_proba(features)[0][1]
        prediction = 1 if probability >= 0.5 else 0

        print("Predicción:", prediction, "Probabilidad:", probability)  # Depuración

        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability)
        })
    except Exception as e:
        print(f"Error en la predicción: {e}")  # Depuración
        return jsonify({'error': str(e)}), 500

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Cargar el modelo, el encoder y el scaler
model = load_model('/code/house_price_model.h5')
encoder = joblib.load('/code/encoder.pkl')
scaler = joblib.load('/code/scaler.pkl')

app = Flask(__name__)
CORS(app)  # Permite todas las solicitudes de cualquier origen

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar el formato de entrada
    input_data = request.get_json()

    try:
        # Extraer datos
        categorical_data = np.array([input_data[col] for col in ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']]).reshape(1, -1)
        numeric_data = np.array([input_data[col] for col in ['MSSubClass', 'LotArea', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF2', 'TotalBsmtSF']]).reshape(1, -1)
        
        # Transformar datos
        categorical_transformed = encoder.transform(categorical_data)
        numeric_transformed = scaler.transform(numeric_data)
        
        # Concatenar
        final_input = np.hstack((categorical_transformed, numeric_transformed))
        
        # Realizar predicción
        prediction = model.predict(final_input)
        return jsonify({'predicted_price': float(prediction[0, 0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

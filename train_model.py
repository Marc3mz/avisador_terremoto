import requests
import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Cargar variables de entorno
EMAIL_ADDRESS = os.environ.get('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')

# URL de datos sísmicos
url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"

# Descargar CSV solo si no existe
if not os.path.exists("earthquakes.csv"):
    response = requests.get(url)
    with open("earthquakes.csv", "wb") as file:
        file.write(response.content)

# Cargar datos
df = pd.read_csv("earthquakes.csv")


# 🔹 Convertir la columna 'time' a formato datetime
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.dropna(subset=['time'])  # Eliminar filas con fechas inválidas
df = df.sort_values('time')  # Ordenar por fecha

# 🔹 Filtrar datos para América del Sur
south_america = df[(df['latitude'] >= -60) & (df['latitude'] <= 15) & 
                   (df['longitude'] >= -90) & (df['longitude'] <= -30)]

# 🔹 Función para calcular la densidad sísmica
def calcular_densidad_sismica(df, radio=1.0):
    coords = np.radians(df[['latitude', 'longitude']].values)
    kde = KernelDensity(kernel='gaussian', bandwidth=radio)
    kde.fit(coords)
    log_densidad = kde.score_samples(coords)
    df['densidad_sismica'] = np.exp(log_densidad)
    return df

# 🔹 Función para crear la variable objetivo (predicción de futuros sismos)
def crear_variable_objetivo(df, magnitud_umbral=5.0, dias_ventana=7):
    df['target'] = 0  # Inicializar la columna de etiquetas
    for i in range(len(df)):
        fecha_actual = df.iloc[i]['time']
        futuros = df[(df['time'] > fecha_actual) & 
                     (df['time'] <= fecha_actual + pd.Timedelta(days=dias_ventana)) & 
                     (df['mag'] >= magnitud_umbral)]
        if not futuros.empty:
            df.at[i, 'target'] = 1  # Marcar como 1 si hay futuros terremotos
            # Enviar alerta por correo
            if EMAIL_ADDRESS and EMAIL_PASSWORD:
                enviar_correo(
                    "Alerta de Terremoto",
                    f"Se ha detectado un terremoto significativo el {fecha_actual} con magnitud {df.iloc[i]['mag']}.",
                    EMAIL_ADDRESS,
                    EMAIL_ADDRESS,
                    EMAIL_PASSWORD
                )
    return df

# 🔹 Función para enviar alertas por correo electrónico
def enviar_correo(asunto, mensaje, destinatario, remitente, contraseña):
    servidor_smtp = 'smtp.gmail.com'
    puerto = 587
    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = asunto
    msg.attach(MIMEText(mensaje, 'plain'))
    try:
        server = smtplib.SMTP(servidor_smtp, puerto)
        server.starttls()
        server.login(remitente, contraseña)
        text = msg.as_string()
        server.sendmail(remitente, destinatario, text)
        server.quit()
        print("Correo enviado correctamente")
    except Exception as e:
        print(f"Error al enviar el correo: {e}")

# 🔹 Aplicar funciones de transformación de datos
df = calcular_densidad_sismica(df)
df = crear_variable_objetivo(df)

# 🔹 Preparar datos para el modelo
features = ['latitude', 'longitude', 'mag', 'depth', 'densidad_sismica']
X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Entrenar modelo XGBoost
modelo = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
modelo.fit(X_train, y_train)

# 🔹 Evaluar modelo
y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# 🔹 Guardar el modelo entrenado
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(modelo, os.path.join(model_dir, 'earthquake_model.joblib'))
print("Modelo guardado exitosamente en 'model/earthquake_model.joblib'.")
   

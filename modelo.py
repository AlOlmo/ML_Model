import pandas as pd
from pyspark.sql import SparkSession
import joblib #Exportar

# Crear una sesión de Spark (si no está ya creada)
spark = SparkSession.builder.appName("DiabetesModel").getOrCreate()

# Ajustar el número máximo de columnas que se muestran
pd.options.display.max_columns = None

# Ajustar el ancho máximo de cada columna
pd.options.display.max_colwidth = None

# Ajustar el número máximo de filas (opcional, si lo necesitas)
pd.options.display.max_rows = None


#Lectura con spark
df = spark.read.csv("diabetes.csv", header=True, inferSchema=True)

data = df.toPandas()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

columnas_con_ceros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
porcentaje_ceros = (data[columnas_con_ceros] == 0).sum() / len(data) * 100

# 1. Cargar los datos (ya cargados previamente con Spark y convertido a Pandas)
# Asegúrate de manejar cualquier valor nulo o cero si es necesario

# 2. Reemplazar ceros con NaN en columnas específicas (si los ceros son erróneos)
# Usar .applymap() para reemplazar ceros por NaN de manera más segura
data[columnas_con_ceros] = data[columnas_con_ceros].applymap(lambda x: pd.NA if x == 0 else x)

# 3. Imputar los valores nulos con la media de cada columna
data[columnas_con_ceros] = data[columnas_con_ceros].fillna(data[columnas_con_ceros].mean())

# 4. Seleccionar las características (features) y la variable objetivo (target)
X = data[columnas_con_ceros]  # Características
y = data['Outcome']  # Variable objetivo (0 o 1, diabetes o no)

# 5. Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Escalar los datos (importante para algunos modelos, como la regresión logística)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#### Paso 2: Crear y entrenar el modelo

# 7. Crear el modelo de regresión logística
model = LogisticRegression()

# 8. Entrenar el modelo
model.fit(X_train, y_train)

#### Paso 3: Evaluar el modelo

# 9. Hacer predicciones con el conjunto de prueba
y_pred = model.predict(X_test)

# 10. Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# 11. Mostrar el reporte de clasificación
print(classification_report(y_test, y_pred))

# 12. Matriz de confusión
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print("Coeficientes:", model.coef_)
print("Intercepto:", model.intercept_)

#USAR:
#y_pred = model.predict(X_test)  # Predice etiquetas
#y_prob = model.predict_proba(X_test)  # Predice probabilidades

#GUARDAR Y REUSAR:

# Guardar el modelo entrenado
joblib.dump(model, 'modelo_entrenado.pkl')

# Cargar el modelo entrenado
#modelo_cargado = joblib.load('modelo_entrenado.pkl')

# Usar el modelo cargado para predicciones
#predicciones = modelo_cargado.predict(X_test)



# 📈 Predicción de precios BBVA y Santander con redes neuronales recurrentes

https://github.com/josean9/Caso02_Prediccion_BBVA_SANTANDER.git

Este proyecto desarrolla un modelo predictivo basado en **redes neuronales recurrentes (RNN, GRU y LSTM)** para estimar la evolución del precio de las acciones de **BBVA** y **Santander**, combinando información de análisis técnico y variables macroeconómicas relevantes.

---

## 🧠 Descripción general

El objetivo del proyecto es analizar y predecir la evolución del precio de acciones bancarias mediante técnicas de aprendizaje profundo, integrando tanto datos financieros históricos como indicadores económicos que influyen en el mercado bursátil.

Se parte de los datos obtenidos desde *E-Finance*, incluyendo precios diarios de apertura, cierre, máximos, mínimos y volumen de negociación.  
Posteriormente se añadieron indicadores derivados como las **medias móviles de 7, 30 y 60 días**, junto con variables macroeconómicas: **inflación** y **tipos de interés del BCE** (depósito, marginal y refinanciación).

Estas variables combinan la visión técnica del comportamiento del precio con el contexto económico general, proporcionando una base sólida para el entrenamiento de modelos de predicción.

---

## ⚙️ Procesamiento y entrenamiento

1. **Preprocesamiento de datos**
   - Interpolación de valores faltantes y limpieza de outliers.  
   - Escalado de variables mediante **MinMaxScaler**.  
   - Normalización de variables macroeconómicas dividiendo entre 10 para mantener coherencia numérica.  
   - Generación de secuencias temporales de 60 días como entrada a los modelos.

2. **Modelos implementados**
   - **RNN simple:** limitada en memoria temporal, adecuada solo para relaciones de corto plazo.  
   - **GRU:** arquitectura más eficiente y ligera, buena capacidad predictiva en corto plazo.  
   - **LSTM:** modelo con mejor rendimiento, capaz de retener información a largo plazo.

3. **Parámetros comunes**
   - 2 capas recurrentes de **128 neuronas**.  
   - **Dropout:** 0.2  
   - **Optimizador:** Adam  
   - **Función de pérdida:** MSE  
   - **Early stopping:** para evitar sobreentrenamiento.

---

## 📊 Resultados y comparación

Los tres modelos fueron evaluados mediante las métricas **MSE**, **MAE**, **RMSE** y **R²**.

| Modelo | MSE | MAE | R² |
|:-------|:----|:----|:----|
| LSTM | 0.0018 | 0.0336 | **0.995** |
| GRU  | 0.0021 | 0.0350 | 0.993 |
| RNN  | 0.0450 | 0.1200 | 0.970 |

Los resultados muestran que el **LSTM** obtiene el mejor rendimiento, con alta precisión tanto en los datos históricos como en las proyecciones futuras.  
El **GRU** presenta un comportamiento similar, pero con menor capacidad de memoria, mientras que el **RNN** tiende a perder información con el paso del tiempo.

---

## 💻 Aplicación interactiva

Se desarrolló una **aplicación web en Streamlit** que integra todo el flujo de predicción de precios, ofreciendo una interfaz intuitiva para el usuario.

### Funcionalidades principales
- Selección del banco (**BBVA o Santander**).  
- Visualización del histórico de precios y medias móviles.  
- Predicción futura entre **1 y 60 días hábiles**.  
- Ajuste del escenario macroeconómico:
  - Mantener último valor (*ffill*).  
  - Escenario manual (ajuste directo de inflación y tipos BCE).  
  - Deriva simple (tendencia gradual diaria).  
- Simulación de **compra y venta de acciones** con control dinámico del capital invertido.  
- Gráficas interactivas con colores corporativos y desplazamiento lateral.

---

## 🚀 Ejecución de la aplicación

Para ejecutar la aplicación, utiliza el siguiente comando desde la raíz del proyecto:

```bash
streamlit run app/bbva_santander_app.py

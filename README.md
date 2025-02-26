# 💳 Predicción de Incumplimiento en Pagos de Tarjetas de Crédito

## 📌 Descripción del Proyecto

Este proyecto utiliza Machine Learning para predecir la probabilidad de incumplimiento en pagos de tarjetas de crédito, utilizando un conjunto de datos de [Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset). Ahora bien, se implementaron modelos de clasificación avanzados junto con técnicas de preprocesamiento de datos y análisis de importancia de variables para mejorar la precisión en la detección de clientes con mayor riesgo de impago.

En si, el enfoque principal es optimizar métricas como Recall Weighted y AUC-ROC para minimizar falsos negativos, ya que en el contexto financiero es crucial identificar correctamente a los clientes que pueden incumplir su pago.

---

## 🎯 Objetivo del Proyecto

El objetivo principal es desarrollar un modelo de clasificación robusto para predecir incumplimientos en pagos de tarjetas de crédito, utilizando un enfoque end-to-end, que incluye:

- 🔍 **Análisis Exploratorio de Datos (EDA)** para entender la distribución y correlación entre variables.
- 🛠 **Preprocesamiento de datos**, incluyendo manejo de valores atípicos y datos faltantes.
- 📊 **Evaluación de la importancia de variables** para determinar los principales factores de incumplimiento.
- 🏆 **Entrenamiento y comparación de modelos de clasificación**.
- 🔎 **Análisis de métricas avanzadas**, incluyendo **Matriz de Confusión, AUC-ROC y Log Loss**

## 📊 Descripción del Dataset

🔗 **Fuente**: [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

📌 **Descripción:**
 
Este conjunto de datos contiene información sobre pagos incumplidos, factores demográficos, datos crediticios, historial de pagos y estados de cuenta de clientes de tarjetas de crédito en Taiwán, recopilados entre abril y septiembre de 2005.

**Variables del dataset**

El dataset consta de 30,000 observaciones, con 23 variables predictoras y una variable objetivo (`default.payment.next.month`).

| **Columna**                     | **Descripción** |
|---------------------------------|---------------|
| **ID**                          | ID of each client. |
| **LIMIT_BAL**                   | Amount of given credit in NT dollars (includes individual and family/supplementary credit). |
| **SEX**                          | Gender (1=male, 2=female). |
| **EDUCATION**                    | (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown). |
| **MARRIAGE**                     | Marital status (1=married, 2=single, 3=others). |
| **AGE**                          | Age in years. |
| **PAY_0**                        | Repayment status in September 2005 (-1=pay duly, 1=payment delay for one month, ..., 9=payment delay for nine months and above). |
| **PAY_2**                        | Repayment status in August 2005 (same scale as above). |
| **PAY_3**                        | Repayment status in July 2005 (same scale as above). |
| **PAY_4**                        | Repayment status in June 2005 (same scale as above). |
| **PAY_5**                        | Repayment status in May 2005 (same scale as above). |
| **PAY_6**                        | Repayment status in April 2005 (same scale as above). |
| **BILL_AMT1**                    | Amount of bill statement in September 2005 (NT dollar). |
| **BILL_AMT2**                    | Amount of bill statement in August 2005 (NT dollar). |
| **BILL_AMT3**                    | Amount of bill statement in July 2005 (NT dollar). |
| **BILL_AMT4**                    | Amount of bill statement in June 2005 (NT dollar). |
| **BILL_AMT5**                    | Amount of bill statement in May 2005 (NT dollar). |
| **BILL_AMT6**                    | Amount of bill statement in April 2005 (NT dollar). |
| **PAY_AMT1**                     | Amount of previous payment in September 2005 (NT dollar). |
| **PAY_AMT2**                     | Amount of previous payment in August 2005 (NT dollar). |
| **PAY_AMT3**                     | Amount of previous payment in July 2005 (NT dollar). |
| **PAY_AMT4**                     | Amount of previous payment in June 2005 (NT dollar). |
| **PAY_AMT5**                     | Amount of previous payment in May 2005 (NT dollar). |
| **PAY_AMT6**                     | Amount of previous payment in April 2005 (NT dollar). |
| **default.payment.next.month**    | Default payment (1=yes, 0=no). |

### 📌 Variables predictoras:

#### 🏦 **Factores Demográficos:**
- **`SEX`** : Género (1=Masculino, 2=Femenino).
- **`EDUCATION`** : Nivel educativo (1=Graduado, 2=Universidad, etc.).
- **`MARRIAGE`** : Estado civil (1=Casado, 2=Soltero, 3=Otros).
- **`AGE`** : Edad en años.

#### 💳 **Historial de Crédito y Estado de Pago:**
- **`LIMIT_BAL`** : Monto del crédito otorgado (en NT dólares).
- **`PAY_0`** : Estado de pago en septiembre 2005 (-1=Sin retraso, 1=Retraso de 1 mes, etc.).
- **`PAY_2`** : Estado de pago en agosto 2005.
- **`PAY_3`** : Estado de pago en julio 2005.
- **`PAY_4`** : Estado de pago en junio 2005.
- **`PAY_5`** : Estado de pago en mayo 2005.
- **`PAY_6`** : Estado de pago en abril 2005.

#### 💰 **Montos Facturados y Pagos Realizados:**
- **`BILL_AMT1`** : Monto de la factura en septiembre 2005.
- **`BILL_AMT2`** : Monto de la factura en agosto 2005.
- **`BILL_AMT3`** : Monto de la factura en julio 2005.
- **`BILL_AMT4`** : Monto de la factura en junio 2005.
- **`BILL_AMT5`** : Monto de la factura en mayo 2005.
- **`BILL_AMT6`** : Monto de la factura en abril 2005.
- **`PAY_AMT1`** : Monto del pago realizado en septiembre 2005.
- **`PAY_AMT2`** : Monto del pago realizado en agosto 2005.
- **`PAY_AMT3`** : Monto del pago realizado en julio 2005.
- **`PAY_AMT4`** : Monto del pago realizado en junio 2005.
- **`PAY_AMT5`** : Monto del pago realizado en mayo 2005.
- **`PAY_AMT6`** : Monto del pago realizado en abril 2005.

### 🎯 **Variable Objetivo**
- **`default.payment.next.month`** : Indicador binario (0 = No incumplimiento, 1 = Incumplimiento de pago).


##  Modelos de Clasificación Utilizados

Se evaluaron y compararon los siguientes algoritmos de clasificación para predecir el incumplimiento de pago:

- **Regresión Logística** (Ridge y Lasso)
- **SVM** (Support Vector Machine)
- **Árbol de Decisión** (Decision Tree)
- **Random Forest**
- **XGBoost**
- **K-Nearest Neighbors** (KNN)

📌 Estos modelos fueron seleccionados por su eficacia en problemas de clasificación binaria y su capacidad para manejar datos financieros con características tanto categóricas como numéricas.

## Metodología



## Resultados y Análisis


## Conclusiones y Recomendaciones


## Requisitos de Instalación


## Licencia

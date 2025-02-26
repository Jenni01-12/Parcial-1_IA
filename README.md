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

## 📂 Dataset  

- **Fuente**: [Kaggle - Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)  

- **Descripción**:
- 
Este conjunto de datos contiene información sobre pagos incumplidos, factores demográficos, datos crediticios, historial de pagos y estados de cuenta de clientes de tarjetas de crédito en Taiwán, recopilados entre abril y septiembre de 2005.

**Variables del dataset**

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




## Modelos de Clasificación Utilizados


## Metodología



## Resultados y Análisis


## Conclusiones y Recomendaciones


## Requisitos de Instalación


## Licencia

# Análisis de Préstamos Hipotecarios con Machine Learning — HMDA Utah 2022

## 📋 Descripción del Proyecto

Análisis completo de **226,790 solicitudes de préstamos hipotecarios** del estado de Utah (año 2022) utilizando datos del **Home Mortgage Disclosure Act (HMDA)**. El proyecto combina técnicas de Machine Learning supervisado y no supervisado para predecir la aprobación de préstamos, identificar variables clave, detectar sesgos demográficos y segmentar el mercado hipotecario.

## 🎯 Objetivos

1. **Clasificación supervisada**: Predecir la aprobación/denegación de solicitudes hipotecarias usando Regresión Logística, Random Forest y Gradient Boosting.
2. **Interpretabilidad**: Identificar las variables más influyentes mediante SHAP y Feature Importance.
3. **Análisis ético**: Evaluar equidad del modelo por raza, sexo y edad.
4. **Segmentación no supervisada**: Descubrir perfiles de solicitantes con K-Means, DBSCAN y Clustering Jerárquico.

## 📊 Resultados Principales

### Rendimiento de Modelos

| Modelo | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| Regresión Logística | ~90% | 88.67% | ~0.94 |
| Random Forest | ~96% | 96.80% | ~0.99 |
| **Gradient Boosting** 🏆 | **~96%** | **96.67%** | **~0.99** |

**Mejor modelo**: Gradient Boosting, por su equilibrio óptimo entre precision (95.49%) y recall (97.87%).

### Variables Más Influyentes

1. **`debt_to_income_ratio`** — Factor más crítico; rango óptimo: 30-42%
2. **`interest_rate`** — Tasas bajas se asocian con aprobación
3. **`property_value`** — Mayor valor de propiedad aumenta la probabilidad
4. **`income`** — Relación directa con la aprobación
5. **Completitud de datos** — Información faltante reduce significativamente la aprobación

### Análisis Ético

- **Raza**: Brecha de ~16 pp entre White (~65%) y grupos minoritarios (~48-52%), pero las métricas del modelo son consistentes entre grupos (< 3.5 pp de diferencia en F1).
- **Sexo**: Equidad confirmada (diferencia Male-Female de ~1 pp).
- **Edad**: Sin evidencia de discriminación entre grupos conocidos.

### Segmentación del Mercado

Se identificaron **4 perfiles** de solicitantes mediante K-Means (K=4):

| Cluster | Perfil | % Mercado | Tasa Aprob. |
|---------|--------|-----------|-------------|
| 0 | Propiedades Premium, LTV Bajo | 8.5% | 92.9% |
| 1 | Hipoteca Estándar 30 Años | 55.1% | 48.5% |
| 2 | Largo Plazo, Tasa Elevada | 16.2% | 82.7% |
| 3 | Hipoteca Modesta, Corto Plazo | 20.2% | 51.8% |

## 🗂️ Estructura del Proyecto

```
hmda-loan-approval-ml-analysis/
├── notebook.ipynb              # Notebook principal con todo el análisis
├── data/
│   └── utah_hmda_data_2022.csv # Dataset HMDA Utah 2022
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Este archivo
└── LICENSE                     # Licencia del proyecto
```

## 📓 Contenido del Notebook

| Sección | Contenido |
|---------|-----------|
| **1-3** | Introducción, descripción del dataset y distribución del trabajo |
| **4** | Carga de datos y análisis exploratorio (EDA) |
| **5** | Preprocesamiento: limpieza, codificación, pipelines, manejo de desbalanceo |
| **6** | Entrenamiento de modelos: Regresión Logística, Random Forest, Gradient Boosting |
| **7** | Evaluación: métricas, matrices de confusión, curvas ROC y Precision-Recall |
| **8** | Interpretabilidad: SHAP (TreeExplainer), Feature Importance (RF y GB) |
| **9** | Análisis ético: tasas de aprobación y rendimiento del modelo por subgrupos |
| **10** | Clustering: K-Means, DBSCAN, Jerárquico, perfilado y análisis demográfico |
| **11** | Conclusiones generales y recomendaciones |

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **pandas** / **numpy** — Manipulación y análisis de datos
- **matplotlib** / **seaborn** — Visualización
- **scikit-learn** — Modelos de ML, pipelines, métricas, clustering
- **SHAP** — Interpretabilidad de modelos (TreeExplainer)
- **scipy** — Clustering jerárquico (dendrogramas)

## 🚀 Ejecución

### Requisitos previos

```bash
pip install -r requirements.txt
```

### Ejecución del notebook

```bash
jupyter notebook notebook.ipynb
```

O abrir directamente en **VS Code** con la extensión de Jupyter.

## 📈 Dataset

- **Fuente**: [Consumer Financial Protection Bureau (CFPB)](https://ffiec.cfpb.gov/data-browser/)
- **Alcance**: Estado de Utah, año 2022
- **Registros**: ~226,790 solicitudes de préstamos hipotecarios
- **Variables**: Características del préstamo (monto, tipo, tasa), del solicitante (ingresos, raza, sexo, edad) y resultado de la solicitud

## 👥 Autores

| Nombre | Rol |
|--------|-----|
| **Robert** | Limpieza de datos y preprocesamiento (Secciones 4-5) |
| **Rodrigo** | Modelado e interpretabilidad (Secciones 6-8) |
| **Juan** | Análisis ético y detección de sesgos (Sección 9) |
| **Conjunto** | Clustering y segmentación (Sección 10) |

## 📄 Licencia

Este proyecto está bajo la licencia especificada en el archivo [LICENSE](LICENSE).

# 📰 Noticias peruanas: del titular al clasificador

## Descripción

Este proyecto tiene como objetivo recolectar, analizar y clasificar noticias peruanas a partir de sus titulares.

Se trabajó con noticias reales del diario **El Comercio**, organizadas en cuatro categorías:

- politica
- economia
- deportes
- ciencia_tecnologia

El proyecto incluye recolección de datos, clasificación con modelos de lenguaje, fine-tuning de DistilBERT y un dashboard interactivo desarrollado en Streamlit.

---

## Fuente de datos

Se utilizaron noticias del diario **El Comercio**, obtenidas mediante RSS y páginas web del medio.

El dataset final contiene:

- 360 noticias
- 4 categorías
- Titular
- Bajada
- Medio
- Sección original
- URL
- Fecha
- Etiqueta

---

## Modelos utilizados

Se evaluaron tres enfoques de clasificación:

| Modelo | Accuracy | F1 Macro |
|---|---:|---:|
| Zero-shot | 65.00% | 63.28% |
| Ollama Qwen 2.5 3B | 73.33% | 73.10% |
| DistilBERT fine-tuned | 64.81% | 66.56% |

El mejor desempeño fue obtenido por **Ollama Qwen 2.5 3B**, con un Accuracy de 73.33% y un F1 Macro de 73.10%.

---

## Clasificación Zero-shot

Se realizó una evaluación inicial utilizando un modelo Zero-shot sobre una muestra de 40 titulares.

Resultados:

- Accuracy: 65.00%
- F1 Macro: 63.28%

Este enfoque permitió establecer una línea base para comparar los demás métodos de clasificación.

---

## Clasificación con LLM

Se utilizó **Ollama** con el modelo:

`qwen2.5:3b`

El modelo generó etiquetas para las noticias utilizando las cuatro categorías definidas en el proyecto.

Además, se realizó una revisión manual de una muestra de 20 noticias, obteniéndose un nivel de acuerdo del 90%.

Resultados generales:

- Accuracy: 73.33%
- F1 Macro: 73.10%

---

## Fine-tuning de DistilBERT

Se utilizó el modelo:

`distilbert-base-multilingual-cased`

La división de datos fue:

- 70% entrenamiento
- 15% validación
- 15% prueba

Se utilizó `random_state=42` para mantener la reproducibilidad.

El modelo se entrenó durante 3 épocas.

### Resultados de validación

| Época | Accuracy | F1 Macro |
|---:|---:|---:|
| 1 | 50.00% | 34.37% |
| 2 | 66.67% | 62.15% |
| 3 | 77.78% | 77.35% |

### Resultados finales en test

- Accuracy: 64.81%
- F1 Macro: 66.56%

La matriz de confusión mostró un mejor desempeño en las categorías de política y deportes, mientras que economía y ciencia y tecnología presentaron una mayor cantidad de confusiones.

---

## Comparación de modelos

Los resultados finales muestran que **Ollama Qwen 2.5 3B** obtuvo el mejor desempeño general.

DistilBERT obtuvo un Accuracy similar al enfoque Zero-shot, pero logró un F1 Macro superior.

Esto indica que el fine-tuning permitió mejorar el equilibrio de clasificación entre las categorías, aunque no logró superar al modelo LLM.

---

## Dashboard

El proyecto incluye un dashboard interactivo desarrollado con **Streamlit**.

El dashboard permite:

- Visualizar el total de noticias recolectadas.
- Mostrar el medio utilizado.
- Visualizar la cantidad de categorías.
- Mostrar el F1 Macro del modelo final.
- Visualizar la distribución de noticias por categoría.
- Comparar el rendimiento de los tres modelos.
- Filtrar noticias por categoría.
- Consultar la tabla completa de noticias.
- Escribir un titular nuevo.
- Obtener una categoría predicha.
- Visualizar el nivel de confianza de la predicción.

---

## Clasificador de titulares

El dashboard permite escribir un titular nuevo y clasificarlo automáticamente en una de las siguientes categorías:

- Política
- Economía
- Deportes
- Ciencia y tecnología

La aplicación utiliza el modelo DistilBERT entrenado y muestra también el porcentaje de confianza de la predicción.

Ejemplo:

```text
Selección peruana se prepara para su próximo partido internacional

Resultado esperado:
Categoría predicha: Deportes
---

## Estructura del proyecto


---
tarea_1/
│
├── 01_recoleccion.ipynb
├── 02_clasificacion.ipynb
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── datos/
│   ├── noticias.csv
│   └── etiquetas_llm.csv
│
└── modelo/
    ├── config.json
    ├── tokenizer.json
    └── tokenizer_config.json

    Por motivos de tamaño, los siguientes archivos están excluidos mediante .gitignore:

    modelo/*.safetensors
modelo/*.bin
modelo/checkpoints/
modelo/checkpoint-*/
modelo/**/checkpoint-*/
El archivo principal de pesos generado durante el entrenamiento es:
modelo/model.safetensors

## Nota sobre el modelo entrenado
Los archivos pesados del modelo, como model.safetensors y los checkpoints de entrenamiento, no se incluyen en GitHub para evitar subir archivos de gran tamaño.
El dashboard puede mostrar correctamente los datos, gráficos, métricas y comparación de modelos aunque los pesos no estén disponibles.
Para utilizar la función de clasificación automática de titulares se requiere contar localmente con:

modelo/model.safetensors

## Cómo ejecutar el dashboard
Opción 1: desde la carpeta raíz del repositorio

streamlit run tarea_1/app.py

Instalación de dependencias
Desde la carpeta raíz del repositorio:

pip install -r tarea_1/requirements.txt

Si ya se encuentra dentro de la carpeta tarea_1:

pip install -r requirements.txt

Tecnologías utilizadas
- Python
- Pandas
- NumPy
- Requests
- BeautifulSoup
- Plotly
- Scikit-learn
- Hugging Face Transformers
- PyTorch
- Datasets
- Ollama
- Qwen 2.5 3B
- DistilBERT
- Streamlit

Conclusiones
El proyecto permitió aplicar diferentes técnicas de clasificación de texto sobre noticias peruanas reales.
El modelo Ollama Qwen 2.5 3B obtuvo el mejor desempeño, alcanzando un Accuracy de 73.33% y un F1 Macro de 73.10%.
El modelo DistilBERT fine-tuned obtuvo un Accuracy de 64.81% y un F1 Macro de 66.56%. Aunque no superó a Ollama, sí obtuvo un F1 Macro superior al modelo Zero-shot.
La matriz de confusión mostró que DistilBERT presentó un buen desempeño en noticias de política y deportes, mientras que tuvo mayores dificultades en economía y ciencia y tecnología.
Una posible explicación es el tamaño limitado del dataset y el uso de etiquetas generadas mediante supervisión débil. Una mayor cantidad de noticias y una revisión manual más amplia de las etiquetas podría mejorar el desempeño del modelo en futuros trabajos.

Autor:Mirian de la Cruz
Trabajo desarrollado para:
Tarea 1 – Noticias peruanas: del titular al clasificador


# Reconocimiento del Alfabeto LSE en Tiempo Real
### Real-Time Spanish Sign Language Alphabet Recognition

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/MediaPipe-Hands-00897B?style=flat-square&logo=google&logoColor=white" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/License-MIT-2dc98b?style=flat-square" alt="MIT License"/>
</p>

<p align="center">
  Sistema de reconocimiento del alfabeto dactilológico LSE en tiempo real.<br>
  Extracción de landmarks con MediaPipe · Clasificador MLP+MobileNetV2 · Interfaz web.
</p>

---

> **Autor:** Oscar de Simone Benítez  
> **Universidad:** Universidad Francisco de Vitoria (UFV)  
> **Asignatura:** Sistemas de Percepción y Visión Artificial  
> **Fecha:** Mayo 2026

---

## Descripción / Description

Este proyecto implementa un sistema de visión artificial capaz de reconocer las 26 letras del alfabeto dactilológico del **Lenguaje de Signos Español (LSE)** en tiempo real a través de la webcam del ordenador.

**Características principales:**

- Reconocimiento en tiempo real a **30 fps** sobre CPU estándar (sin GPU requerida)
- Detección de **22 letras estáticas** (A–Z menos J,V,Y,Z) mediante clasificador híbrido MLP + MobileNetV2
- Detección de **4 letras dinámicas** (J, V, Y, Z) mediante análisis de trayectoria de landmarks
- Vector de características geométricas de **91 dimensiones** (63 brutas + 28 derivadas) robusto a variaciones de escala y posición
- Dataset propio de **44.079 muestras** recogidas con MediaPipe Hands
- Exactitud en test del **90,98 %** · Macro F1 del **89,05 %**
- Interfaz web accesible desde el navegador sin instalación adicional

---

## Demostración / Demo

**[Ver demo en YouTube →](https://youtu.be/s2fB30ND7Ng)**

### Dataset y recursos adicionales

| Recurso | Enlace |
|---|---|
| Dataset imágenes (4.4 GB) | [Google Drive](https://drive.google.com/drive/folders/1U8pjtmTOfe3LwwoDTvOcZzSaXe_kRURU?usp=sharing) |
| Dataset landmarks CSV (113 MB) | [GitHub Releases v1.0](https://github.com/Oscar-DSB/lse-vision-artificial/releases/tag/v1.0) |

La interfaz web muestra el fotograma de cámara, la letra detectada, la confianza del clasificador y un historial de las últimas letras reconocidas.

---

## Arquitectura del sistema / System Architecture

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────────────┐
│              │    │                   │    │                      │
│    Webcam    │───▶│  MediaPipe Hands  │───▶│  Feature Engineering │
│   30 fps     │    │  21 landmarks 3D  │    │  91 dimensiones      │
│              │    │                   │    │  (63 raw + 28 der.)  │
└──────────────┘    └───────────────────┘    └──────────┬───────────┘
                                                         │
                              ┌──────────────────────────┤
                              │                          │
                    ┌─────────▼──────────┐   ┌──────────▼──────────┐
                    │       MLP          │   │    MobileNetV2       │
                    │  91→512→256→128→22 │   │   (64×64 px crop)   │
                    │   Dropout(0.4)     │   │   fine-tuned         │
                    │     peso: 95%      │   │     peso: 5%         │
                    └─────────┬──────────┘   └──────────┬──────────┘
                              │                          │
                              └───────────┬──────────────┘
                                          │
                                ┌─────────▼─────────┐
                                │  Fusión Softmax    │
                                │  (media ponderada) │
                                └─────────┬─────────┘
                                          │
                             ┌────────────▼────────────┐
                             │   Predicción final       │
                             │   Letra + confianza      │
                             │   J/V/Y/Z: reglas tray.  │
                             └──────────────────────────┘
```

**Descripción de componentes:**

| Componente | Tecnología | Función |
|---|---|---|
| Captura de vídeo | OpenCV | Lectura de fotogramas de webcam a 30 fps |
| Detección de mano | MediaPipe Hands v2 | 21 landmarks 3D con confianza ≥ 0.7 |
| Normalización | NumPy | Centrado en muñeca, escalado por distancia MCP |
| Clasificador principal | PyTorch MLP | 91→512→256→128→22, softmax |
| Clasificador apoyo | MobileNetV2 | Fine-tuning sobre recorte 64×64 px |
| Letras dinámicas | Reglas Python | Trayectoria landmark 8 en ventana de 15 frames |
| Interfaz | FastAPI + HTML/JS | Servidor local, streaming MJPEG |

---

## Resultados / Results

### Métricas globales

| Métrica | Valor |
|---|---|
| **Accuracy (test)** | **90.98 %** |
| Accuracy (validación) | 99.88 % |
| Macro Precision | 86.28 % |
| Macro Recall | 93.67 % |
| **Macro F1** | **89.05 %** |
| Dataset (total filas) | 44,079 |
| Dimensiones de entrada | 91 |
| Clases estáticas | 22 |

### Resultados por clase

| Letra | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| A | 98.77 % | 85.11 % | 91.43 % | 282 |
| B | 92.69 % | 88.93 % | 90.77 % | 271 |
| C | 93.55 % | 95.96 % | 94.74 % | 272 |
| D | 100.00 % | 100.00 % | 100.00 % | 16 |
| **E** | **54.55 %** | 93.75 % | **68.97 %** | 32 |
| F | 94.12 % | 100.00 % | 96.97 % | 16 |
| G | 71.76 % | 94.57 % | 81.61 % | 129 |
| H | 72.73 % | 100.00 % | 84.21 % | 16 |
| I | 84.38 % | 81.82 % | 83.08 % | 33 |
| K | 83.78 % | 96.88 % | 89.86 % | 32 |
| L | 96.30 % | 91.23 % | 93.69 % | 285 |
| M | 86.49 % | 96.97 % | 91.43 % | 33 |
| N | 94.12 % | 96.97 % | 95.52 % | 33 |
| O | 88.89 % | 100.00 % | 94.12 % | 16 |
| **P** | **60.38 %** | 96.97 % | **74.42 %** | 33 |
| Q | 100.00 % | 88.24 % | 93.75 % | 34 |
| R | 93.52 % | 85.87 % | 89.53 % | 269 |
| S | 84.38 % | 87.10 % | 85.71 % | 31 |
| T | 90.91 % | 96.77 % | 93.75 % | 31 |
| **U** | **61.54 %** | 100.00 % | **76.19 %** | 16 |
| W | 98.39 % | 92.78 % | 95.50 % | 263 |
| X | 97.05 % | 90.91 % | 93.88 % | 253 |

> Las letras en **negrita** son las más problemáticas. E, P y U presentan baja precisión por sobrepredicción. Ver `memoria.html` para análisis completo de errores.

---

## Instalación / Installation

### Requisitos previos

- Python 3.13+
- Webcam funcional
- `uv` (recomendado) o `pip`

### Con uv (recomendado)

```powershell
# Clonar el repositorio
git clone <repo-url>
cd sistemas_percepcion_vision_artificial

# Crear entorno e instalar dependencias
uv sync

# Activar el entorno
.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate     # Linux/macOS
```

### Con pip

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencias principales

| Paquete | Versión mínima | Uso |
|---|---|---|
| `torch` + `torchvision` | 2.0+ | MLP + MobileNetV2 |
| `mediapipe` | 0.10+ | Detección de landmarks |
| `opencv-python` | 4.8+ | Captura de vídeo |
| `numpy` | 1.24+ | Procesado numérico |
| `scikit-learn` | 1.3+ | Métricas de evaluación |
| `fastapi` + `uvicorn` | 0.100+ | Servidor web |
| `matplotlib` | 3.7+ | Visualización |

---

## Uso / Usage

### Modo web (recomendado)

Lanza el servidor web y abre el navegador en `http://localhost:8000`:

```powershell
python -m lse_vision.main --mode web
```

La interfaz muestra en tiempo real:
- El fotograma de webcam con overlay de landmarks
- La letra detectada y su porcentaje de confianza
- Historial de las últimas 10 letras reconocidas
- Indicador de FPS y estado del detector

### Modo consola (webcam demo)

```powershell
python -m lse_vision.main --mode webcam
```

### Entrenamiento del MLP

```powershell
# Entrenar con el dataset existente
python -m lse_vision.main --mode train

# Opciones adicionales
python -m lse_vision.main --mode train --epochs 300 --batch-size 256 --lr 1e-3
```

El modelo entrenado se guarda en `models/mlp_lse.pt`.

### Evaluación

```powershell
# Evaluar sobre el conjunto de test
python -m lse_vision.main --mode evaluate

# Generar informe detallado con matriz de confusión
python -m lse_vision.main --mode evaluate --report
```

### Predicción sobre imagen estática

```powershell
python -m lse_vision.main --mode predict --image ruta/a/imagen.jpg
```

### Recogida de datos

```powershell
# Capturar muestras para una letra nueva
python -m lse_vision.main --mode capture --letter A --samples 500
```

Los landmarks se guardan automáticamente en `data/landmarks.csv`.

---

## Estructura del proyecto / Project Structure

```
sistemas_percepcion_vision_artificial/
│
├── lse_vision/                    # Paquete principal
│   ├── main.py                    # Punto de entrada CLI
│   ├── config/                    # Configuración global
│   ├── capture/                   # Captura de datos con webcam
│   ├── models/                    # Definición de arquitecturas (MLP, CNN)
│   ├── training/                  # Loop de entrenamiento y evaluación
│   ├── inference/                 # Pipeline de inferencia en tiempo real
│   ├── web/                       # Servidor FastAPI + frontend HTML/JS
│   └── utils/                     # Utilidades comunes
│
├── models/                        # Modelos entrenados guardados
│   └── label_names.pkl            # Mapeo índice → letra
│
├── data/                          # Datos procesados
│   └── landmarks.csv              # 44,079 filas × 92 columnas
│
├── dataset/                       # Imágenes de referencia por clase
│   └── Image/                     # 22 clases × imágenes PNG
│
├── outputs/                       # Resultados de evaluación
│   ├── checkpoints/               # Modelos entrenados (.pt)
│   └── metrics/                   # test_metrics.json, confusion_matrix.csv
│
├── Info_para_trabajo/             # Documentación de referencia académica
│
├── LSE_demo.mp4                   # Vídeo demo del sistema en funcionamiento
├── memoria.html                   # Memoria técnica completa (auto-contenida)
├── README.md                      # Este archivo
├── pyproject.toml                 # Metadatos del proyecto (uv/pip)
├── requirements.txt               # Dependencias pip
└── uv.lock                        # Lock file de uv
```

---

## Dataset

El dataset fue construido mediante captura propia con webcam utilizando MediaPipe Hands.

| Característica | Valor |
|---|---|
| Total de muestras | 44,079 |
| Clases | 22 letras estáticas del LSE |
| Dimensiones de entrada | 91 (63 coordenadas + 28 derivadas) |
| Formato | CSV (landmarks.csv) |
| Sujetos | 1 (sesiones con variación de iluminación y orientación) |
| Split | 70% train / 15% val / 15% test (estratificado) |

### Vector de características (91 dimensiones)

```
Bloque A — 63 coordenadas brutas normalizadas:
  21 landmarks × (x, y, z) normalizados respecto a la muñeca

Bloque B — 28 características derivadas:
  10 distancias inter-dedo (pares de puntas)
   5 ángulos de flexión en MCP por dedo
   5 ratios de apertura entre dedos adyacentes
   4 ángulos de curvatura por falange
   4 distancias punta-palma por dedo
```

---

## Análisis de errores destacados / Key Error Analysis

| Par confundido | Muestras erróneas | Causa probable |
|---|---|---|
| A → E | 13 | Curvatura similar de dedos cerrados |
| B → G | 14 | Extensión parcial del índice |
| G → C | 8 / C → G: 3 | Forma de pinza, diferencia en orientación de palma |
| I → P | 5 | Configuración pulgar-índice similar |
| W → U | 7 | Número de dedos extendidos adyacentes |
| X → R | 8 | Cruce de dedos con morfología parecida |

Ver `memoria.html` §07 para el análisis completo con explicación geométrica de cada par.

---

## Contribución / Contributing

Las contribuciones son bienvenidas. Para contribuir:

1. Haz un fork del repositorio
2. Crea una rama con tu funcionalidad: `git checkout -b feature/mi-funcionalidad`
3. Haz commit de tus cambios: `git commit -m 'Add: descripción de la mejora'`
4. Haz push a la rama: `git push origin feature/mi-funcionalidad`
5. Abre un Pull Request con descripción detallada

### Ideas para contribuir

- Añadir muestras de nuevos sujetos al dataset para mejorar la generalización
- Implementar el clasificador LSTM para las letras dinámicas (J, V, Y, Z)
- Añadir soporte para mano izquierda de forma explícita
- Optimizar el modelo para despliegue en dispositivos móviles (TFLite/ONNX)
- Ampliar el vocabulario más allá del alfabeto (palabras completas en LSE)
- Mejorar la interfaz web con modo de práctica/aprendizaje guiado

---

## Referencias / References

1. Zhang, F. et al. (2020). *MediaPipe Hands: On-device Real-time Hand Landmark Detection*. arXiv:2006.10214. CVPR Workshop.
2. Pigou, L., Dieleman, S., Roodhooft, P., & Dhoedt, B. (2014). *Sign Language Recognition Using Convolutional Neural Networks*. ECCV Workshop on Human Behavior Understanding.
3. Rastgoo, R., Kiani, K., & Escalera, S. (2021). *Sign Language Recognition: A Deep Survey*. Expert Systems with Applications, 164, 113794.

---

## Licencia / License

Este proyecto está bajo la licencia **MIT**. Consulta el archivo `LICENSE` para más detalles.

```
MIT License

Copyright (c) 2026 Oscar de Simone Benítez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<p align="center">
  Desarrollado con PyTorch · MediaPipe · OpenCV<br>
  Universidad Francisco de Vitoria · Sistemas de Percepción y Visión Artificial · 2026
</p>

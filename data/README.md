# Datos del proyecto

Esta carpeta organiza el ciclo de vida de los datos del prototipo de traduccion LSE:

- `raw/`: capturas originales y secuencias recien adquiridas
- `interim/`: datos validados o transformaciones intermedias
- `processed/`: particiones finales, metadatos serializados y tensores preparados para entrenamiento

No se versionan los datos pesados ni sensibles. La estructura si se mantiene para asegurar reproducibilidad.

# Clasificador de Audio CNN - K-Fold Ensemble

Sistema de clasificación de audio usando CNN con ensemble de 5 k-folds.

## Inicio Rápido

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecutar servidor:
```bash
python app.py
```

3. Acceder a: http://localhost:5000

## Estructura del Proyecto

- `app.py` - Backend Flask
- `models/` - Modelos entrenados (mfcc, mel, concat)
- `templates/` - Frontend HTML
- `uploads/` - Archivos temporales
- `results/` - Historial de predicciones

## API Endpoints

- `GET /api/info` - Información del sistema
- `POST /api/predict` - Clasificar un audio
- `POST /api/predict/batch` - Clasificar múltiples audios
- `GET /api/history` - Historial de predicciones

## Producción

Ver archivo de configuración para deployment en cloud.

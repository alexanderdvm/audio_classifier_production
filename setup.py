"""
Script de Setup Automático para Clasificador de Audio
Copia modelos, genera clases y verifica la instalación
"""

import os
import sys
import json
import shutil
from pathlib import Path
from glob import glob

# Colores para consola
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_step(msg):
    print(f"{Colors.BLUE}[STEP]{Colors.END} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

# ==================== CONFIGURACIÓN ====================
# IMPORTANTE: Ajusta esta ruta a tu directorio de entrenamiento
TRAINING_DIR = r"D:\dataset pf\COPIA\organized_by_type\data augmentation 200\data_augmentation_train2"

# Ruta del proyecto actual
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
TEMPLATES_DIR = PROJECT_DIR / "templates"

# Feature types a copiar
FEATURE_TYPES = ['mfcc', 'mel', 'concat']

# ==================== FUNCIONES ====================

def check_python_version():
    """Verifica versión de Python"""
    print_step("Verificando versión de Python...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} no es compatible. Requiere Python 3.8+")
        return False

def create_project_structure():
    """Crea estructura de directorios"""
    print_step("Creando estructura de directorios...")
    
    dirs = [
        MODELS_DIR,
        TEMPLATES_DIR,
        PROJECT_DIR / "uploads",
        PROJECT_DIR / "results",
        PROJECT_DIR / "static"
    ]
    
    for d in dirs:
        d.mkdir(exist_ok=True)
        print_success(f"Directorio creado: {d}")

def copy_models():
    """Copia modelos desde directorio de entrenamiento"""
    print_step("Copiando modelos entrenados...")
    
    training_path = Path(TRAINING_DIR)
    
    if not training_path.exists():
        print_error(f"Directorio de entrenamiento no encontrado: {TRAINING_DIR}")
        print_warning("Por favor, ajusta la variable TRAINING_DIR en este script")
        return False
    
    copied_count = 0
    
    for feature in FEATURE_TYPES:
        source_dir = training_path / feature
        dest_dir = MODELS_DIR / feature
        
        if not source_dir.exists():
            print_warning(f"Feature '{feature}' no encontrado en {source_dir}")
            continue
        
        print_step(f"Copiando {feature}...")
        
        # Copiar carpeta de modelos
        source_models = source_dir / "models"
        if source_models.exists():
            dest_models = dest_dir / "models"
            dest_models.mkdir(parents=True, exist_ok=True)
            
            for model_file in source_models.glob("*.h5"):
                dest_file = dest_models / model_file.name
                shutil.copy2(model_file, dest_file)
                print_success(f"  → {model_file.name}")
                copied_count += 1
        
        # Copiar summary.json
        summary_file = source_dir / "summary.json"
        if summary_file.exists():
            shutil.copy2(summary_file, dest_dir / "summary.json")
            print_success(f"  → summary.json")
        
        # Copiar fold_metrics.csv
        metrics_file = source_dir / "fold_metrics.csv"
        if metrics_file.exists():
            shutil.copy2(metrics_file, dest_dir / "fold_metrics.csv")
            print_success(f"  → fold_metrics.csv")
    
    print_success(f"Total de modelos copiados: {copied_count}")
    return copied_count > 0

def extract_classes_from_training():
    """Extrae las clases desde el directorio de entrenamiento"""
    print_step("Extrayendo clases desde datos de entrenamiento...")
    
    training_path = Path(TRAINING_DIR)
    
    # Buscar en test_web o en metadata
    test_web_dir = training_path / "test_web"
    classes = []
    
    if test_web_dir.exists():
        # Obtener clases desde subdirectorios
        classes = sorted([d.name for d in test_web_dir.iterdir() if d.is_dir()])
        print_success(f"Clases encontradas en test_web: {len(classes)}")
    else:
        # Intentar desde metadata
        for feature in FEATURE_TYPES:
            meta_path = training_path / feature / "summary.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    data = json.load(f)
                    if 'classes' in data:
                        classes = data['classes']
                        break
        
        if classes:
            print_success(f"Clases encontradas en metadata: {len(classes)}")
    
    return classes

def create_classes_file():
    """Crea archivo classes.json"""
    print_step("Creando archivo de clases...")
    
    # Intentar extraer desde entrenamiento
    classes = extract_classes_from_training()
    
    if not classes:
        print_warning("No se pudieron extraer clases automáticamente")
        print_warning("Por favor, edita manualmente models/classes.json")
        
        # Crear archivo de ejemplo
        classes = ["clase_ejemplo_1", "clase_ejemplo_2", "clase_ejemplo_3"]
    
    classes_file = MODELS_DIR / "classes.json"
    with open(classes_file, 'w', encoding='utf-8') as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)
    
    print_success(f"Archivo creado: {classes_file}")
    print_success(f"Clases totales: {len(classes)}")
    
    for i, cls in enumerate(classes[:5], 1):
        print(f"  {i}. {cls}")
    if len(classes) > 5:
        print(f"  ... y {len(classes) - 5} más")
    
    return True

def verify_models():
    """Verifica que los modelos estén correctamente copiados"""
    print_step("Verificando modelos...")
    
    all_valid = True
    
    for feature in FEATURE_TYPES:
        models_dir = MODELS_DIR / feature / "models"
        
        if not models_dir.exists():
            print_warning(f"Carpeta de modelos no existe: {models_dir}")
            all_valid = False
            continue
        
        model_files = list(models_dir.glob("*.h5"))
        expected_folds = 5
        
        if len(model_files) >= expected_folds:
            print_success(f"{feature}: {len(model_files)} modelos encontrados")
        else:
            print_warning(f"{feature}: Solo {len(model_files)}/{expected_folds} modelos")
            all_valid = False
    
    return all_valid

def check_dependencies():
    """Verifica dependencias instaladas"""
    print_step("Verificando dependencias Python...")
    
    required = [
        'flask',
        'tensorflow',
        'librosa',
        'numpy',
        'pandas'
    ]
    
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print_success(f"{package}")
        except ImportError:
            print_error(f"{package} - NO INSTALADO")
            missing.append(package)
    
    if missing:
        print_warning(f"\nPara instalar dependencias faltantes:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True

def create_requirements():
    """Crea archivo requirements.txt"""
    print_step("Creando requirements.txt...")
    
    requirements = """# Backend
flask==3.0.0
flask-cors==4.0.0
werkzeug==3.0.1

# Audio processing
librosa==0.10.1
soundfile==0.12.1

# Deep Learning
tensorflow==2.15.0

# Data processing
numpy==1.24.3
pandas==2.1.4

# Utils
python-dateutil==2.8.2

# Production server
gunicorn==21.2.0
waitress==2.1.2
"""
    
    req_file = PROJECT_DIR / "requirements.txt"
    with open(req_file, 'w') as f:
        f.write(requirements)
    
    print_success(f"Archivo creado: {req_file}")
    print_warning("Instalar con: pip install -r requirements.txt")
    
    return True

def create_readme():
    """Crea README.md básico"""
    print_step("Creando README.md...")
    
    readme = """# Clasificador de Audio CNN - K-Fold Ensemble

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

3. Acceder a: http://localhost:8000

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
"""
    
    readme_file = PROJECT_DIR / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme)
    
    print_success(f"Archivo creado: {readme_file}")
    return True

def print_summary():
    """Imprime resumen final"""
    print("\n" + "="*60)
    print(f"{Colors.GREEN}SETUP COMPLETADO{Colors.END}")
    print("="*60)
    
    print("\n📋 Próximos pasos:")
    print("\n1. Verificar/editar archivo de clases:")
    print(f"   {MODELS_DIR / 'classes.json'}")
    
    print("\n2. Instalar dependencias:")
    print(f"   pip install -r requirements.txt")
    
    print("\n3. Ejecutar servidor:")
    print(f"   python app.py")
    
    print("\n4. Acceder a la aplicación:")
    print(f"   http://localhost:8000")
    
    print("\n" + "="*60 + "\n")

# ==================== MAIN ====================

def main():
    print("\n" + "="*60)
    print(f"{Colors.BLUE}SETUP AUTOMÁTICO - CLASIFICADOR DE AUDIO{Colors.END}")
    print("="*60 + "\n")
    
    # 1. Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # 2. Crear estructura
    create_project_structure()
    
    # 3. Copiar modelos
    if not copy_models():
        print_warning("Algunos modelos no se copiaron. Verifica la ruta TRAINING_DIR")
    
    # 4. Crear archivo de clases
    create_classes_file()
    
    # 5. Verificar modelos
    verify_models()
    
    # 6. Crear requirements.txt
    create_requirements()
    
    # 7. Crear README
    create_readme()
    
    # 8. Verificar dependencias
    check_dependencies()
    
    # 9. Resumen final
    print_summary()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print_error(f"Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
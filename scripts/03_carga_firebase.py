"""
========================================================
Script 03 - Carga de Datos en Firebase Firestore
Proyecto: Predicción de Suscripción a Depósito Bancario
Equipo:   Sebastián Aravena | Enrique Oporto | Alexei Sandoval
Etapa:    Pipeline - Carga en Base de Datos (Actividad 2.3)
========================================================

Descripción:
    Carga el dataset limpio y procesado en Firebase Firestore,
    organizándolo en 5 colecciones:

    1. clientes_raw       → Dataset original sin transformar
    2. clientes_procesado → Dataset limpio y transformado (listo para modelo)
    3. feature_engineering→ Features nuevas creadas en la etapa anterior
    4. metricas_dataset   → Estadísticas y resumen del dataset
    5. predicciones       → Colección vacía lista para logs de la API

Uso:
    python scripts/03_carga_firebase.py
"""

import os
import json
import logging
import math
import time
from datetime import datetime

import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore


# ─────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LOGGING
# ─────────────────────────────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(
    LOG_DIR,
    f"carga_firebase_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────
CREDENCIALES_PATH  = "firebase_credentials.json"
DATASET_RAW_PATH   = "data/raw/bank_marketing.csv"
DATASET_CLEAN_PATH = "data/processed/bank_clean.csv"
PROJECT_ID         = "prediccion-deposito-bancario"
METADATOS_PATH     = "data/processed/carga_firebase_metadata.json"

# Tamaño del batch para subida (Firestore permite máx 500 por batch)
BATCH_SIZE = 400

# Colecciones a crear en Firestore
COLECCION_RAW         = "clientes_raw"
COLECCION_PROCESADO   = "clientes_procesado"
COLECCION_FEATURES    = "feature_engineering"
COLECCION_METRICAS    = "metricas_dataset"
COLECCION_PREDICCIONES = "predicciones"


# ─────────────────────────────────────────────────────────
# INICIALIZACIÓN DE FIREBASE
# ─────────────────────────────────────────────────────────

def inicializar_firebase() -> firestore.Client:
    """Inicializa la conexión con Firebase usando las credenciales."""
    if not os.path.isfile(CREDENCIALES_PATH):
        logger.error(
            "Archivo de credenciales no encontrado: %s", CREDENCIALES_PATH
        )
        raise FileNotFoundError(
            f"Coloca el archivo firebase_credentials.json en la raíz del proyecto."
        )

    cred = credentials.Certificate(CREDENCIALES_PATH)
    firebase_admin.initialize_app(cred, {"projectId": PROJECT_ID})
    db = firestore.client()
    logger.info("Conexión con Firebase Firestore establecida. ✓")
    return db


# ─────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────

def limpiar_valor(valor):
    """
    Convierte valores no serializables por Firestore (NaN, inf)
    a None para evitar errores al subir los documentos.
    """
    if isinstance(valor, float) and (math.isnan(valor) or math.isinf(valor)):
        return None
    return valor


def df_a_documentos(df: pd.DataFrame) -> list:
    """Convierte un DataFrame a una lista de diccionarios limpios."""
    documentos = []
    for _, fila in df.iterrows():
        doc = {k: limpiar_valor(v) for k, v in fila.items()}
        documentos.append(doc)
    return documentos


def subir_en_batches(db, coleccion: str, documentos: list, id_prefix: str = "doc"):
    """
    Sube documentos a Firestore en lotes de BATCH_SIZE.
    Firestore tiene un límite de 500 operaciones por batch.
    """
    total = len(documentos)
    lotes = math.ceil(total / BATCH_SIZE)
    subidos = 0

    logger.info(
        "  Subiendo %d documentos a '%s' en %d lote(s)...",
        total, coleccion, lotes
    )

    for i in range(lotes):
        batch = db.batch()
        inicio = i * BATCH_SIZE
        fin = min(inicio + BATCH_SIZE, total)
        fragmento = documentos[inicio:fin]

        for j, doc in enumerate(fragmento):
            doc_id = f"{id_prefix}_{inicio + j:05d}"
            ref = db.collection(coleccion).document(doc_id)
            batch.set(ref, doc)

        batch.commit()
        time.sleep(1)
        subidos += len(fragmento)
        logger.info(
            "    Lote %d/%d completado — %d/%d documentos subidos.",
            i + 1, lotes, subidos, total
        )

    logger.info("  ✓ Colección '%s' cargada exitosamente.", coleccion)
    return subidos


# ─────────────────────────────────────────────────────────
# CARGAS POR COLECCIÓN
# ─────────────────────────────────────────────────────────

def cargar_clientes_raw(db, df_raw: pd.DataFrame):
    """Carga el dataset original sin transformar."""
    logger.info("── Cargando colección: %s ──────────────────────────", COLECCION_RAW)
    documentos = df_a_documentos(df_raw)
    return subir_en_batches(db, COLECCION_RAW, documentos, id_prefix="raw")


def cargar_clientes_procesado(db, df_clean: pd.DataFrame):
    """Carga el dataset limpio y transformado."""
    logger.info("── Cargando colección: %s ──────────────────", COLECCION_PROCESADO)

    # Solo columnas originales transformadas (sin las de feature engineering)
    cols_base = [
        "age", "education", "default", "balance", "housing", "loan",
        "day", "duration", "campaign", "pdays", "previous", "deposit"
    ] + [c for c in df_clean.columns if any(
        c.startswith(p) for p in ["job_", "marital_", "contact_", "month_", "poutcome_"]
    )]

    cols_existentes = [c for c in cols_base if c in df_clean.columns]
    df_base = df_clean[cols_existentes].copy()
    documentos = df_a_documentos(df_base)
    return subir_en_batches(db, COLECCION_PROCESADO, documentos, id_prefix="clean")


def cargar_feature_engineering(db, df_clean: pd.DataFrame):
    """Carga las features creadas en la etapa de ingeniería."""
    logger.info("── Cargando colección: %s ─────────────────", COLECCION_FEATURES)

    cols_features = [
        "contacted_before", "balance_bin",
        "contact_intensity", "age_group", "deposit"
    ]
    cols_existentes = [c for c in cols_features if c in df_clean.columns]
    df_features = df_clean[cols_existentes].copy()
    documentos = df_a_documentos(df_features)
    return subir_en_batches(db, COLECCION_FEATURES, documentos, id_prefix="feat")


def cargar_metricas_dataset(db, df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    """
    Carga un resumen estadístico del dataset como documento único.
    Útil para el dashboard y análisis posterior.
    """
    logger.info("── Cargando colección: %s ─────────────────────", COLECCION_METRICAS)

    metricas = {
        "timestamp": datetime.now().isoformat(),
        "dataset_raw": {
            "total_registros": int(len(df_raw)),
            "total_columnas": int(len(df_raw.columns)),
            "valores_nulos": int(df_raw.isnull().sum().sum()),
            "duplicados": int(df_raw.duplicated().sum()),
            "deposit_yes": int((df_raw["deposit"] == "yes").sum()),
            "deposit_no": int((df_raw["deposit"] == "no").sum()),
        },
        "dataset_procesado": {
            "total_registros": int(len(df_clean)),
            "total_columnas": int(len(df_clean.columns)),
            "deposit_1": int((df_clean["deposit"] == 1).sum()),
            "deposit_0": int((df_clean["deposit"] == 0).sum()),
        },
        "estadisticas_numericas": {
            col: {
                "media": round(float(df_raw[col].mean()), 2),
                "std": round(float(df_raw[col].std()), 2),
                "min": round(float(df_raw[col].min()), 2),
                "max": round(float(df_raw[col].max()), 2),
            }
            for col in ["age", "balance", "duration", "campaign"]
        },
        "distribucion_job": df_raw["job"].value_counts().to_dict(),
        "distribucion_education": df_raw["education"].value_counts().to_dict(),
        "distribucion_marital": df_raw["marital"].value_counts().to_dict(),
    }

    db.collection(COLECCION_METRICAS).document("resumen_general").set(metricas)
    logger.info("  ✓ Colección '%s' cargada exitosamente.", COLECCION_METRICAS)
    return 1


def crear_coleccion_predicciones(db):
    """
    Crea la colección 'predicciones' con un documento inicial de ejemplo.
    Esta colección será usada por la API para registrar logs en tiempo real.
    """
    logger.info("── Creando colección: %s ────────────────────────", COLECCION_PREDICCIONES)

    doc_inicial = {
        "timestamp": datetime.now().isoformat(),
        "tipo": "inicializacion",
        "descripcion": "Colección inicializada. Lista para recibir logs de predicciones.",
        "version_modelo": "pendiente",
        "campos_esperados": [
            "age", "job", "marital", "education", "default",
            "balance", "housing", "loan", "contact", "day",
            "month", "duration", "campaign", "pdays", "previous", "poutcome"
        ],
        "target": "deposit (0=no suscribe, 1=suscribe)"
    }

    db.collection(COLECCION_PREDICCIONES).document("_init").set(doc_inicial)
    logger.info("  ✓ Colección '%s' creada exitosamente.", COLECCION_PREDICCIONES)
    return 1


# ─────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────

def ejecutar_carga():
    """
    Ejecuta la carga completa a Firebase Firestore:
      1. Inicializa Firebase
      2. Carga dataset raw     → clientes_raw
      3. Carga dataset limpio  → clientes_procesado
      4. Carga features        → feature_engineering
      5. Carga métricas        → metricas_dataset
      6. Crea colección logs   → predicciones
      7. Guarda metadatos locales
    """
    logger.info("=" * 60)
    logger.info("INICIO — Carga en Firebase Firestore")
    logger.info("Proyecto: %s", PROJECT_ID)
    logger.info("=" * 60)

    # Inicializar Firebase
    db = inicializar_firebase()

    # Cargar datasets
    logger.info("Leyendo datasets...")
    if not os.path.isfile(DATASET_RAW_PATH):
        raise FileNotFoundError(f"Dataset raw no encontrado: {DATASET_RAW_PATH}")
    if not os.path.isfile(DATASET_CLEAN_PATH):
        raise FileNotFoundError(f"Dataset procesado no encontrado: {DATASET_CLEAN_PATH}")

    df_raw   = pd.read_csv(DATASET_RAW_PATH)
    df_clean = pd.read_csv(DATASET_CLEAN_PATH)
    df_raw   = df_raw.sample(n=2000, random_state=42).reset_index(drop=True)
    df_clean = df_clean.sample(n=2000, random_state=42).reset_index(drop=True)
    logger.info(
        "Datasets cargados — Raw: %d registros | Procesado: %d registros",
        len(df_raw), len(df_clean)
    )

    # Ejecutar cargas
    resumen = {}
    resumen[COLECCION_RAW]          = cargar_clientes_raw(db, df_raw)
    resumen[COLECCION_PROCESADO]    = cargar_clientes_procesado(db, df_clean)
    resumen[COLECCION_FEATURES]     = cargar_feature_engineering(db, df_clean)
    resumen[COLECCION_METRICAS]     = cargar_metricas_dataset(db, df_raw, df_clean)
    resumen[COLECCION_PREDICCIONES] = crear_coleccion_predicciones(db)

    # Metadatos locales
    os.makedirs(os.path.dirname(METADATOS_PATH), exist_ok=True)
    metadatos = {
        "timestamp": datetime.now().isoformat(),
        "proyecto_firebase": PROJECT_ID,
        "colecciones_creadas": list(resumen.keys()),
        "documentos_subidos": resumen,
        "total_documentos": sum(resumen.values()),
    }
    with open(METADATOS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=4, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("CARGA COMPLETADA EXITOSAMENTE")
    logger.info("Colecciones creadas en Firestore:")
    for col, total in resumen.items():
        logger.info("  %-30s → %d documentos", col, total)
    logger.info("Total documentos subidos: %d", sum(resumen.values()))
    logger.info("Próxima etapa: ejecutar scripts/04_entrenamiento.py")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        ejecutar_carga()
    except Exception as e:
        logger.critical("La carga a Firebase falló: %s", e)
        exit(1)
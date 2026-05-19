"""
========================================================
Script 03 - Carga de Datos en MongoDB
Proyecto: Predicción de Suscripción a Depósito Bancario
Equipo:   Sebastián Aravena | Enrique Oporto | Alexei Sandoval
Etapa:    Pipeline - Carga en Base de Datos
========================================================

Descripción:
    Carga el dataset limpio y procesado en MongoDB, organizándolo en
    5 colecciones equivalentes a la versión anterior en Firebase:

    1. clientes_raw        → Dataset original sin transformar
    2. clientes_procesado  → Dataset limpio y transformado listo para modelo
    3. feature_engineering → Features nuevas creadas en la etapa anterior
    4. metricas_dataset    → Estadísticas y resumen del dataset
    5. predicciones        → Colección inicial lista para logs de la API

Requisitos:
    pip install pymongo pandas

Variable de entorno requerida:
    MONGODB_URI="mongodb+srv://usuario:password@cluster.mongodb.net/?retryWrites=true&w=majority"

Uso:
    python scripts/03_carga_mongodb.py
    python scripts/03_carga_mongodb.py --database prediccion_deposito_bancario
    python scripts/03_carga_mongodb.py --modo agregar
    python scripts/03_carga_mongodb.py --limite 2000
"""

import os
import json
import logging
import math
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, ServerSelectionTimeoutError

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ─────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LOGGING
# ─────────────────────────────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(
    LOG_DIR,
    f"carga_mongodb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
MONGODB_URI_ENV     = "MONGODB_URI"
DATABASE_DEFAULT    = "prediccion_deposito_bancario"
DATASET_RAW_PATH    = "data/raw/bank_marketing.csv"
DATASET_CLEAN_PATH  = "data/processed/bank_clean.csv"
METADATOS_PATH      = "data/processed/carga_mongodb_metadata.json"

BATCH_SIZE_DEFAULT = 1000

COLECCION_RAW          = "clientes_raw"
COLECCION_PROCESADO    = "clientes_procesado"
COLECCION_FEATURES     = "feature_engineering"
COLECCION_METRICAS     = "metricas_dataset"
COLECCION_PREDICCIONES = "predicciones"


# ─────────────────────────────────────────────────────────
# INICIALIZACIÓN DE MONGODB
# ─────────────────────────────────────────────────────────
def inicializar_mongodb(database_name: str):
    """Inicializa la conexión con MongoDB usando MONGODB_URI."""
    mongodb_uri = os.getenv(MONGODB_URI_ENV)

    if not mongodb_uri:
        raise ValueError(
            "No se encontró la variable de entorno MONGODB_URI. "
            "Configúrala antes de ejecutar el script."
        )

    try:
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
        client.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        raise ConnectionError(
            "No se pudo conectar a MongoDB. Revisa tu URI, usuario, contraseña, "
            "IP permitida en Atlas o conexión a internet."
        ) from e

    db = client[database_name]
    logger.info("Conexión con MongoDB establecida. Base de datos: %s ✓", database_name)
    return client, db


# ─────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────
def limpiar_valor(valor: Any):
    """
    Convierte valores de Pandas/Numpy a tipos compatibles con MongoDB/BSON.
    También convierte NaN, NaT e infinitos a None.
    """
    if pd.isna(valor):
        return None

    if isinstance(valor, np.integer):
        return int(valor)

    if isinstance(valor, np.floating):
        valor_float = float(valor)
        if math.isnan(valor_float) or math.isinf(valor_float):
            return None
        return valor_float

    if isinstance(valor, np.bool_):
        return bool(valor)

    if isinstance(valor, pd.Timestamp):
        return valor.to_pydatetime()

    if isinstance(valor, float) and (math.isnan(valor) or math.isinf(valor)):
        return None

    return valor


def df_a_documentos(df: pd.DataFrame, id_prefix: str) -> List[Dict[str, Any]]:
    """Convierte un DataFrame a documentos para MongoDB con _id determinístico."""
    documentos = []
    for idx, fila in df.reset_index(drop=True).iterrows():
        doc = {k: limpiar_valor(v) for k, v in fila.items()}
        doc["_id"] = f"{id_prefix}_{idx:05d}"
        doc["fecha_carga"] = datetime.now()
        documentos.append(doc)
    return documentos


def preparar_coleccion(db, nombre_coleccion: str, modo: str) -> Collection:
    """
    Prepara la colección de destino.
    - reemplazar: elimina documentos existentes antes de cargar.
    - agregar: conserva documentos existentes e intenta insertar nuevos.
    """
    coleccion = db[nombre_coleccion]

    if modo == "reemplazar":
        resultado = coleccion.delete_many({})
        logger.info(
            "  Colección '%s' limpiada antes de cargar (%d documentos eliminados).",
            nombre_coleccion,
            resultado.deleted_count
        )

    return coleccion


def subir_en_batches(
    db,
    nombre_coleccion: str,
    documentos: List[Dict[str, Any]],
    batch_size: int,
    modo: str
) -> int:
    """Sube documentos a MongoDB usando insert_many por lotes."""
    total = len(documentos)
    if total == 0:
        logger.warning("  No hay documentos para subir en '%s'.", nombre_coleccion)
        return 0

    coleccion = preparar_coleccion(db, nombre_coleccion, modo)
    lotes = math.ceil(total / batch_size)
    subidos = 0

    logger.info(
        "  Subiendo %d documentos a '%s' en %d lote(s)...",
        total,
        nombre_coleccion,
        lotes
    )

    for i in range(lotes):
        inicio = i * batch_size
        fin = min(inicio + batch_size, total)
        fragmento = documentos[inicio:fin]

        try:
            resultado = coleccion.insert_many(fragmento, ordered=False)
            subidos += len(resultado.inserted_ids)
        except BulkWriteError as e:
            errores = e.details.get("writeErrors", [])
            duplicados = sum(1 for error in errores if error.get("code") == 11000)
            insertados = e.details.get("nInserted", 0)
            subidos += insertados
            logger.warning(
                "    Lote %d/%d con advertencias: %d insertados, %d duplicados.",
                i + 1,
                lotes,
                insertados,
                duplicados
            )

        logger.info(
            "    Lote %d/%d completado — %d/%d documentos subidos.",
            i + 1,
            lotes,
            subidos,
            total
        )

    logger.info("  ✓ Colección '%s' cargada exitosamente.", nombre_coleccion)
    return subidos


def crear_indices(db):
    """Crea índices simples para consultas frecuentes."""
    db[COLECCION_RAW].create_index([("deposit", ASCENDING)])
    db[COLECCION_PROCESADO].create_index([("deposit", ASCENDING)])
    db[COLECCION_FEATURES].create_index([("deposit", ASCENDING)])
    db[COLECCION_PREDICCIONES].create_index([("timestamp", ASCENDING)])
    logger.info("Índices creados/verificados correctamente. ✓")


# ─────────────────────────────────────────────────────────
# CARGAS POR COLECCIÓN
# ─────────────────────────────────────────────────────────
def cargar_clientes_raw(db, df_raw: pd.DataFrame, batch_size: int, modo: str):
    """Carga el dataset original sin transformar."""
    logger.info("── Cargando colección: %s ──────────────────────────", COLECCION_RAW)
    documentos = df_a_documentos(df_raw, id_prefix="raw")
    return subir_en_batches(db, COLECCION_RAW, documentos, batch_size, modo)


def cargar_clientes_procesado(db, df_clean: pd.DataFrame, batch_size: int, modo: str):
    """Carga el dataset limpio y transformado."""
    logger.info("── Cargando colección: %s ──────────────────", COLECCION_PROCESADO)

    cols_base = [
        "age", "education", "default", "balance", "housing", "loan",
        "day", "duration", "campaign", "pdays", "previous", "deposit"
    ] + [
        c for c in df_clean.columns if any(
            c.startswith(p) for p in ["job_", "marital_", "contact_", "month_", "poutcome_"]
        )
    ]

    cols_existentes = [c for c in cols_base if c in df_clean.columns]
    df_base = df_clean[cols_existentes].copy()
    documentos = df_a_documentos(df_base, id_prefix="clean")
    return subir_en_batches(db, COLECCION_PROCESADO, documentos, batch_size, modo)


def cargar_feature_engineering(db, df_clean: pd.DataFrame, batch_size: int, modo: str):
    """Carga las features creadas en la etapa de ingeniería."""
    logger.info("── Cargando colección: %s ─────────────────", COLECCION_FEATURES)

    cols_features = [
        "contacted_before", "balance_bin",
        "contact_intensity", "age_group", "deposit"
    ]
    cols_existentes = [c for c in cols_features if c in df_clean.columns]
    df_features = df_clean[cols_existentes].copy()
    documentos = df_a_documentos(df_features, id_prefix="feat")
    return subir_en_batches(db, COLECCION_FEATURES, documentos, batch_size, modo)


def cargar_metricas_dataset(db, df_raw: pd.DataFrame, df_clean: pd.DataFrame, modo: str):
    """Carga un resumen estadístico del dataset como documento único."""
    logger.info("── Cargando colección: %s ─────────────────────", COLECCION_METRICAS)

    coleccion = preparar_coleccion(db, COLECCION_METRICAS, modo)

    metricas = {
        "_id": "resumen_general",
        "timestamp": datetime.now(),
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
            if col in df_raw.columns
        },
        "distribucion_job": df_raw["job"].value_counts().to_dict() if "job" in df_raw.columns else {},
        "distribucion_education": df_raw["education"].value_counts().to_dict() if "education" in df_raw.columns else {},
        "distribucion_marital": df_raw["marital"].value_counts().to_dict() if "marital" in df_raw.columns else {},
    }

    coleccion.replace_one({"_id": "resumen_general"}, metricas, upsert=True)
    logger.info("  ✓ Colección '%s' cargada exitosamente.", COLECCION_METRICAS)
    return 1


def crear_coleccion_predicciones(db, modo: str):
    """Crea la colección 'predicciones' con un documento inicial."""
    logger.info("── Creando colección: %s ────────────────────────", COLECCION_PREDICCIONES)

    coleccion = preparar_coleccion(db, COLECCION_PREDICCIONES, modo)

    doc_inicial = {
        "_id": "_init",
        "timestamp": datetime.now(),
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

    coleccion.replace_one({"_id": "_init"}, doc_inicial, upsert=True)
    logger.info("  ✓ Colección '%s' creada exitosamente.", COLECCION_PREDICCIONES)
    return 1


# ─────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────
def ejecutar_carga(
    database_name: str = DATABASE_DEFAULT,
    modo: str = "reemplazar",
    limite: Optional[int] = None,
    batch_size: int = BATCH_SIZE_DEFAULT
):
    """
    Ejecuta la carga completa a MongoDB:
      1. Inicializa MongoDB
      2. Carga dataset raw      → clientes_raw
      3. Carga dataset limpio   → clientes_procesado
      4. Carga features         → feature_engineering
      5. Carga métricas         → metricas_dataset
      6. Crea colección logs    → predicciones
      7. Crea índices
      8. Guarda metadatos locales
    """
    logger.info("=" * 60)
    logger.info("INICIO — Carga en MongoDB")
    logger.info("Base de datos: %s", database_name)
    logger.info("Modo de carga: %s", modo)
    logger.info("=" * 60)

    client = None
    try:
        client, db = inicializar_mongodb(database_name)

        logger.info("Leyendo datasets...")
        if not os.path.isfile(DATASET_RAW_PATH):
            raise FileNotFoundError(f"Dataset raw no encontrado: {DATASET_RAW_PATH}")
        if not os.path.isfile(DATASET_CLEAN_PATH):
            raise FileNotFoundError(f"Dataset procesado no encontrado: {DATASET_CLEAN_PATH}")

        df_raw = pd.read_csv(DATASET_RAW_PATH)
        df_clean = pd.read_csv(DATASET_CLEAN_PATH)

        if limite is not None:
            logger.warning("Se aplicará límite de carga: %d registros.", limite)
            df_raw = df_raw.head(limite).reset_index(drop=True)
            df_clean = df_clean.head(limite).reset_index(drop=True)

        logger.info(
            "Datasets cargados — Raw: %d registros | Procesado: %d registros",
            len(df_raw), len(df_clean)
        )

        resumen = {}
        resumen[COLECCION_RAW]          = cargar_clientes_raw(db, df_raw, batch_size, modo)
        resumen[COLECCION_PROCESADO]    = cargar_clientes_procesado(db, df_clean, batch_size, modo)
        resumen[COLECCION_FEATURES]     = cargar_feature_engineering(db, df_clean, batch_size, modo)
        resumen[COLECCION_METRICAS]     = cargar_metricas_dataset(db, df_raw, df_clean, modo)
        resumen[COLECCION_PREDICCIONES] = crear_coleccion_predicciones(db, modo)

        crear_indices(db)

        os.makedirs(os.path.dirname(METADATOS_PATH), exist_ok=True)
        metadatos = {
            "timestamp": datetime.now().isoformat(),
            "database_mongodb": database_name,
            "modo": modo,
            "batch_size": batch_size,
            "limite": limite,
            "colecciones_creadas": list(resumen.keys()),
            "documentos_subidos": resumen,
            "total_documentos": int(sum(resumen.values())),
        }
        with open(METADATOS_PATH, "w", encoding="utf-8") as f:
            json.dump(metadatos, f, indent=4, ensure_ascii=False)

        logger.info("=" * 60)
        logger.info("CARGA COMPLETADA EXITOSAMENTE")
        logger.info("Colecciones creadas en MongoDB:")
        for col, total in resumen.items():
            logger.info("  %-30s → %d documentos", col, total)
        logger.info("Total documentos subidos: %d", sum(resumen.values()))
        logger.info("Metadatos guardados en: %s", METADATOS_PATH)
        logger.info("Próxima etapa: ejecutar scripts/04_entrenamiento.py")
        logger.info("=" * 60)

    finally:
        if client is not None:
            client.close()
            logger.info("Conexión con MongoDB cerrada.")


# ─────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Carga de datos en MongoDB — Predicción de Depósito Bancario"
    )
    parser.add_argument(
        "--database",
        type=str,
        default=DATABASE_DEFAULT,
        help=f"Nombre de la base de datos en MongoDB (default: {DATABASE_DEFAULT})"
    )
    parser.add_argument(
        "--modo",
        choices=["reemplazar", "agregar"],
        default="reemplazar",
        help="reemplazar limpia las colecciones antes de cargar; agregar conserva datos existentes."
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Cantidad máxima de registros a cargar. Por defecto carga todo el dataset."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help=f"Tamaño del lote para insert_many (default: {BATCH_SIZE_DEFAULT})"
    )

    args = parser.parse_args()

    try:
        ejecutar_carga(
            database_name=args.database,
            modo=args.modo,
            limite=args.limite,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.critical("La carga a MongoDB falló: %s", e)
        exit(1)

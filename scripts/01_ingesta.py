"""
========================================================
Script 01 - Ingesta de Datos
Proyecto: Predicción de Suscripción a Depósito Bancario
Equipo:   Sebastián Aravena | Enrique Oporto | Alexei Sandoval
Etapa:    Pipeline - Ingesta (Actividad 2.1)
========================================================

Descripción:
    Lee el archivo CSV raw (Bank Marketing Dataset), valida su estructura
    básica, registra metadatos del proceso y lo copia al directorio
    organizado data/raw/ como punto de partida del pipeline.

Fuente de datos:
    Bank Marketing Dataset - UCI Machine Learning Repository
    Archivo: 02_bank.csv  (11.162 registros, 17 columnas)

Uso:
    python scripts/01_ingesta.py
    python scripts/01_ingesta.py --origen ruta/alternativa.csv
"""

import os
import shutil
import logging
import argparse
import hashlib
import json
from datetime import datetime

import pandas as pd


# ─────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LOGGING
# ─────────────────────────────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(
    LOG_DIR,
    f"ingesta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
# CONSTANTES DEL PROYECTO
# ─────────────────────────────────────────────────────────
ORIGEN_DEFAULT       = "data/source/02_bank.csv"
DESTINO_RAW          = "data/raw/bank_marketing.csv"
METADATOS_PATH       = "data/raw/ingesta_metadata.json"

COLUMNAS_ESPERADAS = [
    "age", "job", "marital", "education", "default",
    "balance", "housing", "loan", "contact", "day",
    "month", "duration", "campaign", "pdays", "previous",
    "poutcome", "deposit"
]

COLUMNAS_NUMERICAS   = ["age", "balance", "day", "duration",
                         "campaign", "pdays", "previous"]
COLUMNAS_CATEGORICAS = ["job", "marital", "education", "default",
                         "housing", "loan", "contact", "month",
                         "poutcome", "deposit"]


# ─────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────

def calcular_hash_md5(ruta: str) -> str:
    """Calcula el hash MD5 del archivo para verificación de integridad."""
    hash_md5 = hashlib.md5()
    with open(ruta, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def validar_estructura(df: pd.DataFrame) -> bool:
    """
    Valida que el DataFrame tenga las columnas esperadas según el
    diccionario de datos definido en el Documento de Diseño Técnico.

    Returns:
        True si la estructura es válida, False en caso contrario.
    """
    columnas_presentes = set(df.columns.tolist())
    columnas_requeridas = set(COLUMNAS_ESPERADAS)
    columnas_faltantes = columnas_requeridas - columnas_presentes

    if columnas_faltantes:
        logger.error(
            "Columnas faltantes en el dataset: %s", columnas_faltantes
        )
        return False

    columnas_extra = columnas_presentes - columnas_requeridas
    if columnas_extra:
        logger.warning(
            "Columnas adicionales no esperadas (serán ignoradas): %s",
            columnas_extra
        )

    logger.info("Validación de estructura: OK — %d columnas detectadas.",
                len(columnas_presentes))
    return True


def validar_tipos(df: pd.DataFrame) -> bool:
    """
    Verifica que las columnas numéricas contengan valores numéricos
    y que la columna target ('deposit') tenga solo 'yes'/'no'.
    """
    errores = []

    for col in COLUMNAS_NUMERICAS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errores.append(f"Columna '{col}' debería ser numérica.")

    if "deposit" in df.columns:
        valores_invalidos = df[~df["deposit"].isin(["yes", "no"])]["deposit"].unique()
        if len(valores_invalidos) > 0:
            errores.append(f"Columna 'deposit' contiene valores inválidos: {valores_invalidos}")

    if errores:
        for error in errores:
            logger.error("Error de tipo: %s", error)
        return False

    logger.info("Validación de tipos de datos: OK.")
    return True


def generar_reporte_basico(df: pd.DataFrame) -> dict:
    """
    Genera un reporte exploratorio básico del dataset ingestado
    para registrar en el log y guardar como metadato.
    """
    reporte = {
        "total_registros": int(len(df)),
        "total_columnas": int(len(df.columns)),
        "valores_nulos": int(df.isnull().sum().sum()),
        "registros_duplicados": int(df.duplicated().sum()),
        "distribucion_target": df["deposit"].value_counts().to_dict(),
        "columnas": df.columns.tolist(),
    }
    return reporte


def guardar_metadatos(origen: str, destino: str, reporte: dict, hash_md5: str):
    """
    Persiste los metadatos de la ingesta en formato JSON para
    garantizar la trazabilidad del proceso (principio DataOps).
    """
    os.makedirs(os.path.dirname(METADATOS_PATH), exist_ok=True)
    metadatos = {
        "timestamp": datetime.now().isoformat(),
        "archivo_origen": os.path.abspath(origen),
        "archivo_destino": os.path.abspath(destino),
        "hash_md5_origen": hash_md5,
        "reporte": reporte,
    }
    with open(METADATOS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=4, ensure_ascii=False)
    logger.info("Metadatos de ingesta guardados en: %s", METADATOS_PATH)


# ─────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE INGESTA
# ─────────────────────────────────────────────────────────

def ejecutar_ingesta(origen: str = ORIGEN_DEFAULT):
    """
    Ejecuta el pipeline de ingesta completo:
      1. Verificación de existencia del archivo origen
      2. Cálculo de hash MD5 (integridad)
      3. Lectura con Pandas
      4. Validación de estructura y tipos
      5. Generación de reporte básico
      6. Copia al directorio data/raw/
      7. Persistencia de metadatos (trazabilidad)
    """
    logger.info("=" * 60)
    logger.info("INICIO DE INGESTA — Predicción de Depósito Bancario")
    logger.info("Archivo origen: %s", origen)
    logger.info("=" * 60)

    # ── 1. Verificar existencia
    if not os.path.isfile(origen):
        logger.error(
            "Archivo de origen no encontrado: '%s'. "
            "Asegúrese de colocar el CSV en la ruta correcta.", origen
        )
        raise FileNotFoundError(f"No se encontró el archivo: {origen}")

    # ── 2. Hash MD5
    logger.info("Calculando hash MD5 para verificación de integridad...")
    hash_md5 = calcular_hash_md5(origen)
    logger.info("Hash MD5: %s", hash_md5)

    # ── 3. Lectura del CSV
    logger.info("Leyendo archivo CSV...")
    try:
        df = pd.read_csv(origen)
    except Exception as e:
        logger.error("Error al leer el archivo CSV: %s", e)
        raise

    logger.info(
        "Archivo leído correctamente — %d registros, %d columnas.",
        len(df), len(df.columns)
    )

    # ── 4. Validaciones
    estructura_ok = validar_estructura(df)
    tipos_ok = validar_tipos(df)

    if not estructura_ok or not tipos_ok:
        logger.error(
            "La ingesta fue ABORTADA por errores de validación. "
            "Revise el log para más detalles."
        )
        raise ValueError("El dataset no cumple las validaciones requeridas.")

    # ── 5. Reporte básico
    reporte = generar_reporte_basico(df)
    logger.info("── Reporte del dataset ──────────────────────────────")
    logger.info("  Total registros  : %d", reporte["total_registros"])
    logger.info("  Total columnas   : %d", reporte["total_columnas"])
    logger.info("  Valores nulos    : %d", reporte["valores_nulos"])
    logger.info("  Duplicados       : %d", reporte["registros_duplicados"])
    logger.info(
        "  Distribución target (deposit): yes=%s | no=%s",
        reporte["distribucion_target"].get("yes", 0),
        reporte["distribucion_target"].get("no", 0)
    )
    logger.info("────────────────────────────────────────────────────")

    # ── 6. Copia al directorio raw
    os.makedirs(os.path.dirname(DESTINO_RAW), exist_ok=True)
    shutil.copy2(origen, DESTINO_RAW)
    logger.info("Archivo copiado exitosamente a: %s", DESTINO_RAW)

    # ── 7. Metadatos
    guardar_metadatos(origen, DESTINO_RAW, reporte, hash_md5)

    logger.info("=" * 60)
    logger.info("INGESTA COMPLETADA EXITOSAMENTE")
    logger.info("Próxima etapa: ejecutar scripts/02_limpieza_transformacion.py")
    logger.info("=" * 60)

    return df


# ─────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script de ingesta — Predicción de Depósito Bancario"
    )
    parser.add_argument(
        "--origen",
        type=str,
        default=ORIGEN_DEFAULT,
        help=f"Ruta al CSV de origen (default: {ORIGEN_DEFAULT})"
    )
    args = parser.parse_args()

    try:
        ejecutar_ingesta(origen=args.origen)
    except (FileNotFoundError, ValueError) as e:
        logger.critical("El proceso de ingesta falló: %s", e)
        exit(1)
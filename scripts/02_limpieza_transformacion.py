"""
========================================================
Script 02 - Limpieza, Transformación e Ingeniería de Características
Proyecto: Predicción de Suscripción a Depósito Bancario
Equipo:   Sebastián Aravena | Enrique Oporto | Alexei Sandoval
Etapa:    Pipeline - Limpieza y Transformación (Actividad 2.2)
========================================================

Descripción:
    Aplica reglas de limpieza, codificación de variables categóricas,
    escalado de variables numéricas y creación de nuevas features
    sobre el dataset raw ingestado en la etapa anterior.

    El dataset limpio se guarda en data/processed/ listo para ser
    cargado en Firebase Firestore (siguiente etapa del pipeline).

Reglas de limpieza aplicadas:
    1.  Eliminación de filas duplicadas
    2.  Verificación de valores nulos
    3.  Detección de valores atípicos (IQR) en variables numéricas
    4.  Estandarización de strings (lowercase + strip)
    5.  Codificación binaria (yes/no → 1/0) para columnas binarias
    6.  Codificación ordinal de 'education'
    7.  One-Hot Encoding para variables categóricas nominales
    8.  Creación de feature 'contacted_before' (pdays != -1)
    9.  Escalado estándar (StandardScaler) para variables numéricas
    10. Recodificación de la variable target 'deposit' (yes=1, no=0)

Uso:
    python scripts/02_limpieza_transformacion.py
    python scripts/02_limpieza_transformacion.py --entrada ruta/raw.csv
"""

import os
import json
import logging
import argparse
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LOGGING
# ─────────────────────────────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(
    LOG_DIR,
    f"limpieza_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
ENTRADA_DEFAULT  = "data/raw/bank_marketing.csv"
SALIDA_PROCESADO = "data/processed/bank_clean.csv"
SCALER_PATH      = "models/scaler.pkl"
METADATOS_PATH   = "data/processed/limpieza_metadata.json"

# Variables binarias (yes/no → 1/0)
COLS_BINARIAS = ["default", "housing", "loan"]

# Orden educativo para codificación ordinal
ORDEN_EDUCACION = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}

# Variables categóricas nominales para One-Hot Encoding
COLS_OHE = ["job", "marital", "contact", "month", "poutcome"]

# Variables numéricas a escalar
COLS_NUMERICAS = ["age", "balance", "day", "duration",
                  "campaign", "pdays", "previous"]


# ─────────────────────────────────────────────────────────
# PASO 1 — CARGA DEL DATASET RAW
# ─────────────────────────────────────────────────────────

def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga el CSV desde data/raw y registra información básica."""
    if not os.path.isfile(ruta):
        logger.error("Archivo de entrada no encontrado: %s", ruta)
        raise FileNotFoundError(
            f"Ejecute primero 01_ingesta.py. Archivo esperado: {ruta}"
        )

    df = pd.read_csv(ruta)
    logger.info(
        "Dataset raw cargado — %d registros, %d columnas.", len(df), len(df.columns)
    )
    return df


# ─────────────────────────────────────────────────────────
# PASO 2 — LIMPIEZA DE DATOS
# ─────────────────────────────────────────────────────────

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica reglas de limpieza al dataset:
    - Duplicados
    - Nulos
    - Estandarización de strings
    - Atípicos (registro, sin eliminación agresiva)
    """
    registros_inicial = len(df)
    logger.info("── LIMPIEZA DE DATOS ────────────────────────────────")

    # 2.1 Eliminar duplicados
    df = df.drop_duplicates()
    eliminados = registros_inicial - len(df)
    logger.info("  Registros duplicados eliminados: %d", eliminados)

    # 2.2 Valores nulos
    nulos = df.isnull().sum()
    if nulos.sum() == 0:
        logger.info("  Valores nulos: ninguno detectado. ✓")
    else:
        logger.warning("  Valores nulos detectados:\n%s", nulos[nulos > 0])
        # Imputación de columnas numéricas con mediana
        for col in COLS_NUMERICAS:
            if df[col].isnull().sum() > 0:
                mediana = df[col].median()
                df[col].fillna(mediana, inplace=True)
                logger.info("    Columna '%s': nulos imputados con mediana (%.2f).", col, mediana)
        # Imputación de columnas categóricas con moda
        for col in COLS_OHE + COLS_BINARIAS + ["education"]:
            if df[col].isnull().sum() > 0:
                moda = df[col].mode()[0]
                df[col].fillna(moda, inplace=True)
                logger.info("    Columna '%s': nulos imputados con moda ('%s').", col, moda)

    # 2.3 Estandarizar strings (minúsculas + sin espacios)
    cols_str = df.select_dtypes(include="object").columns.tolist()
    for col in cols_str:
        df[col] = df[col].str.lower().str.strip()
    logger.info("  Strings normalizados (lowercase + strip) en %d columnas.", len(cols_str))

    # 2.4 Reportar atípicos con IQR (sin eliminar — se documenta para el equipo)
    logger.info("  Detección de atípicos (IQR) en variables numéricas:")
    for col in ["age", "balance", "duration", "campaign"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        atipicos = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        logger.info(
            "    %s: %d atípicos detectados (%.1f%% del total).",
            col, len(atipicos), 100 * len(atipicos) / len(df)
        )

    logger.info(
        "  Registros tras limpieza: %d (se eliminaron %d).",
        len(df), registros_inicial - len(df)
    )
    return df


# ─────────────────────────────────────────────────────────
# PASO 3 — CODIFICACIÓN DE VARIABLES
# ─────────────────────────────────────────────────────────

def codificar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica las variables categóricas:
    - Binarias (yes/no → 1/0)
    - Ordinal (education)
    - Nominal (One-Hot Encoding)
    - Target (deposit: yes=1, no=0)
    """
    logger.info("── CODIFICACIÓN DE VARIABLES ────────────────────────")

    # 3.1 Variables binarias
    for col in COLS_BINARIAS:
        df[col] = df[col].map({"yes": 1, "no": 0})
        logger.info("  '%s' codificada: yes→1, no→0.", col)

    # 3.2 Education (ordinal)
    df["education"] = df["education"].map(ORDEN_EDUCACION)
    logger.info(
        "  'education' codificada ordinalmente: %s.", ORDEN_EDUCACION
    )

    # 3.3 One-Hot Encoding para variables nominales
    df = pd.get_dummies(df, columns=COLS_OHE, drop_first=False, dtype=int)
    nuevas_cols = [c for c in df.columns if any(c.startswith(f"{col}_") for col in COLS_OHE)]
    logger.info(
        "  One-Hot Encoding aplicado a %s → %d nuevas columnas creadas.",
        COLS_OHE, len(nuevas_cols)
    )

    # 3.4 Target
    df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})
    logger.info(
        "  'deposit' (target) codificada: yes→1, no→0. "
        "Distribución: %s",
        df["deposit"].value_counts().to_dict()
    )

    return df


# ─────────────────────────────────────────────────────────
# PASO 4 — INGENIERÍA DE CARACTERÍSTICAS
# ─────────────────────────────────────────────────────────

def ingenieria_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevas características a partir de las variables existentes
    para mejorar la capacidad predictiva del modelo.

    Features creadas:
    - contacted_before : 1 si el cliente fue contactado en campañas anteriores
    - balance_bin      : categoría de saldo (negativo, bajo, medio, alto)
    - contact_intensity: ratio duration / campaign (segundos por contacto)
    - age_group        : grupo etario del cliente
    """
    logger.info("── INGENIERÍA DE CARACTERÍSTICAS ────────────────────")

    # 4.1 ¿Fue contactado antes de esta campaña?
    df["contacted_before"] = (df["pdays"] != -1).astype(int)
    logger.info(
        "  'contacted_before' creada — %d clientes contactados previamente.",
        df["contacted_before"].sum()
    )

    # 4.2 Categoría de saldo bancario
    df["balance_bin"] = pd.cut(
        df["balance"],
        bins=[-np.inf, 0, 500, 2000, np.inf],
        labels=[0, 1, 2, 3]  # negativo, bajo, medio, alto
    ).astype(int)
    logger.info(
        "  'balance_bin' creada (0=negativo, 1=bajo, 2=medio, 3=alto): %s",
        df["balance_bin"].value_counts().sort_index().to_dict()
    )

    # 4.3 Intensidad de contacto (segundos / nro. contactos)
    df["contact_intensity"] = (df["duration"] / df["campaign"]).round(2)
    logger.info("  'contact_intensity' creada (duration / campaign).")

    # 4.4 Grupo etario
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 55, 100],
        labels=[0, 1, 2, 3]   # joven, adulto, maduro, mayor
    ).astype(int)
    logger.info(
        "  'age_group' creada (0=joven≤30, 1=adulto≤40, 2=maduro≤55, 3=mayor): %s",
        df["age_group"].value_counts().sort_index().to_dict()
    )

    logger.info("  Total de columnas tras feature engineering: %d.", len(df.columns))
    return df


# ─────────────────────────────────────────────────────────
# PASO 5 — ESCALADO DE VARIABLES NUMÉRICAS
# ─────────────────────────────────────────────────────────

def escalar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica StandardScaler a las variables numéricas originales.
    El scaler se persiste en models/scaler.pkl para ser reutilizado
    en la predicción sin reentrenamiento (principio de reproducibilidad).
    """
    logger.info("── ESCALADO DE VARIABLES NUMÉRICAS ─────────────────")

    # Solo escalar columnas que existen aún como numéricas continuas
    cols_a_escalar = [c for c in COLS_NUMERICAS if c in df.columns]

    scaler = StandardScaler()
    df[cols_a_escalar] = scaler.fit_transform(df[cols_a_escalar])

    # Persistir el scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    logger.info(
        "  StandardScaler aplicado a: %s.", cols_a_escalar
    )
    logger.info("  Scaler guardado en: %s (para uso en producción).", SCALER_PATH)
    return df


# ─────────────────────────────────────────────────────────
# PASO 6 — GUARDADO Y METADATOS
# ─────────────────────────────────────────────────────────

def guardar_dataset_limpio(df: pd.DataFrame):
    """Guarda el dataset procesado en data/processed/."""
    os.makedirs(os.path.dirname(SALIDA_PROCESADO), exist_ok=True)
    df.to_csv(SALIDA_PROCESADO, index=False)
    logger.info("Dataset limpio guardado en: %s", SALIDA_PROCESADO)


def guardar_metadatos(df: pd.DataFrame, df_original_len: int):
    """Guarda los metadatos del proceso de limpieza para trazabilidad."""
    os.makedirs(os.path.dirname(METADATOS_PATH), exist_ok=True)
    metadatos = {
        "timestamp": datetime.now().isoformat(),
        "registros_entrada": df_original_len,
        "registros_salida": len(df),
        "columnas_salida": df.columns.tolist(),
        "total_columnas": len(df.columns),
        "distribucion_target": df["deposit"].value_counts().to_dict(),
        "archivo_salida": os.path.abspath(SALIDA_PROCESADO),
        "scaler": os.path.abspath(SCALER_PATH),
        "transformaciones_aplicadas": [
            "eliminacion_duplicados",
            "verificacion_nulos",
            "normalizacion_strings",
            "codificacion_binaria",
            "codificacion_ordinal_education",
            "one_hot_encoding",
            "ingenieria_features",
            "standard_scaler",
            "target_encoding"
        ]
    }
    with open(METADATOS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=4, ensure_ascii=False)
    logger.info("Metadatos de limpieza guardados en: %s", METADATOS_PATH)


# ─────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────

def ejecutar_limpieza(entrada: str = ENTRADA_DEFAULT):
    """
    Ejecuta el pipeline completo de limpieza y transformación:
      1. Carga del dataset raw
      2. Limpieza (duplicados, nulos, strings, atípicos)
      3. Codificación de variables
      4. Ingeniería de características
      5. Escalado estándar
      6. Guardado + metadatos
    """
    logger.info("=" * 60)
    logger.info("INICIO — Limpieza y Transformación")
    logger.info("Archivo de entrada: %s", entrada)
    logger.info("=" * 60)

    # Paso 1 — Carga
    df = cargar_datos(entrada)
    len_original = len(df)

    # Paso 2 — Limpieza
    df = limpiar_datos(df)

    # Paso 3 — Codificación
    df = codificar_variables(df)

    # Paso 4 — Ingeniería de características
    df = ingenieria_features(df)

    # Paso 5 — Escalado
    df = escalar_variables(df)

    # Paso 6 — Guardado
    guardar_dataset_limpio(df)
    guardar_metadatos(df, len_original)

    logger.info("=" * 60)
    logger.info("LIMPIEZA Y TRANSFORMACIÓN COMPLETADA EXITOSAMENTE")
    logger.info(
        "Dataset final: %d registros | %d columnas", len(df), len(df.columns)
    )
    logger.info("Próxima etapa: ejecutar scripts/03_carga_firebase.py")
    logger.info("=" * 60)

    return df


# ─────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Limpieza y transformación — Predicción de Depósito Bancario"
    )
    parser.add_argument(
        "--entrada",
        type=str,
        default=ENTRADA_DEFAULT,
        help=f"Ruta al CSV raw (default: {ENTRADA_DEFAULT})"
    )
    args = parser.parse_args()

    try:
        ejecutar_limpieza(entrada=args.entrada)
    except (FileNotFoundError, ValueError) as e:
        logger.critical("El proceso de limpieza falló: %s", e)
        exit(1)
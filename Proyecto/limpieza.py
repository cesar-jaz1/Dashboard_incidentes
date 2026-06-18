"""
limpieza.py
===========
Fase 1 del proyecto: Limpieza y preparación del dataset de incidentes viales CDMX.
Dataset fuente: inViales_2022_2024.csv  (504,261 registros, 17 columnas)

Decisiones de limpieza documentadas:
  1. Eliminar incidentes no viales (Mi Taxi, Sismo, Mi Calle, Detención ciudadana)
  2. Eliminar registros con fecha_creacion en 2021 (73 registros fuera del rango del proyecto)
  3. Eliminar duraciones negativas (3 registros con inconsistencia fecha_creacion > fecha_cierre)
  4. Marcar como outlier (no eliminar) las duraciones > 24h (23,041 registros)
  5. Imputar colonia_catalogo nula con 'Sin colonia' (11,176 registros, 2.22%)
  6. Imputar tipo_entrada nula con 'DESCONOCIDO' (5 registros)
  7. Seleccionar solo las 13 columnas útiles para el proyecto (descartar alcaldia_inicio, alcaldia_cierre, folio no es util para análisis)
  8. Crear columnas derivadas: anio, mes, hora, minutos_atencion, es_grave, nivel_severidad
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────

# Directorio donde vive este script (funciona sin importar desde dónde lo corras)
BASE_DIR = Path(__file__).parent

RUTA_ENTRADA = BASE_DIR / "datos" / "inViales_2022_2024.csv"
RUTA_SALIDA  = BASE_DIR / "datos" / "viales_limpio.csv"
RUTA_PARQUET = BASE_DIR / "datos" / "viales_limpio.parquet"

# Columnas que sí necesita el dashboard/predictor (versión ligera para el deploy)
COLUMNAS_PARQUET = [
    'tipo_incidente_c4', 'incidente_c4', 'codigo_cierre', 'tipo_entrada',
    'clas_con_f_alarma', 'alcaldia_catalogo', 'colonia_catalogo',
    'longitud', 'latitud', 'minutos_atencion', 'anio', 'mes', 'hora',
    'dia_semana', 'es_fin_semana', 'nivel_severidad', 'es_grave', 'atencion_outlier',
]

# Tipos de incidente que NO son viales y se excluyen del análisis
TIPOS_NO_VIALES = {'Mi Taxi', 'Detención ciudadana', 'Sismo', 'Mi Calle'}

# Bbox de CDMX para validar coordenadas
LAT_MIN, LAT_MAX = 19.05, 19.65
LON_MIN, LON_MAX = -99.40, -98.90

# Columnas que sí usamos (las demás se descartan)
COLUMNAS_UTILES = [
    'folio',
    'fecha_creacion',
    'hora_creacion',
    'dia_semana',
    'fecha_cierre',
    'hora_cierre',
    'tipo_incidente_c4',
    'incidente_c4',
    'codigo_cierre',
    'clas_con_f_alarma',
    'tipo_entrada',
    'alcaldia_catalogo',
    'colonia_catalogo',
    'longitud',
    'latitud',
]


# ──────────────────────────────────────────────
# FUNCIONES
# ──────────────────────────────────────────────

def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga el CSV y reporta dimensiones iniciales."""
    df = pd.read_csv(ruta)
    print(f"[CARGA]  {len(df):>7,} registros | {len(df.columns)} columnas")
    return df


def seleccionar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva solo las columnas relevantes para el proyecto."""
    df = df[COLUMNAS_UTILES].copy()
    print(f"[COLS]   Columnas reducidas a {len(df.columns)}: {COLUMNAS_UTILES}")
    return df


def parsear_fechas(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte strings de fecha/hora a tipos datetime y crea dt_creacion / dt_cierre."""
    df['fecha_creacion'] = pd.to_datetime(df['fecha_creacion'], dayfirst=True, errors='coerce')
    df['fecha_cierre']   = pd.to_datetime(df['fecha_cierre'],   dayfirst=True, errors='coerce')

    df['dt_creacion'] = pd.to_datetime(
        df['fecha_creacion'].dt.strftime('%Y-%m-%d') + ' ' + df['hora_creacion'],
        errors='coerce'
    )
    df['dt_cierre'] = pd.to_datetime(
        df['fecha_cierre'].dt.strftime('%Y-%m-%d') + ' ' + df['hora_cierre'],
        errors='coerce'
    )
    print(f"[FECHAS] Parseadas. Nulos en dt_creacion: {df['dt_creacion'].isnull().sum()}")
    return df


def filtrar_registros(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todos los filtros de exclusión y reporta cuántos se eliminan."""
    n0 = len(df)

    # 1. Excluir tipos no viales
    mask_no_viales = df['tipo_incidente_c4'].isin(TIPOS_NO_VIALES)
    df = df[~mask_no_viales]
    print(f"[FILTRO] No viales eliminados:           {mask_no_viales.sum():>6,}")

    # 2. Excluir registros de 2021 (fuera del rango del proyecto)
    mask_2021 = df['fecha_creacion'].dt.year == 2021
    df = df[~mask_2021]
    print(f"[FILTRO] Registros 2021 eliminados:      {mask_2021.sum():>6,}")

    # 3. Excluir duraciones negativas (fecha_cierre < fecha_creacion)
    df['minutos_atencion'] = (df['dt_cierre'] - df['dt_creacion']).dt.total_seconds() / 60
    mask_neg = df['minutos_atencion'] < 0
    df = df[~mask_neg]
    print(f"[FILTRO] Duraciones negativas eliminadas:{mask_neg.sum():>6,}")

    print(f"[FILTRO] Total eliminados: {n0 - len(df):,} | Quedan: {len(df):,}")
    return df


def imputar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa valores nulos con categorías centinela documentadas."""
    n_colonia   = df['colonia_catalogo'].isnull().sum()
    n_entrada   = df['tipo_entrada'].isnull().sum()
    n_alcaldia  = df['alcaldia_catalogo'].isnull().sum()

    df['colonia_catalogo']  = df['colonia_catalogo'].fillna('Sin colonia')
    df['tipo_entrada']      = df['tipo_entrada'].fillna('DESCONOCIDO')
    df['alcaldia_catalogo'] = df['alcaldia_catalogo'].fillna('Sin alcaldía')

    print(f"[NULOS]  colonia imputada:  {n_colonia:,}  |  tipo_entrada: {n_entrada}  |  alcaldía: {n_alcaldia}")
    return df


def crear_columnas_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea columnas nuevas útiles para el análisis y para los métodos de las clases POO.

    Nuevas columnas:
      - anio           : año del incidente (2022 / 2023 / 2024)
      - mes            : mes numérico (1–12)
      - hora           : hora de creación (0–23)
      - es_fin_semana  : True si día es Sábado o Domingo
      - es_grave       : True si tipo es Cadáver o Lesionado
      - nivel_severidad: 0=Accidente leve, 1=Lesionado, 2=Cadáver
      - atencion_outlier: True si duración > 1440 min (>24h)
    """
    df['anio']          = df['fecha_creacion'].dt.year
    df['mes']           = df['fecha_creacion'].dt.month
    df['hora']          = df['dt_creacion'].dt.hour

    df['es_fin_semana'] = df['dia_semana'].isin(['Sábado', 'Domingo'])

    # Nivel de severidad basado en tipo_incidente_c4
    severidad_map = {'Accidente': 0, 'Lesionado': 1, 'Cadáver': 2}
    df['nivel_severidad'] = df['tipo_incidente_c4'].map(severidad_map).fillna(0).astype(int)
    df['es_grave']        = df['nivel_severidad'] >= 1

    # Marcar outliers de duración (se conservan pero se etiquetan)
    df['atencion_outlier'] = df['minutos_atencion'] > 1440

    print(f"[COLS+]  Columnas derivadas creadas: anio, mes, hora, es_fin_semana, "
          f"nivel_severidad, es_grave, atencion_outlier")
    print(f"         Incidentes graves: {df['es_grave'].sum():,} "
          f"({df['es_grave'].mean()*100:.1f}%)")
    print(f"         Outliers duración: {df['atencion_outlier'].sum():,} "
          f"({df['atencion_outlier'].mean()*100:.1f}%)")
    return df


def guardar(df: pd.DataFrame, ruta: str) -> None:
    """Guarda el dataset limpio en CSV (versión completa, para el análisis OOP)."""
    df.to_csv(ruta, index=False)
    print(f"\n[LISTO]  Dataset limpio guardado en: {ruta}")
    print(f"         Dimensiones finales: {df.shape[0]:,} filas × {df.shape[1]} columnas")


def guardar_parquet(df: pd.DataFrame, ruta: Path = RUTA_PARQUET) -> None:
    """
    Guarda una versión LIGERA en Parquet con tipos optimizados.
    El CSV completo pesa ~128 MB (no cabe en GitHub); este Parquet ronda los
    ~9 MB, así que es el que usa el dashboard y el que se sube para el deploy.
    """
    d = df[COLUMNAS_PARQUET].copy()
    for c in ['tipo_incidente_c4', 'incidente_c4', 'codigo_cierre', 'tipo_entrada',
              'clas_con_f_alarma', 'alcaldia_catalogo', 'colonia_catalogo', 'dia_semana']:
        d[c] = d[c].astype('category')
    for c in ['es_grave', 'es_fin_semana', 'atencion_outlier']:
        d[c] = d[c].astype(bool)
    d['hora'] = d['hora'].astype('int8')
    d['mes']  = d['mes'].astype('int8')
    d['anio'] = d['anio'].astype('int16')
    d['nivel_severidad']  = d['nivel_severidad'].astype('int8')
    d['minutos_atencion'] = d['minutos_atencion'].astype('float32')

    d.to_parquet(ruta, index=False, compression='snappy')
    mb = ruta.stat().st_size / 1e6
    print(f"[PARQUET] Versión ligera (deploy) guardada en: {ruta}  ({mb:.1f} MB)")


def reporte_final(df: pd.DataFrame) -> None:
    """Imprime un resumen ejecutivo del dataset limpio."""
    print("\n" + "="*55)
    print("  RESUMEN DEL DATASET LIMPIO")
    print("="*55)
    print(f"  Período:        {df['anio'].min()} – {df['anio'].max()}")
    print(f"  Registros:      {len(df):,}")
    print(f"  Alcaldías:      {df['alcaldia_catalogo'].nunique()}")
    print(f"  Colonias:       {df['colonia_catalogo'].nunique()}")
    print(f"  Tipos incidente:{df['tipo_incidente_c4'].nunique()}")
    print()
    print("  Distribución por tipo:")
    for tipo, cnt in df['tipo_incidente_c4'].value_counts().items():
        print(f"    {tipo:<20} {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")
    print()
    print("  Duración promedio de atención: "
          f"{df[~df['atencion_outlier']]['minutos_atencion'].mean():.1f} min")
    print("="*55)


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def limpiar(ruta_entrada: Path = RUTA_ENTRADA,
            ruta_salida:  Path = RUTA_SALIDA) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de limpieza.
    Retorna el DataFrame limpio (útil para importar desde notebooks).
    """
    print("╔══════════════════════════════════════════╗")
    print("║   PIPELINE DE LIMPIEZA — VIALES CDMX     ║")
    print("╚══════════════════════════════════════════╝\n")

    df = cargar_datos(ruta_entrada)
    df = seleccionar_columnas(df)
    df = parsear_fechas(df)
    df = filtrar_registros(df)
    df = imputar_nulos(df)
    df = crear_columnas_derivadas(df)
    guardar(df, ruta_salida)
    guardar_parquet(df)
    reporte_final(df)

    return df


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    limpiar()

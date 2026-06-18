"""
descriptivo.py
==============
Exportador de figuras ESTÁTICAS (PNG) del análisis descriptivo, para pegar
en el reporte escrito. Son la contraparte fija de los 6 análisis que el
tablero (app.py) muestra de forma interactiva:

  D1 · Pastel  — distribución de los 7 tipos de incidente (usa CatalogoTipos)
  D2 · Mapa    — densidad espacial comparando dos horas (hexbin)
  D3 · Líneas  — incidentes por hora del día (leve vs grave)
  D4 · Barras  — ranking de alcaldías por volumen y por gravedad
  D5 · Disp.   — tiempo de atención vs volumen por alcaldía
  D6 · Columnas— tendencia mensual comparada por año

Trabaja de forma vectorizada sobre el DataFrame (rápido) y reutiliza las
clases de modelos.py para el conocimiento agregado.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from modelos import CatalogoTipos, agregar_categoria, CATEGORIAS_7

# ── Configuración ─────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
RUTA_PARQUET  = BASE_DIR / "datos" / "viales_limpio.parquet"
RUTA_CSV      = BASE_DIR / "datos" / "viales_limpio.csv"
RUTA_GRAFICAS = BASE_DIR / "graficas"
RUTA_GRAFICAS.mkdir(exist_ok=True)

COL_PRINCIPAL = "#D85A30"
COL_GRAVE     = "#A32D2D"
COL_LEVE      = "#378ADD"
SECUENCIA_7   = ["#D85A30", "#E8833F", "#378ADD", "#5FA8E0",
                 "#A32D2D", "#E0A458", "#B4B2A9"]
MESES = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
         7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}

plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "font.family": "sans-serif",
})


def cargar_df() -> pd.DataFrame:
    df = pd.read_parquet(RUTA_PARQUET) if RUTA_PARQUET.exists() else pd.read_csv(RUTA_CSV)
    return agregar_categoria(df)


# ── D1 · Pastel de tipos (vía CatalogoTipos) ───────────────

def fig_pastel(df: pd.DataFrame) -> None:
    cat = CatalogoTipos.desde_dataframe(df)
    tipos = cat.tipos()
    nombres = [t.nombre for t in tipos]
    valores = [t.total for t in tipos]

    fig, ax = plt.subplots(figsize=(8, 6))
    explode = [0.06 if i < 3 else 0 for i in range(len(nombres))]
    ax.pie(valores, labels=nombres, colors=SECUENCIA_7, explode=explode,
           autopct=lambda p: f"{p:.1f}%" if p > 2 else "", startangle=90,
           textprops={'fontsize': 9}, wedgeprops={'width': 0.65})
    ax.set_title("Distribución de incidentes por tipo (toda la CDMX)")
    top3 = ", ".join(f"{t.nombre} ({t.porcentaje():.1f}%)" for t in cat.top(3))
    ax.text(0, -1.35, f"Predominantes: {top3}", ha='center', fontsize=9, color='#444')
    fig.tight_layout()
    fig.savefig(RUTA_GRAFICAS / "D1_pastel_tipos.png")
    plt.close()
    print("  D1_pastel_tipos.png")


# ── D2 · Mapa de densidad por hora (hexbin, 2 horas) ───────

def fig_mapa(df: pd.DataFrame) -> None:
    d = df.dropna(subset=['latitud', 'longitud'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, hora in zip(axes, [8, 19]):
        sub = d[d['hora'] == hora]
        hb = ax.hexbin(sub['longitud'], sub['latitud'], gridsize=55,
                       cmap='inferno', mincnt=1)
        ax.set_title(f"Densidad de incidentes — {hora}:00 h")
        ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
        ax.set_aspect('equal', adjustable='box')
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="Incidentes")
    fig.suptitle("Los focos de incidentes cambian con la hora del día",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(RUTA_GRAFICAS / "D2_mapa_hora.png")
    plt.close()
    print("  D2_mapa_hora.png")


# ── D3 · Incidentes por hora del día (líneas) ──────────────

def fig_hora(df: pd.DataFrame) -> None:
    por_hora = (df.assign(g=df['es_grave'])
                  .groupby('hora')
                  .agg(total=('g', 'size'), graves=('g', 'sum'))
                  .reindex(range(24), fill_value=0))
    leves = por_hora['total'] - por_hora['graves']

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(por_hora.index, leves, color=COL_LEVE, alpha=0.85, label='Leve')
    ax.fill_between(por_hora.index, por_hora['total'], leves,
                    color=COL_GRAVE, alpha=0.9, label='Grave')
    ax.plot(por_hora.index, por_hora['total'], color="#333", linewidth=1.5)
    ax.set_title("Incidentes por hora del día")
    ax.set_xlabel("Hora"); ax.set_ylabel("Incidentes")
    ax.set_xticks(range(0, 24, 2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUTA_GRAFICAS / "D3_hora_dia.png")
    plt.close()
    print("  D3_hora_dia.png")


# ── D4 · Ranking de alcaldías (barras) ─────────────────────

def fig_ranking(df: pd.DataFrame) -> None:
    g = (df.groupby('alcaldia_catalogo', observed=True)
           .agg(total=('es_grave', 'size'), tasa=('es_grave', 'mean'))
           .reset_index())
    g = g[g['alcaldia_catalogo'] != 'Sin alcaldía']
    g['tasa'] *= 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    d1 = g.sort_values('total')
    ax1.barh(d1['alcaldia_catalogo'], d1['total'], color=COL_PRINCIPAL)
    ax1.set_title("Volumen total de incidentes"); ax1.set_xlabel("Incidentes")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    d2 = g.sort_values('tasa')
    ax2.barh(d2['alcaldia_catalogo'], d2['tasa'], color=COL_GRAVE)
    ax2.set_title("Tasa de incidentes graves (%)"); ax2.set_xlabel("% graves")

    fig.suptitle("Ranking de alcaldías — CDMX 2022–2024", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(RUTA_GRAFICAS / "D4_ranking_alcaldias.png")
    plt.close()
    print("  D4_ranking_alcaldias.png")


# ── D5 · Dispersión: atención vs volumen ───────────────────

def fig_dispersion(df: pd.DataFrame) -> None:
    base = df[~df['atencion_outlier']]
    g = (base.groupby('alcaldia_catalogo', observed=True)
             .agg(total=('es_grave', 'size'),
                  tiempo=('minutos_atencion', 'mean'),
                  tasa=('es_grave', 'mean'))
             .reset_index())
    g = g[g['alcaldia_catalogo'] != 'Sin alcaldía']
    g['tasa'] *= 100

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(g['total'], g['tiempo'], s=g['total'] / 200,
                    c=g['tasa'], cmap='OrRd', edgecolors='#555', linewidths=0.5)
    for _, r in g.iterrows():
        ax.annotate(r['alcaldia_catalogo'], (r['total'], r['tiempo']),
                    fontsize=7.5, textcoords='offset points', xytext=(5, 3))
    ax.set_title("Tiempo de atención vs volumen por alcaldía")
    ax.set_xlabel("Incidentes"); ax.set_ylabel("Minutos de atención (sin outliers)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.colorbar(sc, ax=ax, label="% graves")
    fig.tight_layout()
    fig.savefig(RUTA_GRAFICAS / "D5_dispersion_atencion.png")
    plt.close()
    print("  D5_dispersion_atencion.png")


# ── D6 · Tendencia mensual por año (columnas) ──────────────

def fig_tendencia(df: pd.DataFrame) -> None:
    g = df.groupby(['anio', 'mes'], observed=True).size().reset_index(name='total')
    anios = sorted(g['anio'].unique())
    colores = {a: c for a, c in zip(anios, [COL_LEVE, COL_PRINCIPAL, COL_GRAVE])}
    x = np.arange(1, 13)
    ancho = 0.8 / len(anios)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, anio in enumerate(anios):
        sub = g[g['anio'] == anio].set_index('mes')['total'].reindex(range(1, 13), fill_value=0)
        ax.bar(x + (i - len(anios) / 2 + 0.5) * ancho, sub.values, ancho,
               label=str(anio), color=colores.get(anio, "#888"))
    ax.set_title("Tendencia mensual de incidentes por año")
    ax.set_xlabel("Mes"); ax.set_ylabel("Incidentes")
    ax.set_xticks(x); ax.set_xticklabels([MESES[m] for m in range(1, 13)])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(title="Año")
    fig.tight_layout()
    fig.savefig(RUTA_GRAFICAS / "D6_tendencia_anual.png")
    plt.close()
    print("  D6_tendencia_anual.png")


# ── Resumen (calculado, no hardcodeado) ────────────────────

def resumen(df: pd.DataFrame) -> None:
    g = (df.groupby('alcaldia_catalogo', observed=True)
           .agg(total=('es_grave', 'size'), tasa=('es_grave', 'mean')))
    g = g[g.index != 'Sin alcaldía']
    top_vol = g['total'].idxmax()
    top_grave = (g['tasa'] * 100).idxmax()

    print("\n" + "═" * 55)
    print("  RESUMEN — ANÁLISIS DESCRIPTIVO")
    print("═" * 55)
    print(f"  Incidentes analizados : {len(df):,}")
    print(f"  Incidentes graves     : {df['es_grave'].sum():,} ({df['es_grave'].mean()*100:.1f}%)")
    print(f"  Hora pico             : {int(df['hora'].value_counts().idxmax())}:00 h")
    print(f"  Alcaldía + incidentes : {top_vol} ({g.loc[top_vol,'total']:,})")
    print(f"  Mayor tasa de graves  : {top_grave} ({g.loc[top_grave,'tasa']*100:.1f}%)")
    print("═" * 55)


# ── Pipeline ──────────────────────────────────────────────

def analizar() -> None:
    print("╔══════════════════════════════════════════╗")
    print("║  ANÁLISIS DESCRIPTIVO — VIALES CDMX      ║")
    print("╚══════════════════════════════════════════╝\n")
    df = cargar_df()
    print(f"[DATOS] {len(df):,} registros\n[FIGURAS] Exportando a graficas/...")
    fig_pastel(df)
    fig_mapa(df)
    fig_hora(df)
    fig_ranking(df)
    fig_dispersion(df)
    fig_tendencia(df)
    resumen(df)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    analizar()

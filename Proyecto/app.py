"""
app.py — Dashboard interactivo de Incidentes Viales CDMX 2022–2024
=================================================================
Tablero construido con Streamlit + Plotly. Pensado para desplegarse en
Streamlit Community Cloud: el usuario solo abre la URL, no instala nada.

Estructura (descriptivo → predictivo, como pidió el profe):
  • Inicio        — qué es y por qué analizamos así
  • Dataset       — ficha y explorador de registros
  • Descriptivo   — 6 análisis interactivos (pastel, mapa, líneas,
                    barras, dispersión, columnas), cada uno con su "por qué"
  • Predictivo    — alcaldía + hora → tipo de accidente más probable

El conocimiento (agrupación en 7 tipos, distribuciones) vive en modelos.py
y lo reutilizan tanto el descriptivo como el predictivo.

Uso local:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modelos import (
    CATEGORIAS_7, ALCALDIAS_CDMX, agregar_categoria, CatalogoTipos,
)
from predictivo import PredictorTipoAccidente

# ── Configuración general ──────────────────────────────────
st.set_page_config(page_title="Incidentes Viales CDMX",
                   page_icon="🚦", layout="wide")

BASE_DIR     = Path(__file__).parent
RUTA_PARQUET = BASE_DIR / "datos" / "viales_limpio.parquet"
RUTA_CSV     = BASE_DIR / "datos" / "viales_limpio.csv"

# Paleta del proyecto
COL_PRINCIPAL = "#D85A30"
COL_GRAVE     = "#A32D2D"
COL_LEVE      = "#378ADD"
SECUENCIA_7   = ["#D85A30", "#E8833F", "#378ADD", "#5FA8E0",
                 "#A32D2D", "#E0A458", "#B4B2A9"]

MESES = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
         7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}
DIAS_ORDEN = ["Lunes", "Martes", "Miércoles", "Jueves",
              "Viernes", "Sábado", "Domingo"]


# ══════════════════════════════════════════════════════════
#  CARGA DE DATOS Y MODELO (en caché)
# ══════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Cargando datos...")
def cargar_datos() -> pd.DataFrame:
    if RUTA_PARQUET.exists():
        df = pd.read_parquet(RUTA_PARQUET)
    elif RUTA_CSV.exists():
        df = pd.read_csv(RUTA_CSV)
    else:
        return pd.DataFrame()
    return agregar_categoria(df)


@st.cache_resource(show_spinner="Entrenando predictor...")
def cargar_predictor(_df: pd.DataFrame) -> PredictorTipoAccidente:
    return PredictorTipoAccidente().entrenar(_df)


def porque(texto: str) -> None:
    """Cajita de '¿por qué este análisis / este gráfico?'."""
    with st.expander("¿Por qué este análisis y este tipo de gráfico?"):
        st.markdown(texto)


# ══════════════════════════════════════════════════════════
#  PÁGINA: INICIO
# ══════════════════════════════════════════════════════════

def pagina_inicio(df: pd.DataFrame) -> None:
    st.title("🚦 Incidentes Viales CDMX · 2022–2024")
    st.markdown(
        "Análisis de **503,339 incidentes viales** registrados por el C5 de "
        "la Ciudad de México. Este tablero recorre el proyecto en dos fases: "
        "primero **describimos** lo que pasó y, con ese conocimiento, "
        "**predecimos** qué tipo de accidente es más probable según la zona y la hora."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidentes", f"{len(df):,}")
    c2.metric("Alcaldías", df['alcaldia_catalogo'].nunique())
    c3.metric("Colonias", f"{df['colonia_catalogo'].nunique():,}")
    c4.metric("% graves", f"{df['es_grave'].mean()*100:.1f}%")

    st.divider()
    st.subheader("¿Por qué analizamos así?")
    st.markdown(
        """
        - **De lo descriptivo a lo predictivo.** No predecimos "a ciegas": cada
          variable del modelo (hora, alcaldía, mes, fin de semana) la elegimos
          porque el análisis descriptivo mostró que la distribución de
          accidentes **cambia** con ella.
        - **7 tipos en vez de 14.** Los 14 subtipos originales tienen una cola
          de casos rarísimos. Agruparlos en **6 principales + "Otros"** hace los
          gráficos legibles y el modelo más estable, sin perder información útil.
        - **Interactivo.** Es un tablero: eliges alcaldía, mueves la hora y los
          gráficos responden. Así cualquiera explora sus propias preguntas.
        - **Cada gráfico tiene un porqué.** En cada análisis incluimos una nota
          explicando qué pregunta responde y por qué ese tipo de gráfico es el
          adecuado (proporción, comparación, relación, evolución o ubicación).
        """
    )


# ══════════════════════════════════════════════════════════
#  PÁGINA: DATASET
# ══════════════════════════════════════════════════════════

def pagina_dataset(df: pd.DataFrame) -> None:
    st.header("Acerca del Dataset")
    st.markdown(
        """
        Fuente: **C5 CDMX** (Centro de Comando, Control, Cómputo y Comunicaciones)
        vía [datos.cdmx.gob.mx](https://datos.cdmx.gob.mx). Cada fila es un
        incidente vial con fecha, hora, ubicación, tipo y tiempo de atención.

        **Limpieza aplicada:** se quitaron tipos no viales (Mi Taxi, Sismo…),
        registros fuera de 2022–2024 y duraciones negativas; se imputaron nulos
        de colonia/alcaldía y se derivaron columnas (hora, mes, severidad, etc.).
        """
    )
    st.divider()

    st.subheader("Explorar registros")
    c1, c2, c3 = st.columns(3)
    alc = c1.selectbox("Alcaldía", ["Todas"] + sorted(df['alcaldia_catalogo'].unique()))
    tipo = c2.selectbox("Tipo (7 categorías)", ["Todos"] + CATEGORIAS_7)
    anio = c3.selectbox("Año", ["Todos"] + sorted(df['anio'].unique().tolist()))

    f = df
    if alc != "Todas":
        f = f[f['alcaldia_catalogo'] == alc]
    if tipo != "Todos":
        f = f[f['categoria'] == tipo]
    if anio != "Todos":
        f = f[f['anio'] == anio]

    st.caption(f"Mostrando {min(500, len(f)):,} de {len(f):,} registros")
    cols = ['tipo_incidente_c4', 'incidente_c4', 'categoria', 'alcaldia_catalogo',
            'colonia_catalogo', 'hora', 'dia_semana', 'anio', 'mes',
            'minutos_atencion', 'es_grave']
    st.dataframe(f[cols].head(500), width='stretch', height=420)


# ══════════════════════════════════════════════════════════
#  PÁGINA: DESCRIPTIVO  (6 análisis)
# ══════════════════════════════════════════════════════════

def pagina_descriptivo(df: pd.DataFrame) -> None:
    st.header("Análisis Descriptivo")
    st.caption("Seis preguntas a la base de datos, seis tipos de gráfico.")

    tabs = st.tabs([
        "1 · Tipos por alcaldía 🥧",
        "2 · Mapa por hora 🗺️",
        "3 · Hora del día 📈",
        "4 · Ranking alcaldías 📊",
        "5 · Atención vs gravedad ⚄",
        "6 · Tendencia anual 🟧",
    ])

    with tabs[0]:
        analisis_pastel(df)
    with tabs[1]:
        analisis_mapa(df)
    with tabs[2]:
        analisis_hora(df)
    with tabs[3]:
        analisis_ranking(df)
    with tabs[4]:
        analisis_dispersion(df)
    with tabs[5]:
        analisis_tendencia(df)


# --- Análisis 1: PASTEL (idea del usuario) -----------------

def analisis_pastel(df: pd.DataFrame) -> None:
    st.subheader("¿Qué tipos de incidente predominan en una alcaldía?")
    alc = st.selectbox("Elige una alcaldía",
                       ["Todas las alcaldías"] + sorted(df['alcaldia_catalogo'].unique()),
                       key="pastel_alc")

    cat = CatalogoTipos.desde_dataframe(df, alcaldia=alc)
    tipos = cat.tipos()
    nombres = [t.nombre for t in tipos]
    valores = [t.total for t in tipos]

    col_pie, col_top = st.columns([3, 2])
    with col_pie:
        fig = go.Figure(go.Pie(
            labels=nombres, values=valores, hole=0.35,
            marker=dict(colors=SECUENCIA_7),
            textinfo='percent', sort=False,
            pull=[0.06 if i < 3 else 0 for i in range(len(nombres))],
        ))
        fig.update_layout(title=f"Distribución de tipos — {alc}",
                          height=430, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width='stretch')

    with col_top:
        st.markdown("#### Los 3 predominantes")
        for i, t in enumerate(cat.top(3), 1):
            st.metric(f"{i}. {t.nombre}",
                      f"{t.porcentaje():.1f}%",
                      f"{t.total:,} incidentes · {t.tasa_graves():.0f}% graves")

    porque(
        "**Pregunta:** ¿cómo se reparte el total de incidentes entre los 7 tipos "
        "en una zona concreta?  \n"
        "**Por qué un gráfico de pastel:** muestra *proporciones de un todo* "
        "(suman 100%) con pocas categorías — exactamente el caso de 7 tipos. "
        "Las 3 rebanadas más grandes se resaltan para leer de un vistazo lo "
        "predominante. Cambiar de alcaldía revela que el perfil de accidentes "
        "**no es igual en toda la ciudad**, idea que el predictivo aprovecha."
    )


# --- Análisis 2: MAPA DE CALOR POR HORA (idea del usuario) -

def analisis_mapa(df: pd.DataFrame) -> None:
    st.subheader("¿Dónde se concentran los incidentes según la hora?")
    hora = st.slider("Hora del día", 0, 23, 19, format="%d:00", key="mapa_hora")
    solo_graves = st.checkbox("Solo incidentes graves", key="mapa_graves")

    d = df[df['hora'] == hora]
    if solo_graves:
        d = d[d['es_grave']]
    d = d.dropna(subset=['latitud', 'longitud'])
    if len(d) > 20000:
        d = d.sample(20000, random_state=42)

    st.caption(f"{len(d):,} incidentes entre las {hora}:00 y las {hora}:59 h")
    fig = mapa_calor(d)
    st.plotly_chart(fig, width='stretch')

    porque(
        "**Pregunta:** ¿los focos de accidentes se mueven por la ciudad a lo "
        "largo del día?  \n"
        "**Por qué un mapa de calor:** la ubicación es geográfica (lat/long), así "
        "que un mapa es el soporte natural; el *calor* codifica densidad — dónde "
        "se amontonan los puntos — algo que una tabla nunca mostraría. El slider "
        "de hora convierte el mapa en una pequeña 'película': mueve la hora y "
        "observa cómo cambian las zonas calientes (p. ej. avenidas a hora pico)."
    )


def mapa_calor(d: pd.DataFrame):
    """Mapa de densidad sin token (OpenStreetMap). Compatible con plotly 5 y 6."""
    centro = dict(lat=19.38, lon=-99.13)
    try:  # plotly >= 6 (MapLibre, sin token)
        fig = px.density_map(d, lat='latitud', lon='longitud', radius=8,
                             center=centro, zoom=9.3,
                             color_continuous_scale="Inferno", height=560)
    except AttributeError:  # plotly 5.x
        fig = px.density_mapbox(d, lat='latitud', lon='longitud', radius=8,
                                center=centro, zoom=9.3,
                                mapbox_style="open-street-map",
                                color_continuous_scale="Inferno", height=560)
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    return fig


# --- Análisis 3: HORA DEL DÍA (líneas) ---------------------

def analisis_hora(df: pd.DataFrame) -> None:
    st.subheader("¿Cómo cambia el volumen de incidentes durante el día?")
    serie = (df.assign(Severidad=np.where(df['es_grave'], 'Grave', 'Leve'))
               .groupby(['hora', 'Severidad']).size().reset_index(name='total'))

    fig = px.area(serie, x='hora', y='total', color='Severidad',
                  color_discrete_map={'Leve': COL_LEVE, 'Grave': COL_GRAVE},
                  labels={'hora': 'Hora del día', 'total': 'Incidentes'},
                  height=460)
    fig.update_xaxes(dtick=2)
    fig.update_layout(legend_title=None, hovermode="x unified")
    st.plotly_chart(fig, width='stretch')

    pico = int(df['hora'].value_counts().idxmax())
    st.info(f"📌 Hora pico: **{pico}:00 h**. El volumen sube de forma sostenida "
            "desde la mañana y cae de madrugada.")

    porque(
        "**Pregunta:** ¿a qué horas hay más incidentes?  \n"
        "**Por qué un gráfico de líneas/área:** la hora es una variable "
        "*ordenada y continua* (0→23). La línea conecta valores consecutivos y "
        "deja ver la *tendencia* y los picos; el área apilada separa leve vs "
        "grave. La hora resultó tan informativa que es una de las variables del "
        "predictor."
    )


# --- Análisis 4: RANKING DE ALCALDÍAS (barras) -------------

def analisis_ranking(df: pd.DataFrame) -> None:
    st.subheader("¿Qué alcaldías tienen más incidentes y cuáles son más graves?")
    g = (df.groupby('alcaldia_catalogo')
           .agg(total=('es_grave', 'size'), tasa_grave=('es_grave', 'mean'))
           .reset_index())
    g = g[g['alcaldia_catalogo'] != 'Sin alcaldía']
    g['tasa_grave'] *= 100

    c1, c2 = st.columns(2)
    with c1:
        d = g.sort_values('total')
        fig = px.bar(d, x='total', y='alcaldia_catalogo', orientation='h',
                     labels={'total': 'Incidentes', 'alcaldia_catalogo': ''},
                     height=520)
        fig.update_traces(marker_color=COL_PRINCIPAL)
        fig.update_layout(title="Volumen total")
        st.plotly_chart(fig, width='stretch')
    with c2:
        d = g.sort_values('tasa_grave')
        fig = px.bar(d, x='tasa_grave', y='alcaldia_catalogo', orientation='h',
                     labels={'tasa_grave': '% graves', 'alcaldia_catalogo': ''},
                     height=520)
        fig.update_traces(marker_color=COL_GRAVE)
        fig.update_layout(title="Tasa de incidentes graves (%)")
        st.plotly_chart(fig, width='stretch')

    porque(
        "**Pregunta:** ¿dónde ocurren más incidentes y dónde son más graves?  \n"
        "**Por qué barras horizontales:** comparamos una magnitud entre "
        "*categorías discretas* (las 16 alcaldías). Las barras permiten leer el "
        "ranking directo y los nombres largos caben mejor en horizontal. "
        "Separamos *volumen* y *gravedad* porque no siempre coinciden: una "
        "alcaldía puede tener muchos choques leves y otra pocos pero graves."
    )


# --- Análisis 5: DISPERSIÓN -------------------------------

def analisis_dispersion(df: pd.DataFrame) -> None:
    st.subheader("¿La rapidez de atención se relaciona con el volumen o la gravedad?")
    base = df[~df['atencion_outlier']]
    g = (base.groupby('alcaldia_catalogo')
             .agg(total=('es_grave', 'size'),
                  tiempo=('minutos_atencion', 'mean'),
                  tasa_grave=('es_grave', 'mean'))
             .reset_index())
    g = g[g['alcaldia_catalogo'] != 'Sin alcaldía']
    g['tasa_grave'] *= 100

    fig = px.scatter(g, x='total', y='tiempo', size='total',
                     color='tasa_grave', text='alcaldia_catalogo',
                     color_continuous_scale="OrRd",
                     labels={'total': 'Incidentes', 'tiempo': 'Min. de atención',
                             'tasa_grave': '% graves'},
                     height=560)
    fig.update_traces(textposition='top center', textfont_size=9)
    st.plotly_chart(fig, width='stretch')

    porque(
        "**Pregunta:** ¿las alcaldías con más incidentes (o más graves) tardan "
        "más en atender?  \n"
        "**Por qué un diagrama de dispersión:** relaciona *dos variables "
        "continuas* (volumen y tiempo de atención); cada punto es una alcaldía. "
        "El color añade una tercera variable (% graves) y el tamaño el volumen. "
        "Es el gráfico ideal para detectar **correlación** y **casos atípicos** "
        "(alcaldías que se salen del patrón)."
    )


# --- Análisis 6: TENDENCIA ANUAL (columnas) ----------------

def analisis_tendencia(df: pd.DataFrame) -> None:
    st.subheader("¿Está creciendo la demanda año con año?")
    g = (df.groupby(['anio', 'mes']).size().reset_index(name='total'))
    g['Mes'] = g['mes'].map(MESES)
    g['anio'] = g['anio'].astype(str)

    fig = px.bar(g, x='Mes', y='total', color='anio', barmode='group',
                 category_orders={'Mes': list(MESES.values())},
                 color_discrete_sequence=[COL_LEVE, COL_PRINCIPAL, COL_GRAVE],
                 labels={'total': 'Incidentes', 'anio': 'Año'}, height=470)
    st.plotly_chart(fig, width='stretch')

    porque(
        "**Pregunta:** ¿hay estacionalidad y crecimiento entre 2022, 2023 y 2024?  \n"
        "**Por qué columnas agrupadas:** comparamos la *misma categoría* (cada "
        "mes) entre *varias series* (los años), colocando las barras lado a lado. "
        "Así se ve de golpe si un mes crece de un año a otro y qué meses son "
        "consistentemente más altos — información que alimenta la idea de "
        "*demanda esperada*."
    )


# ══════════════════════════════════════════════════════════
#  PÁGINA: PREDICTIVO
# ══════════════════════════════════════════════════════════

def pagina_predictivo(df: pd.DataFrame) -> None:
    st.header("Predicción · ¿Qué tipo de accidente es más probable?")
    st.markdown(
        "Usando el conocimiento del descriptivo (el tipo de accidente cambia por "
        "**zona** y por **hora**), el modelo estima la probabilidad de cada uno "
        "de los 7 tipos. Comparamos un **modelo** (Naive Bayes) con un **baseline** "
        "de frecuencias reales."
    )
    pred = cargar_predictor(df)

    c1, c2, c3, c4 = st.columns(4)
    alc = c1.selectbox("Alcaldía", sorted(ALCALDIAS_CDMX))
    hora = c2.slider("Hora", 0, 23, 19, format="%d:00")
    mes = c3.select_slider("Mes", options=list(range(1, 13)),
                           value=6, format_func=lambda m: MESES[m])
    fin = c4.checkbox("¿Fin de semana?")

    tabla = pred.predecir(alc, hora, mes=mes, fin_semana=fin)
    top = tabla.iloc[0]

    st.divider()
    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric("Tipo más probable", top['tipo'],
                  f"{top['prob_modelo']*100:.1f}% (modelo)")
        st.caption(f"Baseline (frecuencia real): "
                   f"{top['prob_baseline']*100:.1f}%")
        st.caption(f"Validación del modelo — accuracy "
                   f"{pred.metricas['accuracy']:.2f} · "
                   f"F1 macro {pred.metricas['f1_macro']:.2f}")
    with m2:
        fig = go.Figure()
        fig.add_bar(y=tabla['tipo'], x=tabla['prob_modelo']*100,
                    orientation='h', name='Modelo (Naive Bayes)',
                    marker_color=COL_PRINCIPAL)
        fig.add_bar(y=tabla['tipo'], x=tabla['prob_baseline']*100,
                    orientation='h', name='Baseline (frecuencias)',
                    marker_color=COL_LEVE)
        fig.update_layout(barmode='group', height=400,
                          xaxis_title="Probabilidad (%)",
                          yaxis=dict(autorange="reversed"),
                          legend=dict(orientation="h", y=1.1),
                          margin=dict(t=10))
        st.plotly_chart(fig, width='stretch')

    porque(
        "**Cómo se conecta con el descriptivo:** el *baseline* es literalmente la "
        "frecuencia P(tipo | alcaldía, hora) que vimos en el pastel y el mapa. El "
        "*modelo* (Naive Bayes) suaviza y generaliza esas frecuencias, de modo que "
        "da una respuesta razonable incluso para combinaciones con pocos datos. "
        "Cuando ambas barras coinciden, el modelo confirma el dato; cuando "
        "difieren, suele ser por escasez de casos en esa celda exacta. "
        "**No usamos Random Forest:** Naive Bayes es la versión probabilística "
        "directa de las frecuencias, más interpretable para este problema."
    )


# ══════════════════════════════════════════════════════════
#  NAVEGACIÓN
# ══════════════════════════════════════════════════════════

def main() -> None:
    df = cargar_datos()
    if df.empty:
        st.error("No se encontró `datos/viales_limpio.parquet` ni el CSV. "
                 "Corre primero `python main.py --paso limpieza`.")
        st.stop()

    st.sidebar.title("🚦 Viales CDMX")
    pagina = st.sidebar.radio(
        "Navegación",
        ["Inicio", "Dataset", "Descriptivo", "Predictivo"],
    )
    st.sidebar.divider()
    st.sidebar.caption("Proyecto final · Aplicaciones para Análisis de Datos\n\n"
                       "Datos: C5 CDMX 2022–2024")

    if pagina == "Inicio":
        pagina_inicio(df)
    elif pagina == "Dataset":
        pagina_dataset(df)
    elif pagina == "Descriptivo":
        pagina_descriptivo(df)
    elif pagina == "Predictivo":
        pagina_predictivo(df)


if __name__ == "__main__":
    main()

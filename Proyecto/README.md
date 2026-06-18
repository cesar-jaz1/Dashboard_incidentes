# 🚦 Incidentes Viales CDMX 2022–2024 — Dashboard interactivo

Proyecto final de **Aplicaciones para Análisis de Datos**.

Análisis de **503,339 incidentes viales** registrados por el C5 de la Ciudad de
México entre 2022 y 2024. El proyecto va **de lo descriptivo a lo predictivo**:
primero entiende *qué pasó* y, con ese conocimiento, predice *qué tipo de
accidente es más probable* según la zona y la hora. Todo se explora en un
**tablero interactivo** hecho con Streamlit.

> **🔴 Demo en vivo:** _pega aquí la URL de Streamlit Cloud cuando despliegues_
> `https://<tu-app>.streamlit.app`

---

## ¿Por qué analizamos así?

Esta es la idea que guía todo el proyecto:

1. **De lo descriptivo a lo predictivo.** No predecimos "a ciegas". Cada
   variable del modelo (hora, alcaldía, mes, fin de semana) se eligió porque el
   análisis descriptivo **demostró** que la distribución de accidentes cambia
   con ella. El predictivo formaliza ese conocimiento.
2. **7 tipos en lugar de 14.** El dataset trae 14 subtipos, pero 6 concentran
   el ~99 %. Los agrupamos en **6 principales + "Otros"**: gráficos legibles y
   un modelo más estable, sin perder señal. Esta agrupación es la *base de
   conocimiento* (clase `TipoAccidente`) que comparten descriptivo y predictivo.
3. **Interactivo, porque es un tablero.** El usuario elige alcaldía, mueve la
   hora y los gráficos responden. Así cualquiera formula sus propias preguntas.
4. **Cada gráfico tiene un porqué.** Usamos el tipo de gráfico adecuado a cada
   pregunta: proporción → pastel, ubicación → mapa, evolución → líneas,
   comparación → barras/columnas, relación → dispersión.

---

## Estructura del proyecto

```
Proyecto/
├── app.py            # ⭐ Dashboard interactivo (Streamlit + Plotly)  → streamlit run app.py
├── main.py           # Pipeline offline (genera CSV/Parquet y PNGs del reporte)
├── limpieza.py       # Fase 1 — limpieza + exporta viales_limpio.csv y .parquet
├── modelos.py        # Fase 2 — clases POO (incluye TipoAccidente / CatalogoTipos)
├── descriptivo.py    # Fase 3 — exporta las 6 gráficas estáticas (PNG)
├── predictivo.py     # Fase 4 — PredictorTipoAccidente (Naive Bayes + baseline)
├── generar_uml.py    # Diagrama UML de clases → docs/UML_clases.svg / .pdf
│
├── datos/
│   ├── inViales_2022_2024.csv   # CSV original (no se sube: ~128 MB)
│   ├── viales_limpio.csv        # limpio completo (no se sube)
│   └── viales_limpio.parquet    # ⭐ versión ligera (~9 MB) que usa el dashboard
│
├── docs/UML_clases.svg          # diagrama de clases
├── graficas/                    # PNGs del reporte (se regeneran)
├── requirements.txt
└── .gitignore
```

---

## Las clases (POO)

El diagrama completo está en [`docs/UML_clases.svg`](docs/UML_clases.svg). Tres capas:

| Capa | Clases | Rol |
|---|---|---|
| **dominio** | `UbicacionGeografica`, `ReporteC4`, `Incidente`, `Colonia`, `Alcaldia` | Modelan un incidente y su agregación geográfica (Incidente **compone** ubicación + reporte; Colonia **agrega** incidentes; Alcaldía **agrega** colonias). |
| **conocimiento** | `TipoAccidente`, `CatalogoTipos` | La base de conocimiento: agrupan los 7 tipos y calculan sus distribuciones (por hora, por alcaldía, % graves). |
| **predictivo** | `AnalisisZona`, `PredictorTipoAccidente` | Operan sobre el conocimiento para predecir. |

`TipoAccidente` (la clase nueva del proyecto) es el puente: el descriptivo la
usa para la gráfica de pastel y el predictivo usa su distribución horaria como
evidencia.

---

## Análisis descriptivo — 6 preguntas, 6 tipos de gráfico

| # | Pregunta | Gráfico | Por qué ese gráfico |
|---|---|---|---|
| 1 | ¿Qué tipos predominan en una alcaldía? | **Pastel** | Proporción de un todo (suma 100 %) con pocas categorías. |
| 2 | ¿Dónde se concentran según la hora? | **Mapa de calor** + slider | La ubicación es geográfica; el calor codifica densidad y el slider la vuelve "película". |
| 3 | ¿Cómo cambia el volumen durante el día? | **Líneas / área** | La hora es continua y ordenada: la línea muestra tendencia y picos. |
| 4 | ¿Qué alcaldías tienen más y cuáles más graves? | **Barras** | Comparan magnitudes entre categorías discretas (ranking directo). |
| 5 | ¿La atención se relaciona con volumen/gravedad? | **Dispersión** | Relaciona dos variables continuas; revela correlación y atípicos. |
| 6 | ¿Crece la demanda año con año? | **Columnas agrupadas** | Comparan la misma categoría (mes) entre series (años) lado a lado. |

Cada análisis incluye en el tablero una nota "¿Por qué este análisis?".

---

## Análisis predictivo — tipo de accidente más probable

Eliges **alcaldía + hora** y el sistema estima la probabilidad de cada uno de
los 7 tipos, con **dos enfoques que se comparan**:

- **Baseline** — frecuencias condicionadas `P(tipo | alcaldía, hora)` leídas
  directo de los datos (el conocimiento descriptivo "crudo").
- **Modelo** — `Naive Bayes` categórico: la versión suavizada y generalizadora
  de esas frecuencias. **No se usa Random Forest**; Naive Bayes es la
  contraparte probabilística directa de las frecuencias, más interpretable aquí.

> **Nota honesta (hallazgo del proyecto):** como "Choque sin lesionados" es el
> ~46 % de todos los casos, el modelo casi siempre lo elige como *el más
> probable* (accuracy ≈ 0.46, F1-macro bajo por el fuerte desbalance de clases).
> El valor del predictor no está en el *argmax* sino en la **distribución de
> probabilidades**: muestra, por ejemplo, que en Cuauhtémoc a las 19 h la
> probabilidad de *Atropellado* sube a ~16 % frente al ~10 % global. Eso es
> exactamente el conocimiento descriptivo, ahora cuantificado.

---

## Cómo correrlo en local

**Requisito:** Python 3.10+.

```bash
# 1. Entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 2. Dependencias
pip install -r requirements.txt

# 3a. Abrir el tablero interactivo (necesita datos/viales_limpio.parquet)
streamlit run app.py

# 3b. (Opcional) Regenerar datos y gráficas del reporte
python main.py                 # limpieza → descriptivo → predictivo
python main.py --paso limpieza # genera viales_limpio.csv y .parquet
python generar_uml.py          # regenera el diagrama UML
```

> El dashboard solo necesita `datos/viales_limpio.parquet` (incluido en el
> repo). El CSV completo y el pipeline `main.py` solo hacen falta si quieres
> regenerar todo desde el original.

---

## Desplegar en Streamlit Community Cloud (gratis, sin instalar nada)

Así cualquiera usa el tablero desde una URL, sin descargar requerimientos:

1. Sube el proyecto a un repositorio **público** de GitHub
   (el `.parquet` de ~9 MB debe subir; el `.csv` queda excluido por `.gitignore`).
2. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
3. **New app** → elige el repo, la rama y `app.py` como archivo principal.
4. **Deploy.** Streamlit instala `requirements.txt` automáticamente y te da la URL.
5. Pega esa URL arriba en la sección **Demo en vivo**.

---

## Dataset

| Campo | Valor |
|---|---|
| Fuente | C5 CDMX — datos abiertos ([datos.cdmx.gob.mx](https://datos.cdmx.gob.mx)) |
| Período | 2022 – 2024 |
| Registros (limpios) | 503,339 |
| Tipos (agrupados) | 7 (6 principales + "Otros") |

---

## Dependencias

`streamlit` · `pandas` · `numpy` · `plotly` · `scikit-learn` · `pyarrow`
(+ `matplotlib` para las gráficas estáticas). Ver `requirements.txt`.

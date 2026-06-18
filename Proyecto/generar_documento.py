"""
generar_documento.py
=====================
Genera el documento del proyecto en PDF (docs/Documento_Proyecto.pdf) con la
estructura: Carátula, Introducción, Objetivos, Metodología (Diseño e
Implementación), Resultados y Conclusión.

La sección de Metodología es la más detallada: explica POR QUÉ esa metodología,
POR QUÉ ese diseño de clases y POR QUÉ esa implementación.

    python generar_documento.py
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Table, TableStyle, HRFlowable)

BASE = Path(__file__).resolve().parent
OUT = BASE / "docs"
OUT.mkdir(exist_ok=True)
PDF = OUT / "Documento_Proyecto.pdf"

ACENTO = colors.HexColor("#D85A30")
GRAVE  = colors.HexColor("#A32D2D")
TINTA  = colors.HexColor("#23303f")
GRIS   = colors.HexColor("#5b6573")

# ── Estilos ────────────────────────────────────────────────
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=ACENTO,
                    fontSize=16, spaceBefore=16, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=TINTA,
                    fontSize=13, spaceBefore=12, spaceAfter=5)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], textColor=GRAVE,
                    fontSize=11.5, spaceBefore=9, spaceAfter=3)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontSize=10.3, leading=15,
                      alignment=TA_JUSTIFY, spaceAfter=7)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=16, bulletIndent=4,
                        spaceAfter=3)
CELDA = ParagraphStyle("CELDA", parent=ss["BodyText"], fontSize=9, leading=12)
CELDA_B = ParagraphStyle("CELDA_B", parent=CELDA, textColor=colors.white,
                         fontName="Helvetica-Bold")

story = []


def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def h3(t): story.append(Paragraph(t, H3))
def p(t):  story.append(Paragraph(t, BODY))
def li(t): story.append(Paragraph(f"• {t}", BULLET))
def gap(h=6): story.append(Spacer(1, h))


def tabla(filas, anchos, encabezado=True):
    data = [[Paragraph(c, CELDA_B if (encabezado and i == 0) else CELDA)
             for c in fila] for i, fila in enumerate(filas)]
    t = Table(data, colWidths=anchos, hAlign="LEFT")
    estilo = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8cfd8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f3ef")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if encabezado:
        estilo.append(("BACKGROUND", (0, 0), (-1, 0), ACENTO))
    t.setStyle(TableStyle(estilo))
    story.append(t)
    gap(10)


# ════════════════════════════════════════════════════════════
#  CARÁTULA
# ════════════════════════════════════════════════════════════
story.append(Spacer(1, 4.5 * cm))
story.append(Paragraph("Análisis de Incidentes Viales de la Ciudad de México",
             ParagraphStyle("T", parent=ss["Title"], fontSize=24, leading=30,
                            textColor=TINTA, alignment=TA_CENTER)))
story.append(Paragraph("2022 – 2024",
             ParagraphStyle("Tsub", parent=ss["Title"], fontSize=18,
                            textColor=ACENTO, alignment=TA_CENTER, spaceBefore=6)))
gap(10)
story.append(HRFlowable(width="60%", thickness=1.4, color=ACENTO,
                        spaceBefore=4, spaceAfter=18, hAlign="CENTER"))
story.append(Paragraph("Un tablero interactivo del análisis descriptivo al predictivo",
             ParagraphStyle("Tlema", parent=ss["Normal"], fontSize=12.5,
                            textColor=GRIS, alignment=TA_CENTER)))
story.append(Spacer(1, 5 * cm))
cover_meta = ParagraphStyle("CM", parent=ss["Normal"], fontSize=12,
                            alignment=TA_CENTER, leading=20, textColor=TINTA)
story.append(Paragraph("<b>Proyecto Final</b>", cover_meta))
story.append(Paragraph("Aplicaciones para Análisis de Datos", cover_meta))
story.append(Paragraph("Julio Antonio Zavala", cover_meta))
story.append(Paragraph("Junio 2026", cover_meta))
story.append(PageBreak())


# ════════════════════════════════════════════════════════════
#  1. INTRODUCCIÓN
# ════════════════════════════════════════════════════════════
h1("1. Introducción")
p("Los incidentes viales son uno de los problemas de seguridad y movilidad más "
  "relevantes de la Ciudad de México. Cada llamada de emergencia atendida por el "
  "C5 (Centro de Comando, Control, Cómputo, Comunicaciones y Contacto Ciudadano) "
  "queda registrada con su fecha, hora, ubicación, tipo de incidente y tiempo de "
  "atención. Analizar este registro permite entender <i>cuándo</i>, <i>dónde</i> "
  "y <i>qué</i> tipo de accidentes ocurren, información valiosa para la prevención "
  "y la asignación de recursos.")
p("Este proyecto trabaja con <b>503,339 incidentes viales</b> registrados entre "
  "2022 y 2024. La propuesta no se queda en describir los datos: los usa como "
  "fuente de conocimiento para construir un modelo capaz de estimar, dada una "
  "alcaldía y una hora, qué tipo de accidente es más probable. Todo el análisis "
  "se presenta en un <b>tablero interactivo</b> (dashboard) que permite a "
  "cualquier persona explorar los datos sin escribir una sola línea de código.")
p("El documento describe la metodología seguida, el diseño orientado a objetos "
  "del sistema, los detalles de su implementación y los resultados obtenidos.")


# ════════════════════════════════════════════════════════════
#  2. OBJETIVOS
# ════════════════════════════════════════════════════════════
h1("2. Objetivos")
h2("Objetivo general")
p("Desarrollar un sistema en Python, orientado a objetos, que analice los "
  "incidentes viales de la CDMX (2022–2024) y, a partir del conocimiento "
  "descriptivo, prediga el tipo de accidente más probable según la zona y la "
  "hora, presentándolo en un tablero interactivo desplegable en la web.")
h2("Objetivos específicos")
li("Limpiar y preparar el conjunto de datos original, documentando cada decisión.")
li("Modelar el dominio con un diseño de clases (POO) que represente incidentes, "
   "ubicaciones, colonias, alcaldías y tipos de accidente.")
li("Realizar un análisis descriptivo que responda preguntas concretas mediante "
   "el tipo de gráfico adecuado a cada una.")
li("Construir, con el conocimiento descriptivo, un modelo predictivo del tipo de "
   "accidente, sin usar Random Forest, y evaluarlo de forma honesta.")
li("Integrar todo en un dashboard interactivo y desplegarlo en Streamlit "
   "Community Cloud para su acceso público.")


# ════════════════════════════════════════════════════════════
#  3. METODOLOGÍA, DISEÑO E IMPLEMENTACIÓN  (sección central)
# ════════════════════════════════════════════════════════════
h1("3. Metodología, Diseño e Implementación")

# ---- 3.1 Metodología ----
h2("3.1 Metodología — ¿por qué este enfoque?")
p("Se adoptó un proceso por fases inspirado en <b>CRISP-DM</b> (el estándar de "
  "facto en proyectos de datos): comprensión del problema, comprensión de los "
  "datos, preparación, modelado, evaluación y despliegue. Se eligió este marco "
  "porque obliga a un orden lógico: <b>no se puede modelar sobre datos sucios ni "
  "predecir sobre fenómenos que no se han entendido primero</b>. Cada fase "
  "alimenta a la siguiente.")
p("La decisión metodológica más importante del proyecto es ir <b>de lo "
  "descriptivo a lo predictivo</b>. En lugar de lanzar un algoritmo a “adivinar”, "
  "primero se describe el fenómeno para <b>generar conocimiento</b> —¿a qué horas "
  "hay más incidentes?, ¿qué zonas concentran cuáles tipos?— y solo entonces se "
  "predice, usando justamente las variables que el análisis descriptivo demostró "
  "relevantes (hora, alcaldía, mes y fin de semana). Así el modelo no es una caja "
  "negra: cada variable que recibe tiene una justificación previa. Este es además "
  "el requisito explícito del proyecto: que el predictivo <b>use</b> el "
  "conocimiento del descriptivo.")
h3("Preparación de los datos — por qué cada decisión de limpieza")
p("El archivo original tenía 504,261 registros. La limpieza buscó conservar solo "
  "incidentes viales reales y bien formados, documentando cada criterio para que "
  "el proceso sea reproducible y defendible:")
tabla([
    ["Decisión de limpieza", "Por qué"],
    ["Eliminar tipos no viales (Mi&nbsp;Taxi, Sismo, Mi&nbsp;Calle, Detención ciudadana)",
     "No son accidentes de tránsito; contaminarían el análisis del fenómeno vial."],
    ["Eliminar registros de 2021 (73)",
     "Quedan fuera del periodo de estudio (2022–2024); son ruido residual."],
    ["Eliminar duraciones negativas (3)",
     "La fecha de cierre es anterior a la de creación: dato imposible, error de captura."],
    ["Marcar (no eliminar) atenciones &gt; 24 h como outlier",
     "Pueden ser reales; se etiquetan para excluirlas de promedios sin perder el registro."],
    ["Imputar colonia nula con “Sin colonia” (2.2%)",
     "Conserva el incidente para el análisis temporal/tipo aunque falte la colonia exacta."],
    ["Derivar columnas (hora, mes, año, fin de semana, severidad, minutos de atención)",
     "Convierten fechas crudas en variables directamente analizables y predictivas."],
], [7.6 * cm, 8.4 * cm])
h3("Agrupación en 7 tipos — por qué")
p("La variable de subtipo traía 14 categorías, pero solo 6 concentran cerca del "
  "99% de los casos; las otras 8 son una cola de eventos rarísimos. Se agruparon "
  "en <b>6 tipos principales + “Otros”</b>. El porqué es doble: <b>legibilidad</b> "
  "(un gráfico de pastel de 7 rebanadas se entiende; uno de 14 no) y "
  "<b>estabilidad del modelo</b> (clases con poquísimos casos generan predicciones "
  "poco fiables). Esta agrupación se encapsuló en una clase para reutilizarla en "
  "todo el sistema.")
h3("Preguntas y gráficos — por qué cada tipo de gráfico")
p("El análisis descriptivo se planteó como seis preguntas, eligiendo en cada una "
  "el gráfico cuya naturaleza corresponde a la pregunta:")
tabla([
    ["Pregunta", "Gráfico", "Por qué"],
    ["¿Qué tipos predominan en una alcaldía?", "Pastel", "Muestra proporción de un todo (100%)."],
    ["¿Dónde se concentran según la hora?", "Mapa de calor", "La densidad espacial se lee en un mapa, no en una tabla."],
    ["¿Cómo cambia el volumen durante el día?", "Líneas / área", "La hora es continua: la línea revela tendencia y picos."],
    ["¿Qué alcaldías tienen más y cuáles más graves?", "Barras", "Comparan magnitudes entre categorías discretas."],
    ["¿La atención se relaciona con el volumen?", "Dispersión", "Relaciona dos variables continuas; revela correlación."],
    ["¿Crece la demanda año con año?", "Columnas agrupadas", "Comparan la misma categoría (mes) entre series (años)."],
], [5.6 * cm, 3.2 * cm, 7.2 * cm])

# ---- 3.2 Diseño ----
h2("3.2 Diseño — ¿por qué este diseño de clases?")
p("Se eligió la <b>programación orientada a objetos</b> porque el dominio se "
  "modela de forma natural como objetos que reflejan la realidad: un incidente "
  "ocurre en una ubicación y se reporta en un momento; una colonia agrupa "
  "incidentes; una alcaldía agrupa colonias. La POO permite <b>encapsular</b> los "
  "datos junto con el comportamiento (cada clase sabe calcular sus propias "
  "métricas), hace el código <b>reutilizable</b> y deja que el descriptivo y el "
  "predictivo compartan la misma representación del conocimiento.")
p("El sistema se organiza en tres capas. El diagrama UML completo está en "
  "<font color='#3f7fcf'>docs/UML_clases.svg</font>.")
tabla([
    ["Clase", "Capa", "Responsabilidad (por qué existe)"],
    ["UbicacionGeografica", "dominio", "Aísla lat/long, alcaldía y colonia; calcula distancias (Haversine)."],
    ["ReporteC4", "dominio", "Aísla el “cuándo y cómo” del reporte: hora, franja, tiempo de respuesta."],
    ["Incidente", "dominio", "Objeto atómico; <b>compone</b> una ubicación y un reporte."],
    ["Colonia", "dominio", "<b>Agrega</b> incidentes y expone métricas locales."],
    ["Alcaldia", "dominio", "<b>Agrega</b> colonias; métricas de demarcación."],
    ["TipoAccidente", "conocimiento", "Encapsula uno de los 7 tipos y su distribución (hora, zona, gravedad)."],
    ["CatalogoTipos", "conocimiento", "<b>Agrega</b> los 7 tipos; se construye vectorizado desde los datos."],
    ["AnalisisZona", "predictivo", "Opera sobre una alcaldía (análisis y agrupamientos)."],
    ["PredictorTipoAccidente", "predictivo", "El modelo: Naive Bayes + baseline de frecuencias."],
], [4.6 * cm, 2.6 * cm, 8.8 * cm])
h3("Composición vs. agregación — por qué cada relación es la que es")
p("Distinguir estas dos relaciones fue una decisión de diseño consciente, no "
  "estética:")
li("<b>Composición</b> (el todo posee la parte y controla su ciclo de vida): "
   "<i>Incidente</i> compone su <i>ReporteC4</i> y su <i>UbicacionGeografica</i>, "
   "porque un reporte o una ubicación <b>no tienen sentido sin su incidente</b>: "
   "nacen y mueren con él.")
li("<b>Agregación</b> (el todo agrupa partes que existen por sí mismas): "
   "<i>Colonia</i> agrega <i>Incidentes</i>, <i>Alcaldia</i> agrega <i>Colonias</i> "
   "y <i>CatalogoTipos</i> agrega <i>TipoAccidente</i>. Los incidentes existen "
   "independientemente de la colonia que los agrupe para un análisis.")
h3("La clase TipoAccidente — por qué se añadió")
p("Es la clase nueva del proyecto (sugerida por la asesoría) y la pieza clave del "
  "enfoque: actúa como <b>puente entre el descriptivo y el predictivo</b>. "
  "Concentra todo el conocimiento estadístico de un tipo de accidente —su "
  "porcentaje, su tasa de gravedad, su distribución por hora y por alcaldía— de "
  "modo que el descriptivo la usa para dibujar el pastel y el predictivo usa su "
  "distribución horaria como evidencia. El conocimiento se calcula una sola vez y "
  "se reutiliza, en lugar de repetir la lógica en cada módulo.")

# ---- 3.3 Implementación ----
h2("3.3 Implementación — ¿por qué se implementó así?")
h3("Tecnologías y por qué")
li("<b>pandas / NumPy</b>: manejo vectorizado de medio millón de filas.")
li("<b>scikit-learn</b>: el modelo Naive Bayes y las métricas de evaluación.")
li("<b>Streamlit</b>: convierte un script de Python en una aplicación web "
   "interactiva sin programar frontend, y se despliega gratis en la nube.")
li("<b>Plotly</b>: gráficos interactivos (hover, zoom) y un mapa que no requiere "
   "token de pago para mostrarse.")
li("<b>PyArrow / Parquet</b>: formato de datos comprimido para el despliegue.")
h3("Rendimiento: POO sobre agregados, no sobre 500 mil objetos")
p("Construir 503,339 objetos <i>Incidente</i> en memoria haría el tablero lento. "
  "Por eso las clases de conocimiento (<i>CatalogoTipos</i>) se construyen de "
  "forma <b>vectorizada</b> directamente desde el DataFrame con un método de "
  "fábrica (<font color='#3f7fcf'>desde_dataframe</font>): se conserva el diseño "
  "orientado a objetos, pero el cálculo es prácticamente instantáneo. La POO "
  "aporta la <i>estructura</i> y el <i>significado</i>; pandas aporta la "
  "<i>velocidad</i>.")
h3("Dos formatos de datos: CSV canónico y Parquet para la nube")
p("El CSV limpio (la fuente canónica del proyecto, ~128&nbsp;MB) es el resultado "
  "auditable de la limpieza y el que alimenta el pipeline de clases. Sin embargo, "
  "GitHub rechaza archivos de más de 100&nbsp;MB y el tablero debe cargar rápido "
  "en la nube. Por eso se genera además un <b>Parquet</b> (~9&nbsp;MB) con tipos "
  "optimizados (categorías, enteros de 8&nbsp;bits, flotantes de 32): es un espejo "
  "ligero del CSV. El dashboard lee el Parquet; el análisis local puede usar "
  "cualquiera de los dos.")
h3("El modelo predictivo — por qué Naive Bayes y un baseline")
p("El requisito era predecir el tipo de accidente <b>sin Random Forest</b> y "
  "aprovechando el conocimiento descriptivo. La solución combina dos vías:")
li("<b>Baseline (frecuencias condicionadas)</b>: la probabilidad P(tipo | "
   "alcaldía, hora) leída directamente de los datos. Es, literalmente, el "
   "conocimiento descriptivo en bruto.")
li("<b>Modelo (Naive Bayes categórico)</b>: la versión <i>suavizada</i> "
   "(con suavizado de Laplace) y <i>generalizadora</i> de esas mismas frecuencias. "
   "Se eligió Naive Bayes porque es la contraparte probabilística <b>directa</b> "
   "de un conteo de frecuencias —no una caja negra— y entrega una probabilidad "
   "para cada tipo, que es justo lo que el dashboard muestra.")
p("Para las combinaciones (alcaldía, hora) con pocos casos se aplica un "
  "<b>respaldo jerárquico</b>: si la celda exacta es escasa, se mezcla con el "
  "comportamiento general de la alcaldía y, en último caso, con el global. Esto "
  "evita predicciones erráticas por falta de datos.")
h3("El tablero: interactividad y despliegue")
p("La aplicación se divide en páginas (Inicio, Dataset, Descriptivo, Predictivo) "
  "y usa el sistema de <b>caché</b> de Streamlit para cargar los datos y entrenar "
  "el modelo una sola vez. El resultado se publica en <b>Streamlit Community "
  "Cloud</b>, que instala las dependencias automáticamente y entrega una URL "
  "pública: el usuario solo abre el enlace, sin instalar nada.")


# ════════════════════════════════════════════════════════════
#  4. RESULTADOS
# ════════════════════════════════════════════════════════════
story.append(PageBreak())
h1("4. Resultados")
p("El análisis descriptivo arrojó hallazgos claros, que el modelo predictivo "
  "después cuantifica:")
li("<b>Distribución de tipos:</b> los choques dominan el panorama —“sin "
   "lesionados” (45.9%) y “con lesionados” (29.1%)—, seguidos de Atropellado "
   "(10.4%) y Motociclista (10.0%).")
li("<b>Gravedad:</b> el 10.6% de los incidentes son graves (lesionados o "
   "fallecidos). Un dato estructural revelador: prácticamente <b>todos los "
   "atropellamientos se clasifican como graves</b>.")
li("<b>Hora pico:</b> las 19:00&nbsp;h concentran el mayor número de incidentes; "
   "el volumen sube de forma sostenida durante el día y cae de madrugada.")
li("<b>Geografía:</b> <i>Iztapalapa</i> es la alcaldía con más incidentes "
   "(82,068), mientras que <i>Cuauhtémoc</i> tiene la mayor tasa de gravedad "
   "(15.3%). Volumen y gravedad no son lo mismo.")
p("El modelo predictivo confirma y cuantifica ese conocimiento. Por ejemplo, en "
  "<b>Cuauhtémoc a las 19:00&nbsp;h</b> la probabilidad de <i>Atropellado</i> "
  "sube a ~16% frente al ~10% global: la zona y la hora sí cambian el perfil de "
  "riesgo.")
h3("Evaluación honesta del modelo")
p("El modelo alcanza una exactitud (accuracy) de 0.46 y un F1-macro bajo. Lejos "
  "de ser un fracaso, esto es un <b>hallazgo</b>: como “Choque sin lesionados” es "
  "casi la mitad de todos los casos, predecir <i>el tipo más probable</i> casi "
  "siempre devuelve esa clase mayoritaria. Por eso el valor del predictor no está "
  "en acertar una sola etiqueta, sino en la <b>distribución de probabilidades</b> "
  "completa, que sí refleja cómo cambian los riesgos por zona y hora. Reportarlo "
  "así, en lugar de inflar una métrica, es parte del rigor del proyecto.")


# ════════════════════════════════════════════════════════════
#  5. CONCLUSIÓN
# ════════════════════════════════════════════════════════════
h1("5. Conclusión")
p("El proyecto cumple su objetivo: convierte medio millón de registros crudos en "
  "conocimiento accionable y lo pone al alcance de cualquiera mediante un tablero "
  "interactivo. La decisión de ir <b>de lo descriptivo a lo predictivo</b> "
  "demostró su valor: cada variable del modelo está justificada por un hallazgo "
  "previo, y el predictor no hace más que formalizar lo que los gráficos ya "
  "sugerían.")
p("El diseño orientado a objetos —con la distinción cuidadosa entre composición y "
  "agregación, y la clase <i>TipoAccidente</i> como puente del conocimiento— "
  "mantuvo el código ordenado y reutilizable. En la implementación, separar el "
  "CSV canónico del Parquet ligero, y construir las clases de forma vectorizada, "
  "permitió un tablero rápido y desplegable en la nube sin sacrificar el rigor del "
  "modelado.")
p("Como trabajo futuro, el modelo podría enriquecerse con variables como el clima "
  "o la presencia de eventos masivos, y el análisis de gravedad podría abordarse "
  "como un problema propio. Aun así, el sistema actual entrega una base sólida, "
  "interpretable y honesta para entender los incidentes viales de la CDMX.")


# ════════════════════════════════════════════════════════════
#  Construcción del PDF (con número de página)
# ════════════════════════════════════════════════════════════

def pie_de_pagina(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRIS)
    canvas.drawCentredString(letter[0] / 2, 1.2 * cm, str(doc.page))
    canvas.restoreState()


def construir():
    doc = SimpleDocTemplate(
        str(PDF), pagesize=letter,
        leftMargin=2.3 * cm, rightMargin=2.3 * cm,
        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
        title="Análisis de Incidentes Viales CDMX 2022-2024",
        author="Julio Antonio Zavala",
    )
    # Sin número en la carátula; numeradas a partir de la página 2.
    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=pie_de_pagina)
    print("Generado:", PDF)


if __name__ == "__main__":
    construir()

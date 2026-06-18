"""Genera el diagrama UML de clases del proyecto de Incidentes Viales CDMX.

Dibuja un SVG calculando posiciones y conectores (sin dependencias del
sistema) y, si están disponibles svglib + reportlab, también un PDF vectorial.
Re-ejecutable:  python generar_uml.py

Refleja las clases reales de modelos.py y predictivo.py, en 3 capas:
  dominio       — UbicacionGeografica, ReporteC4, Incidente, Colonia, Alcaldia
  conocimiento  — TipoAccidente, CatalogoTipos
  predictivo    — AnalisisZona, PredictorTipoAccidente
"""
from __future__ import annotations

import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "docs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- paleta capas
CAPAS = {
    "dominio":      dict(fill="#efeafe", head="#ddd3fb", bord="#6c5cd0", tinta="#352a78"),
    "conocimiento": dict(fill="#e3f4ec", head="#cfeede", bord="#2f9e75", tinta="#0f5a44"),
    "predictivo":   dict(fill="#e6f0fb", head="#d2e4f8", bord="#3f7fcf", tinta="#13406b"),
}

# ----------------------------------------------------------------- definicion
# name: (capa, estereotipo, [atributos], [metodos])
CLASES = {
    "UbicacionGeografica": ("dominio", "entidad",
        ["+latitud: float", "+longitud: float", "+alcaldia: str", "+colonia: str"],
        ["+distancia_a(otro): float", "+en_radio(lat, lon, r): bool",
         "+coordenadas(): tuple"]),
    "ReporteC4": ("dominio", "entidad",
        ["+dt_creacion: datetime", "+dt_cierre: datetime",
         "+dia_semana: str", "+tipo_entrada: str"],
        ["+tiempo_respuesta(): float", "+es_fin_semana(): bool",
         "+franja_horaria(): str", "+es_outlier_duracion(): bool"]),
    "Incidente": ("dominio", "entidad",
        ["+folio: str", "+tipo: str", "+subtipo: str", "+codigo_cierre: str",
         "+reporte: ReporteC4", "+ubicacion: UbicacionGeografica"],
        ["+es_grave(): bool", "+nivel_severidad(): int",
         "+calcular_duracion(): float", "+resumen(): str"]),
    "Colonia": ("dominio", "agregado",
        ["+nombre: str", "+alcaldia: str", "-_incidentes: list<Incidente>"],
        ["+agregar(inc)", "+total(): int", "+tasa_graves(): float",
         "+hora_pico(): int", "+duracion_promedio(): float"]),
    "Alcaldia": ("dominio", "agregado",
        ["+nombre: str", "-_colonias: dict<str, Colonia>"],
        ["+agregar_colonia(c)", "+total_incidentes(): int", "+tasa_graves(): float",
         "+ranking_colonias(n): list", "+tiempo_promedio_atencion(): float"]),
    "TipoAccidente": ("conocimiento", "conocimiento",
        ["+nombre: str", "+total: int", "+graves: int",
         "-_por_hora: ndarray", "-_por_alcaldia: dict"],
        ["+porcentaje(): float", "+tasa_graves(): float", "+hora_pico(): int",
         "+proporcion_horaria(): ndarray", "+alcaldia_top(n): list"]),
    "CatalogoTipos": ("conocimiento", "agregado",
        ["-_tipos: dict<str, TipoAccidente>", "+total: int"],
        ["+desde_dataframe(df, alcaldia): CatalogoTipos", "+tipos(): list",
         "+top(n): list", "+distribucion(): dict"]),
    "AnalisisZona": ("predictivo", "analisis",
        ["+alcaldia: Alcaldia", "-_modelo", "-_features: list"],
        ["+set_modelo(m, features)", "+predecir_gravedad(h, sub, fin): str",
         "+colonias_alto_riesgo(p): list", "+cluster_colonias(k): dict",
         "+exportar_reporte(): str"]),
    "PredictorTipoAccidente": ("predictivo", "Naive Bayes",
        ["+le_alc: LabelEncoder", "+modelo: CategoricalNB",
         "+clases_: list", "+metricas: dict"],
        ["+entrenar(df): self", "+predecir_modelo(alc, h, mes, fin): dict",
         "+predecir_baseline(alc, h): dict", "+predecir(alc, h): DataFrame"]),
}

# columnas (capa por columna); el orden vertical reduce cruces
COLUMNAS = [
    ["UbicacionGeografica", "ReporteC4", "Incidente"],
    ["Colonia", "Alcaldia"],
    ["TipoAccidente", "CatalogoTipos"],
    ["AnalisisZona", "PredictorTipoAccidente"],
]

# (origen, destino, tipo): comp | agg | assoc | dep
RELACIONES = [
    ("Incidente", "ReporteC4", "comp"),
    ("Incidente", "UbicacionGeografica", "comp"),
    ("Colonia", "Incidente", "agg"),
    ("Alcaldia", "Colonia", "agg"),
    ("CatalogoTipos", "TipoAccidente", "agg"),
    ("AnalisisZona", "Alcaldia", "assoc"),
    ("PredictorTipoAccidente", "TipoAccidente", "dep"),
]

# ----------------------------------------------------------------- geometria
CW_MEMBER = 6.0   # ancho aprox por caracter (miembros, 10px)
CW_NAME = 7.6     # ancho aprox por caracter (nombre, 12.5px bold)
PAD = 12
LINE = 16
HEAD = 40
VGAP = 34
HGAP = 150
TOP = 96
LEFT = 40


def medir(nombre):
    capa, est, attrs, meths = CLASES[nombre]
    w = max(
        len(nombre) * CW_NAME,
        len(f"<<{est}>>") * 5.6,
        max((len(s) for s in attrs + meths), default=0) * CW_MEMBER,
    ) + 2 * PAD
    w = max(w, 168)
    h = HEAD + max(len(attrs), 1) * LINE + 8 + max(len(meths), 1) * LINE + 8
    return w, h


def disponer():
    cajas = {}
    x = LEFT
    for col in COLUMNAS:
        anchos = [medir(n)[0] for n in col]
        colw = max(anchos)
        y = TOP
        for n in col:
            w, h = medir(n)
            cajas[n] = dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2)
            y += h + VGAP
        x += colw + HGAP
    return cajas


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def caja_svg(nombre, b):
    capa, est, attrs, meths = CLASES[nombre]
    c = CAPAS[capa]
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    ah = max(len(attrs), 1) * LINE + 8
    s = []
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" '
             f'fill="{c["fill"]}" stroke="{c["bord"]}" stroke-width="1.6"/>')
    s.append(f'<path d="M{x},{y+HEAD} v{-HEAD-(-5)} a5,5 0 0 1 5,-5 h{w-10} '
             f'a5,5 0 0 1 5,5 v{HEAD-5} z" fill="{c["head"]}" stroke="{c["bord"]}" '
             f'stroke-width="1.6"/>')
    cx = x + w / 2
    s.append(f'<text x="{cx}" y="{y+16}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="9.5" font-style="italic" fill="{c["tinta"]}">&#171;{esc(est)}&#187;</text>')
    s.append(f'<text x="{cx}" y="{y+33}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="{c["tinta"]}">{esc(nombre)}</text>')
    s.append(f'<line x1="{x}" y1="{y+HEAD}" x2="{x+w}" y2="{y+HEAD}" stroke="{c["bord"]}" stroke-width="1.2"/>')
    ty = y + HEAD + 14
    for a in (attrs or ["—"]):
        s.append(f'<text x="{x+PAD}" y="{ty}" font-family="sans-serif" font-size="10" '
                 f'fill="#23303f">{esc(a)}</text>')
        ty += LINE
    s.append(f'<line x1="{x}" y1="{y+HEAD+ah}" x2="{x+w}" y2="{y+HEAD+ah}" stroke="{c["bord"]}" stroke-width="1.2"/>')
    ty = y + HEAD + ah + 14
    for m in (meths or ["—"]):
        s.append(f'<text x="{x+PAD}" y="{ty}" font-family="sans-serif" font-size="10" '
                 f'fill="#23303f">{esc(m)}</text>')
        ty += LINE
    return "\n".join(s)


def borde(b, hacia):
    """Punto del borde de la caja b en direccion al centro 'hacia'."""
    dx, dy = hacia[0] - b["cx"], hacia[1] - b["cy"]
    if dx == 0 and dy == 0:
        return b["cx"], b["cy"]
    hw, hh = b["w"] / 2, b["h"] / 2
    sx = hw / abs(dx) if dx else math.inf
    sy = hh / abs(dy) if dy else math.inf
    s = min(sx, sy)
    return b["cx"] + dx * s, b["cy"] + dy * s


def punta_flecha(x, y, ang, abierto=True):
    L, a = 12, math.radians(24)
    p1 = (x - L * math.cos(ang - a), y - L * math.sin(ang - a))
    p2 = (x - L * math.cos(ang + a), y - L * math.sin(ang + a))
    if abierto:
        return (f'<polyline points="{p1[0]:.1f},{p1[1]:.1f} {x:.1f},{y:.1f} '
                f'{p2[0]:.1f},{p2[1]:.1f}" fill="none" stroke="#3a4656" stroke-width="1.4"/>')
    return (f'<polygon points="{x:.1f},{y:.1f} {p1[0]:.1f},{p1[1]:.1f} '
            f'{p2[0]:.1f},{p2[1]:.1f}" fill="#fff" stroke="#3a4656" stroke-width="1.4"/>')


def rombo(x, y, ang, relleno):
    L, W = 16, 7
    bx, by = x + L * math.cos(ang), y + L * math.sin(ang)        # vertice lejano
    mx, my = x + L / 2 * math.cos(ang), y + L / 2 * math.sin(ang)  # centro
    px, py = -math.sin(ang) * W, math.cos(ang) * W
    fill = "#3a4656" if relleno else "#fff"
    pts = f'{x:.1f},{y:.1f} {mx+px:.1f},{my+py:.1f} {bx:.1f},{by:.1f} {mx-px:.1f},{my-py:.1f}'
    return f'<polygon points="{pts}" fill="{fill}" stroke="#3a4656" stroke-width="1.4"/>', (bx, by)


def conector_svg(src, dst, tipo, cajas):
    a, b = cajas[src], cajas[dst]
    misma_col = abs(a["cx"] - b["cx"]) < 60
    partes = []
    dash = ' stroke-dasharray="6,5"' if tipo == "dep" else ""
    col_st = "#3a4656"
    if misma_col:
        lx = min(a["x"], b["x"]) - 28
        pa = (a["x"], a["cy"])
        pb = (b["x"], b["cy"])
        puntos = f'{pa[0]:.1f},{pa[1]:.1f} {lx:.1f},{pa[1]:.1f} {lx:.1f},{pb[1]:.1f} {pb[0]:.1f},{pb[1]:.1f}'
        ang_dst = 0.0
        ang_src = math.pi
        ax, ay = pa
        bx, by = pb
    else:
        ax, ay = borde(a, (b["cx"], b["cy"]))
        bx, by = borde(b, (a["cx"], a["cy"]))
        ang_dst = math.atan2(by - ay, bx - ax)
        ang_src = math.atan2(ay - by, ax - bx)
        puntos = f'{ax:.1f},{ay:.1f} {bx:.1f},{by:.1f}'

    if tipo in ("comp", "agg"):
        marca, nuevo = rombo(ax, ay, ang_src, relleno=(tipo == "comp"))
        if misma_col:
            puntos = (f'{nuevo[0]:.1f},{ay:.1f} {lx:.1f},{ay:.1f} '
                      f'{lx:.1f},{by:.1f} {bx:.1f},{by:.1f}')
        else:
            puntos = f'{nuevo[0]:.1f},{nuevo[1]:.1f} {bx:.1f},{by:.1f}'
        partes.append(f'<polyline points="{puntos}" fill="none" stroke="{col_st}" '
                      f'stroke-width="1.4"{dash}/>')
        partes.append(marca)
        partes.append(punta_flecha(bx, by, ang_dst, abierto=True))
    else:
        partes.append(f'<polyline points="{puntos}" fill="none" stroke="{col_st}" '
                      f'stroke-width="1.4"{dash}/>')
        partes.append(punta_flecha(bx, by, ang_dst, abierto=True))
    return "\n".join(partes)


def leyenda_svg(x, y):
    items = [
        ("comp", "Composición (todo ◆ parte)"),
        ("agg", "Agregación (todo ◇ parte)"),
        ("assoc", "Asociación →"),
        ("dep", "Dependencia (usa) ⇢"),
    ]
    s = [f'<rect x="{x}" y="{y}" width="290" height="116" rx="6" fill="#ffffff" '
         f'stroke="#c8cfd8" stroke-width="1.2"/>',
         f'<text x="{x+14}" y="{y+22}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#23303f">Leyenda de relaciones</text>']
    yy = y + 44
    for tipo, txt in items:
        x0, x1 = x + 16, x + 56
        if tipo == "comp":
            r, _ = rombo(x0, yy, 0.0, True)
            s.append(r); s.append(f'<line x1="{x0+16}" y1="{yy}" x2="{x1}" y2="{yy}" stroke="#3a4656" stroke-width="1.4"/>')
        elif tipo == "agg":
            r, _ = rombo(x0, yy, 0.0, False)
            s.append(r); s.append(f'<line x1="{x0+16}" y1="{yy}" x2="{x1}" y2="{yy}" stroke="#3a4656" stroke-width="1.4"/>')
        elif tipo == "assoc":
            s.append(f'<line x1="{x0}" y1="{yy}" x2="{x1}" y2="{yy}" stroke="#3a4656" stroke-width="1.4"/>')
            s.append(punta_flecha(x1, yy, 0.0, True))
        else:
            s.append(f'<line x1="{x0}" y1="{yy}" x2="{x1}" y2="{yy}" stroke="#3a4656" stroke-width="1.4" stroke-dasharray="6,5"/>')
            s.append(punta_flecha(x1, yy, 0.0, True))
        s.append(f'<text x="{x+66}" y="{yy+4}" font-family="sans-serif" font-size="10.5" fill="#23303f">{txt}</text>')
        yy += 22
    return "\n".join(s)


def construir():
    cajas = disponer()
    ancho = max(b["x"] + b["w"] for b in cajas.values()) + 40
    alto = max(b["y"] + b["h"] for b in cajas.values()) + 150
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" '
           f'viewBox="0 0 {ancho} {alto}">',
           f'<rect x="0" y="0" width="{ancho}" height="{alto}" fill="#ffffff"/>',
           f'<text x="40" y="46" font-family="sans-serif" font-size="22" '
           f'font-weight="bold" fill="#14306b">Incidentes Viales CDMX — Diagrama UML de clases</text>',
           f'<text x="40" y="68" font-family="sans-serif" font-size="12" fill="#5b6573">'
           f'Capas: dominio · conocimiento · predictivo</text>']
    for src, dst, tipo in RELACIONES:
        svg.append(conector_svg(src, dst, tipo, cajas))
    for nombre, b in cajas.items():
        svg.append(caja_svg(nombre, b))
    svg.append(leyenda_svg(ancho - 320, alto - 134))
    svg.append("</svg>")
    return "\n".join(svg)


def main():
    svg = construir()
    svg_path = OUT / "UML_clases.svg"
    svg_path.write_text(svg, encoding="utf-8")
    print("Generado:")
    print(" ", svg_path)

    # PDF opcional (requiere svglib + reportlab)
    try:
        from reportlab.graphics import renderPDF
        from svglib.svglib import svg2rlg
        pdf_path = OUT / "UML_clases.pdf"
        renderPDF.drawToFile(svg2rlg(str(svg_path)), str(pdf_path))
        print(" ", pdf_path)
    except Exception as e:
        print(f"  (PDF omitido: {type(e).__name__}. Instala svglib+reportlab para generarlo.)")

    print(f"Clases: {len(CLASES)} | Relaciones: {len(RELACIONES)}")


if __name__ == "__main__":
    main()

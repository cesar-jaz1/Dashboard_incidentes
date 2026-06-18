"""
modelos.py
==========
Fase 2 del proyecto: Clases POO para el análisis de incidentes viales CDMX.
Dataset base: viales_limpio.csv  (503,339 registros, 25 columnas)

Jerarquía de clases:
  Incidente          — un registro del dataset (el objeto atómico)
  ReporteC4          — contexto temporal y canal de reporte del incidente
  UbicacionGeografica — coordenadas y zona geográfica del incidente
  Colonia            — agrupa incidentes de una misma colonia
  Alcaldia           — agrupa colonias y expone métricas por demarcación
  AnalisisZona       — módulo predictivo sobre una Alcaldia
"""

from __future__ import annotations

import math
import numpy as np
from collections import Counter
from datetime import datetime
from typing import Optional


# ══════════════════════════════════════════════════════════════
#  CATÁLOGOS (valores válidos directamente del dataset)
# ══════════════════════════════════════════════════════════════

TIPOS_INCIDENTE   = {'Accidente', 'Lesionado', 'Cadáver'}
SUBTIPOS_INCIDENTE = {
    'Choque con lesionados', 'Motociclista', 'Choque sin lesionados',
    'Atropellado', 'Volcadura', 'Choque con prensados',
    'Vehiculo desbarrancado', 'Persona atrapada / desbarrancada',
    'Ciclista', 'Vehículo atrapadovarado', 'Accidente automovilístico',
    'Incidente de tránsito', 'Otros', 'Monopatín', 'Ferroviario',
    'Persona atropellada',
}
CODIGOS_CIERRE    = {'A': 'Atendido', 'D': 'Derivado', 'F': 'Falsa alarma', 'I': 'Informativo'}
CLASIFICACIONES   = {'URGENCIAS MEDICAS', 'EMERGENCIA'}
TIPOS_ENTRADA     = {
    'LLAMADA DEL 911', 'BOTÓN DE AUXILIO', 'RADIO', 'CÁMARA',
    'REDES', 'APLICATIVOS', 'LLAMADA APP911', 'SOS MUJERES *765',
    'LECTOR DE PLACAS', 'DESCONOCIDO',
}
ALCALDIAS_CDMX = {
    'Azcapotzalco', 'Benito Juárez', 'Coyoacán', 'Cuajimalpa de Morelos',
    'Cuauhtémoc', 'Gustavo A. Madero', 'Iztacalco', 'Iztapalapa',
    'La Magdalena Contreras', 'Miguel Hidalgo', 'Milpa Alta', 'Tlalpan',
    'Tláhuac', 'Venustiano Carranza', 'Xochimilco', 'Álvaro Obregón',
}

SEVERIDAD = {'Accidente': 0, 'Lesionado': 1, 'Cadáver': 2}
FRANJA_HORARIA = {
    range(0, 6):   'madrugada',
    range(6, 12):  'mañana',
    range(12, 18): 'tarde',
    range(18, 24): 'noche',
}


def _franja(hora: int) -> str:
    for rango, nombre in FRANJA_HORARIA.items():
        if hora in rango:
            return nombre
    return 'desconocida'


# ══════════════════════════════════════════════════════════════
#  AGRUPACIÓN DE SUBTIPOS EN 7 CATEGORÍAS
# ══════════════════════════════════════════════════════════════
#  incidente_c4 trae 14 subtipos, pero 6 concentran ~99% de los casos.
#  Para análisis y visualización los reducimos a 7 categorías:
#  los 6 principales + "Otros" (la cola larga de 8 subtipos raros).
#
#  Esta agrupación es la BASE DE CONOCIMIENTO del proyecto: la usa
#  tanto el análisis descriptivo (gráfica de pastel por alcaldía) como
#  el predictivo (predicción del tipo de accidente más probable).

CATEGORIAS_PRINCIPALES = [
    'Choque sin lesionados',
    'Choque con lesionados',
    'Atropellado',
    'Motociclista',
    'Volcadura',
    'Ciclista',
]
CATEGORIA_OTROS = 'Otros'
CATEGORIAS_7 = CATEGORIAS_PRINCIPALES + [CATEGORIA_OTROS]


def categoria_7(subtipo: str) -> str:
    """Reduce cualquiera de los 14 subtipos de incidente_c4 a una de las 7 categorías."""
    return subtipo if subtipo in CATEGORIAS_PRINCIPALES else CATEGORIA_OTROS


def agregar_categoria(df):
    """
    Agrega la columna 'categoria' (7 grupos) al DataFrame de forma vectorizada.
    Devuelve una copia del df con la columna añadida.
    """
    df = df.copy()
    sub = df['incidente_c4'].astype(str).to_numpy()
    df['categoria'] = np.where(np.isin(sub, CATEGORIAS_PRINCIPALES), sub, CATEGORIA_OTROS)
    return df


# ══════════════════════════════════════════════════════════════
#  CLASE 1 — UbicacionGeografica
# ══════════════════════════════════════════════════════════════

class UbicacionGeografica:
    """
    Encapsula la posición geográfica de un incidente.
    Columnas fuente: latitud, longitud, alcaldia_catalogo, colonia_catalogo
    """

    def __init__(
        self,
        latitud:   float,
        longitud:  float,
        alcaldia:  str,
        colonia:   str,
    ) -> None:
        self.latitud  = latitud
        self.longitud = longitud
        self.alcaldia = alcaldia
        self.colonia  = colonia

    # ── Métodos ──────────────────────────────────────────────

    def distancia_a(self, otro: UbicacionGeografica) -> float:
        """
        Distancia en kilómetros entre dos ubicaciones usando la fórmula de Haversine.
        Útil para agrupar incidentes cercanos o medir cobertura policial.
        """
        R = 6371.0
        lat1, lon1 = math.radians(self.latitud),  math.radians(self.longitud)
        lat2, lon2 = math.radians(otro.latitud),  math.radians(otro.longitud)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    def en_radio(self, lat: float, lon: float, radio_km: float) -> bool:
        """Devuelve True si esta ubicación está dentro del radio dado (en km)."""
        referencia = UbicacionGeografica(lat, lon, '', '')
        return self.distancia_a(referencia) <= radio_km

    def coordenadas(self) -> tuple[float, float]:
        """Retorna (latitud, longitud) como tupla."""
        return (self.latitud, self.longitud)

    def __repr__(self) -> str:
        return f"UbicacionGeografica({self.alcaldia} / {self.colonia})"


# ══════════════════════════════════════════════════════════════
#  CLASE 2 — ReporteC4
# ══════════════════════════════════════════════════════════════

class ReporteC4:
    """
    Encapsula el contexto temporal y el canal de reporte de un incidente.
    Columnas fuente: fecha_creacion, hora_creacion, fecha_cierre,
                     hora_cierre, dia_semana, tipo_entrada
    """

    def __init__(
        self,
        dt_creacion:  datetime,
        dt_cierre:    datetime,
        dia_semana:   str,
        tipo_entrada: str,
    ) -> None:
        self.dt_creacion  = dt_creacion
        self.dt_cierre    = dt_cierre
        self.dia_semana   = dia_semana
        self.tipo_entrada = tipo_entrada

    # ── Métodos ──────────────────────────────────────────────

    def tiempo_respuesta(self) -> float:
        """Minutos transcurridos entre creación y cierre del reporte."""
        delta = self.dt_cierre - self.dt_creacion
        return delta.total_seconds() / 60

    def es_fin_semana(self) -> bool:
        """True si el incidente ocurrió sábado o domingo."""
        return self.dia_semana in ('Sábado', 'Domingo')

    def hora_creacion(self) -> int:
        """Hora (0-23) en que se creó el reporte."""
        return self.dt_creacion.hour

    def franja_horaria(self) -> str:
        """Devuelve 'madrugada', 'mañana', 'tarde' o 'noche'."""
        return _franja(self.dt_creacion.hour)

    def es_outlier_duracion(self, umbral_min: float = 1440.0) -> bool:
        """True si el tiempo de respuesta supera el umbral (default 24 h)."""
        return self.tiempo_respuesta() > umbral_min

    def __repr__(self) -> str:
        return (f"ReporteC4({self.dt_creacion.strftime('%Y-%m-%d %H:%M')} "
                f"vía {self.tipo_entrada})")


# ══════════════════════════════════════════════════════════════
#  CLASE 3 — Incidente
# ══════════════════════════════════════════════════════════════

class Incidente:
    """
    Objeto atómico del sistema. Representa una fila del dataset limpio.
    Columnas fuente: folio, tipo_incidente_c4, incidente_c4,
                     codigo_cierre, clas_con_f_alarma
    Composición con: ReporteC4, UbicacionGeografica
    """

    def __init__(
        self,
        folio:          str,
        tipo:           str,
        subtipo:        str,
        codigo_cierre:  str,
        clasificacion:  str,
        reporte:        ReporteC4,
        ubicacion:      UbicacionGeografica,
    ) -> None:
        self.folio         = folio
        self.tipo          = tipo
        self.subtipo       = subtipo
        self.codigo_cierre = codigo_cierre
        self.clasificacion = clasificacion
        self.reporte       = reporte       # composición
        self.ubicacion     = ubicacion     # composición

    # ── Métodos ──────────────────────────────────────────────

    def es_grave(self) -> bool:
        """True si el incidente involucra lesionados o cadáveres."""
        return self.tipo in ('Lesionado', 'Cadáver')

    def nivel_severidad(self) -> int:
        """0 = Accidente leve, 1 = Lesionado, 2 = Cadáver."""
        return SEVERIDAD.get(self.tipo, 0)

    def fue_atendido(self) -> bool:
        """True si el código de cierre es 'A' (Atendido)."""
        return self.codigo_cierre == 'A'

    def get_hora_dia(self) -> str:
        """Franja horaria del incidente: madrugada / mañana / tarde / noche."""
        return self.reporte.franja_horaria()

    def calcular_duracion(self) -> float:
        """Minutos de duración del incidente (delega a ReporteC4)."""
        return self.reporte.tiempo_respuesta()

    def descripcion_cierre(self) -> str:
        """Texto legible del código de cierre."""
        return CODIGOS_CIERRE.get(self.codigo_cierre, 'Desconocido')

    def resumen(self) -> str:
        """Cadena de texto con los datos clave del incidente."""
        return (
            f"[{self.folio}] {self.tipo} / {self.subtipo} | "
            f"{self.ubicacion.alcaldia}, {self.ubicacion.colonia} | "
            f"{self.reporte.dt_creacion.strftime('%Y-%m-%d %H:%M')} | "
            f"Cierre: {self.descripcion_cierre()} | "
            f"Duración: {self.calcular_duracion():.0f} min"
        )

    def __repr__(self) -> str:
        return f"Incidente({self.folio}, {self.tipo}, {self.ubicacion.alcaldia})"


# ══════════════════════════════════════════════════════════════
#  CLASE 4 — Colonia
# ══════════════════════════════════════════════════════════════

class Colonia:
    """
    Agrega todos los incidentes de una colonia y expone métricas.
    Relación: una Colonia contiene muchos Incidente.
    """

    def __init__(self, nombre: str, alcaldia: str) -> None:
        self.nombre    = nombre
        self.alcaldia  = alcaldia
        self._incidentes: list[Incidente] = []

    # ── Gestión de incidentes ────────────────────────────────

    def agregar(self, incidente: Incidente) -> None:
        """Agrega un incidente a la colonia."""
        self._incidentes.append(incidente)

    def total(self) -> int:
        """Número total de incidentes registrados."""
        return len(self._incidentes)

    # ── Métricas descriptivas ────────────────────────────────

    def tasa_graves(self) -> float:
        """Proporción de incidentes graves (lesionados + cadáveres) sobre el total."""
        if not self._incidentes:
            return 0.0
        graves = sum(1 for i in self._incidentes if i.es_grave())
        return graves / self.total()

    def hora_pico(self) -> int:
        """Hora del día (0-23) con más incidentes registrados."""
        if not self._incidentes:
            return -1
        horas = [i.reporte.hora_creacion() for i in self._incidentes]
        return Counter(horas).most_common(1)[0][0]

    def subtipo_frecuente(self) -> str:
        """Subtipo de incidente más común en la colonia."""
        if not self._incidentes:
            return 'Sin datos'
        subtipos = [i.subtipo for i in self._incidentes]
        return Counter(subtipos).most_common(1)[0][0]

    def duracion_promedio(self) -> float:
        """Tiempo promedio de atención en minutos, excluyendo outliers."""
        tiempos = [
            i.calcular_duracion()
            for i in self._incidentes
            if not i.reporte.es_outlier_duracion()
        ]
        return sum(tiempos) / len(tiempos) if tiempos else 0.0

    def incidentes_por_anio(self) -> dict[int, int]:
        """Conteo de incidentes agrupados por año."""
        anios = [i.reporte.dt_creacion.year for i in self._incidentes]
        return dict(sorted(Counter(anios).items()))

    def __repr__(self) -> str:
        return f"Colonia({self.nombre}, {self.alcaldia}, {self.total()} incidentes)"


# ══════════════════════════════════════════════════════════════
#  CLASE 5 — Alcaldia
# ══════════════════════════════════════════════════════════════

class Alcaldia:
    """
    Agrega todas las colonias de una alcaldía y expone métricas de demarcación.
    Relación: una Alcaldia contiene muchas Colonia.
    """

    def __init__(self, nombre: str) -> None:
        self.nombre   = nombre
        self._colonias: dict[str, Colonia] = {}

    # ── Gestión de colonias ──────────────────────────────────

    def agregar_colonia(self, colonia: Colonia) -> None:
        """Registra una colonia en la alcaldía."""
        self._colonias[colonia.nombre] = colonia

    def obtener_colonia(self, nombre: str) -> Optional[Colonia]:
        """Retorna la colonia por nombre, o None si no existe."""
        return self._colonias.get(nombre)

    def colonias(self) -> list[Colonia]:
        """Lista de todas las colonias de la alcaldía."""
        return list(self._colonias.values())

    # ── Métricas descriptivas ────────────────────────────────

    def total_incidentes(self) -> int:
        """Total de incidentes en toda la alcaldía."""
        return sum(c.total() for c in self._colonias.values())

    def tasa_graves(self) -> float:
        """Proporción de incidentes graves sobre el total de la alcaldía."""
        total = self.total_incidentes()
        if total == 0:
            return 0.0
        graves = sum(
            sum(1 for i in c._incidentes if i.es_grave())
            for c in self._colonias.values()
        )
        return graves / total

    def ranking_colonias(self, top: int = 5) -> list[tuple[str, int]]:
        """
        Lista de colonias ordenadas por número de incidentes (descendente).
        Retorna lista de tuplas (nombre_colonia, total_incidentes).
        """
        ranking = [(c.nombre, c.total()) for c in self._colonias.values()]
        return sorted(ranking, key=lambda x: x[1], reverse=True)[:top]

    def tiempo_promedio_atencion(self) -> float:
        """Tiempo promedio de atención en minutos para toda la alcaldía."""
        tiempos = []
        for colonia in self._colonias.values():
            tiempos += [
                i.calcular_duracion()
                for i in colonia._incidentes
                if not i.reporte.es_outlier_duracion()
            ]
        return sum(tiempos) / len(tiempos) if tiempos else 0.0

    def tendencia_anual(self) -> dict[int, int]:
        """Incidentes por año para detectar tendencia 2022 → 2024."""
        conteo: Counter = Counter()
        for colonia in self._colonias.values():
            for inc in colonia._incidentes:
                conteo[inc.reporte.dt_creacion.year] += 1
        return dict(sorted(conteo.items()))

    def hora_pico(self) -> int:
        """Hora del día con más incidentes en toda la alcaldía."""
        horas: list[int] = []
        for colonia in self._colonias.values():
            horas += [i.reporte.hora_creacion() for i in colonia._incidentes]
        return Counter(horas).most_common(1)[0][0] if horas else -1

    def distribucion_severidad(self) -> dict[str, int]:
        """Conteo de incidentes por nivel de severidad."""
        conteo: Counter = Counter()
        for colonia in self._colonias.values():
            for inc in colonia._incidentes:
                conteo[inc.tipo] += 1
        return dict(conteo)

    def __repr__(self) -> str:
        return (f"Alcaldia({self.nombre}, "
                f"{len(self._colonias)} colonias, "
                f"{self.total_incidentes()} incidentes)")


# ══════════════════════════════════════════════════════════════
#  CLASE 6 — AnalisisZona
# ══════════════════════════════════════════════════════════════

class AnalisisZona:
    """
    Módulo predictivo que opera sobre una Alcaldia.
    Encapsula el modelo de ML y expone métodos de predicción y clustering.
    El modelo se entrena externamente (en el notebook predictivo) y
    se inyecta mediante set_modelo().
    """

    def __init__(self, alcaldia: Alcaldia) -> None:
        self.alcaldia  = alcaldia
        self._modelo   = None          # se asigna con set_modelo()
        self._features: list[str] = [] # nombres de columnas usadas

    # ── Configuración ────────────────────────────────────────

    def set_modelo(self, modelo, features: list[str]) -> None:
        """Inyecta el modelo entrenado y los nombres de sus features."""
        self._modelo   = modelo
        self._features = features
        print(f"[AnalisisZona] Modelo asignado: {type(modelo).__name__} "
              f"| Features: {features}")

    # ── Predicción ───────────────────────────────────────────

    def predecir_gravedad(self, hora: int, subtipo: str,
                          es_fin_semana: bool) -> str:
        """
        Predice si un incidente hipotético sería grave o no.
        Requiere que el modelo haya sido asignado con set_modelo().
        Retorna 'grave' o 'leve'.
        """
        if self._modelo is None:
            raise RuntimeError("Asigna un modelo primero con set_modelo().")

        x = [[hora, subtipo, int(es_fin_semana)]]
        pred = self._modelo.predict(x)[0]
        return 'grave' if pred == 1 else 'leve'

    def predecir_demanda(self, mes: int, anio: int) -> int:
        """
        Estima el número de incidentes esperados para un mes dado.
        Usa la media histórica del mismo mes como baseline.
        """
        conteos = []
        for colonia in self.alcaldia.colonias():
            for inc in colonia._incidentes:
                if inc.reporte.dt_creacion.month == mes:
                    conteos.append(1)
        return len(conteos)

    # ── Análisis de zona ─────────────────────────────────────

    def colonias_alto_riesgo(self, percentil: float = 0.75) -> list[Colonia]:
        """
        Retorna las colonias cuya tasa de graves supera el percentil indicado.
        Útil para identificar zonas prioritarias de intervención.
        """
        colonias = self.alcaldia.colonias()
        tasas    = sorted([c.tasa_graves() for c in colonias])
        if not tasas:
            return []
        umbral   = tasas[int(len(tasas) * percentil)]
        return [c for c in colonias if c.tasa_graves() >= umbral]

    def cluster_colonias(self, n_clusters: int = 3) -> dict[int, list[str]]:
        """
        Agrupa colonias por perfil de riesgo usando k-means sobre
        (total_incidentes, tasa_graves, hora_pico).
        Requiere scikit-learn instalado.
        Retorna dict {cluster_id: [nombre_colonia, ...]}.
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            import numpy as np
        except ImportError:
            raise ImportError("Instala scikit-learn: pip install scikit-learn")

        colonias = self.alcaldia.colonias()
        if len(colonias) < n_clusters:
            raise ValueError(f"Se necesitan al menos {n_clusters} colonias.")

        X = np.array([
            [c.total(), c.tasa_graves(), c.hora_pico()]
            for c in colonias
        ], dtype=float)

        X_scaled = StandardScaler().fit_transform(X)
        etiquetas = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_scaled)

        resultado: dict[int, list[str]] = {i: [] for i in range(n_clusters)}
        for colonia, etiqueta in zip(colonias, etiquetas):
            resultado[int(etiqueta)].append(colonia.nombre)
        return resultado

    def exportar_reporte(self) -> str:
        """
        Genera un texto con las métricas principales de la alcaldía.
        Útil para el reporte final del proyecto.
        """
        a = self.alcaldia
        lineas = [
            f"{'='*50}",
            f"  REPORTE DE ZONA: {a.nombre.upper()}",
            f"{'='*50}",
            f"  Total incidentes : {a.total_incidentes():,}",
            f"  Tasa de graves   : {a.tasa_graves()*100:.1f}%",
            f"  Hora pico        : {a.hora_pico()}:00 hrs",
            f"  Tiempo prom. atención: {a.tiempo_promedio_atencion():.0f} min",
            f"\n  Top 5 colonias con más incidentes:",
        ]
        for nombre, total in a.ranking_colonias(top=5):
            lineas.append(f"    • {nombre:<35} {total:>5,}")
        lineas.append(f"\n  Tendencia anual:")
        for anio, cnt in a.tendencia_anual().items():
            lineas.append(f"    {anio}: {cnt:,}")
        lineas.append("="*50)
        return "\n".join(lineas)

    def __repr__(self) -> str:
        modelo = type(self._modelo).__name__ if self._modelo else 'Sin modelo'
        return f"AnalisisZona({self.alcaldia.nombre}, modelo={modelo})"


# ══════════════════════════════════════════════════════════════
#  CLASE 7 — TipoAccidente  (sugerida por el profe)
# ══════════════════════════════════════════════════════════════

class TipoAccidente:
    """
    Representa UNA de las 7 categorías de incidente y concentra todo el
    conocimiento estadístico sobre ella: cuántos hubo, qué tan graves son,
    a qué horas ocurren y en qué alcaldías.

    Es el puente descriptivo → predictivo:
      - el descriptivo la usa para la gráfica de pastel y el ranking;
      - el predictivo usa distribucion_horaria() como evidencia para
        estimar qué tipo es más probable a cierta hora.
    """

    def __init__(self, nombre: str, total: int, graves: int,
                 por_hora: np.ndarray, por_alcaldia: dict[str, int],
                 total_global: int) -> None:
        self.nombre        = nombre
        self.total         = int(total)
        self.graves        = int(graves)
        self._por_hora     = por_hora              # np.array shape (24,)
        self._por_alcaldia = por_alcaldia          # {alcaldia: conteo}
        self._total_global = int(total_global)     # total del catálogo padre

    # ── Métricas ──────────────────────────────────────────────

    def porcentaje(self) -> float:
        """Porcentaje que representa este tipo sobre el total del catálogo."""
        return 100.0 * self.total / self._total_global if self._total_global else 0.0

    def tasa_graves(self) -> float:
        """Porcentaje de incidentes graves dentro de este tipo."""
        return 100.0 * self.graves / self.total if self.total else 0.0

    def hora_pico(self) -> int:
        """Hora del día (0-23) en que más ocurre este tipo."""
        return int(self._por_hora.argmax()) if self.total else -1

    def distribucion_horaria(self) -> np.ndarray:
        """Conteo de incidentes de este tipo por hora del día (array de 24)."""
        return self._por_hora

    def proporcion_horaria(self) -> np.ndarray:
        """Distribución horaria normalizada (suma 1). Útil como evidencia predictiva."""
        s = self._por_hora.sum()
        return self._por_hora / s if s else self._por_hora.astype(float)

    def alcaldia_top(self, n: int = 3) -> list[tuple[str, int]]:
        """Las n alcaldías donde más ocurre este tipo de accidente."""
        return Counter(self._por_alcaldia).most_common(n)

    def __repr__(self) -> str:
        return f"TipoAccidente({self.nombre}, {self.total:,}, {self.porcentaje():.1f}%)"


class CatalogoTipos:
    """
    Conjunto de las 7 TipoAccidente calculadas sobre un DataFrame (o un
    subconjunto, p. ej. una sola alcaldía). Construcción 100% vectorizada
    para que el dashboard responda al instante sin crear 500k objetos.
    """

    def __init__(self, tipos: dict[str, TipoAccidente], total: int) -> None:
        self._tipos = tipos
        self.total  = total

    @classmethod
    def desde_dataframe(cls, df, alcaldia: Optional[str] = None) -> "CatalogoTipos":
        """
        Construye el catálogo de tipos. Si se pasa `alcaldia`, filtra a esa
        demarcación (esto alimenta la gráfica de pastel interactiva).
        """
        d = df
        if alcaldia not in (None, 'Todas', 'Todas las alcaldías'):
            d = d[d['alcaldia_catalogo'] == alcaldia]

        categoria = np.where(
            d['incidente_c4'].isin(CATEGORIAS_PRINCIPALES),
            d['incidente_c4'], CATEGORIA_OTROS,
        )
        d = d.assign(_cat=categoria)
        total = len(d)

        tipos: dict[str, TipoAccidente] = {}
        for nombre, g in d.groupby('_cat'):
            por_hora = np.zeros(24, dtype=int)
            vc = g['hora'].value_counts()
            for h, c in vc.items():
                if 0 <= int(h) < 24:
                    por_hora[int(h)] = int(c)
            tipos[nombre] = TipoAccidente(
                nombre       = nombre,
                total        = len(g),
                graves       = int(g['es_grave'].astype(bool).sum()),
                por_hora     = por_hora,
                por_alcaldia = g['alcaldia_catalogo'].value_counts().to_dict(),
                total_global = total,
            )

        # Garantizar las 7 categorías aunque alguna no aparezca en el filtro
        for nombre in CATEGORIAS_7:
            tipos.setdefault(nombre, TipoAccidente(
                nombre, 0, 0, np.zeros(24, dtype=int), {}, total))

        return cls(tipos, total)

    # ── Consultas ─────────────────────────────────────────────

    def tipos(self) -> list[TipoAccidente]:
        """Las 7 categorías ordenadas de mayor a menor volumen."""
        return sorted(self._tipos.values(), key=lambda t: t.total, reverse=True)

    def top(self, n: int = 3) -> list[TipoAccidente]:
        """Los n tipos más frecuentes (los 'predominantes' del pastel)."""
        return self.tipos()[:n]

    def obtener(self, nombre: str) -> Optional[TipoAccidente]:
        return self._tipos.get(nombre)

    def distribucion(self) -> dict[str, float]:
        """{categoria: porcentaje} listo para graficar el pastel."""
        return {t.nombre: t.porcentaje() for t in self.tipos()}

    def __repr__(self) -> str:
        return f"CatalogoTipos({len(self._tipos)} tipos, {self.total:,} incidentes)"


# ══════════════════════════════════════════════════════════════
#  FUNCIÓN AUXILIAR — cargar_dataset
# ══════════════════════════════════════════════════════════════

def cargar_dataset(ruta_csv: str, limite: Optional[int] = None) -> list[Incidente]:
    """
    Lee viales_limpio.csv y construye la lista de objetos Incidente.
    Cada fila del CSV se convierte en:
      Incidente → ReporteC4 + UbicacionGeografica

    Args:
        ruta_csv: Ruta al archivo viales_limpio.csv
        limite:   Si se especifica, carga solo las primeras N filas (útil para pruebas)

    Returns:
        Lista de objetos Incidente listos para poblar Colonia y Alcaldia.
    """
    import pandas as pd

    df = pd.read_csv(ruta_csv, nrows=limite)
    incidentes: list[Incidente] = []

    for _, fila in df.iterrows():
        reporte = ReporteC4(
            dt_creacion  = datetime.fromisoformat(str(fila['dt_creacion'])),
            dt_cierre    = datetime.fromisoformat(str(fila['dt_cierre'])),
            dia_semana   = str(fila['dia_semana']),
            tipo_entrada = str(fila['tipo_entrada']),
        )
        ubicacion = UbicacionGeografica(
            latitud  = float(fila['latitud']),
            longitud = float(fila['longitud']),
            alcaldia = str(fila['alcaldia_catalogo']),
            colonia  = str(fila['colonia_catalogo']),
        )
        incidente = Incidente(
            folio         = str(fila['folio']),
            tipo          = str(fila['tipo_incidente_c4']),
            subtipo       = str(fila['incidente_c4']),
            codigo_cierre = str(fila['codigo_cierre']),
            clasificacion = str(fila['clas_con_f_alarma']),
            reporte       = reporte,
            ubicacion     = ubicacion,
        )
        incidentes.append(incidente)

    print(f"[cargar_dataset] {len(incidentes):,} incidentes cargados desde {ruta_csv}")
    return incidentes


def construir_alcaldias(incidentes: list[Incidente]) -> dict[str, Alcaldia]:
    """
    Recibe la lista de Incidente y construye el árbol:
      Alcaldia → Colonia → [Incidente]

    Returns:
        dict {nombre_alcaldia: Alcaldia}
    """
    alcaldias: dict[str, Alcaldia] = {}

    for inc in incidentes:
        nombre_alc = inc.ubicacion.alcaldia
        nombre_col = inc.ubicacion.colonia

        if nombre_alc not in alcaldias:
            alcaldias[nombre_alc] = Alcaldia(nombre_alc)

        alc = alcaldias[nombre_alc]
        col = alc.obtener_colonia(nombre_col)

        if col is None:
            col = Colonia(nombre_col, nombre_alc)
            alc.agregar_colonia(col)

        col.agregar(inc)

    print(f"[construir_alcaldias] {len(alcaldias)} alcaldías | "
          f"{sum(len(a.colonias()) for a in alcaldias.values())} colonias")
    return alcaldias

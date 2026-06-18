"""
predictivo.py
=============
Predice el TIPO de accidente más probable dada una alcaldía y una hora.

El predictivo NO inventa variables nuevas: reutiliza el conocimiento que
arrojó el análisis descriptivo (la distribución de tipos cambia por
alcaldía y por hora) y lo formaliza con dos enfoques complementarios:

  • Baseline  — frecuencias condicionadas P(tipo | alcaldía, hora) leídas
                directo de los datos. Es el conocimiento descriptivo "crudo".
  • Modelo    — Naive Bayes categórico, la versión suavizada y generalizadora
                de esas frecuencias: entrega una probabilidad para cada uno
                de los 7 tipos aunque la combinación exacta tenga pocos casos.

Sin Random Forest. Variables: hora, mes, fin de semana y alcaldía
(las cuatro que el descriptivo señaló como relevantes).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from modelos import CATEGORIAS_7, agregar_categoria

BASE_DIR     = Path(__file__).parent
RUTA_PARQUET = BASE_DIR / "datos" / "viales_limpio.parquet"
RUTA_CSV     = BASE_DIR / "datos" / "viales_limpio.csv"
RUTA_GRAFICAS = BASE_DIR / "graficas"

# Pocos casos en una celda (alcaldía, hora) => mezclamos con el marginal
MIN_CASOS_CELDA = 30
ALPHA = 1.0   # suavizado de Laplace


# ══════════════════════════════════════════════════════════════
#  CLASE — PredictorTipoAccidente
# ══════════════════════════════════════════════════════════════

class PredictorTipoAccidente:
    """
    Encapsula el modelo predictivo de tipo de accidente.
    Expone las dos vías (modelo y baseline) y las métricas de validación.
    """

    def __init__(self, alpha: float = ALPHA) -> None:
        self.le_alc   = LabelEncoder()
        self.modelo   = CategoricalNB(alpha=alpha)
        self.alpha    = alpha
        self.clases_: list[str] = []          # orden de clases del modelo
        self._t_ah = None                     # conteos (alcaldía, hora, tipo)
        self._t_a  = None                     # conteos (alcaldía, tipo)
        self._t_g  = None                     # conteos (tipo) global
        self.metricas: dict = {}
        self.matriz_confusion = None
        self.entrenado = False

    # ── Entrenamiento ─────────────────────────────────────────

    def entrenar(self, df: pd.DataFrame, test_size: float = 0.2,
                 seed: int = 42) -> "PredictorTipoAccidente":
        df = agregar_categoria(df)

        # --- Baseline: tablas de frecuencia condicionada ---
        self._t_ah = df.groupby(['alcaldia_catalogo', 'hora', 'categoria'], observed=True).size()
        self._t_a  = df.groupby(['alcaldia_catalogo', 'categoria'], observed=True).size()
        self._t_g  = df.groupby('categoria', observed=True).size().reindex(CATEGORIAS_7, fill_value=0)

        # --- Modelo: Naive Bayes categórico ---
        alc_str = df['alcaldia_catalogo'].astype(str).to_numpy()
        self.le_alc.fit(alc_str)
        X = np.column_stack([
            df['hora'].to_numpy(dtype='int64'),
            df['mes'].to_numpy(dtype='int64'),
            df['es_fin_semana'].astype(int).to_numpy(dtype='int64'),
            self.le_alc.transform(alc_str),
        ]).astype('int64')
        y = df['categoria'].astype(str).to_numpy()

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y)

        self.modelo.fit(X_tr, y_tr)
        self.clases_ = [str(c) for c in self.modelo.classes_]

        # --- Validación ---
        y_pred = self.modelo.predict(X_te)
        self.metricas = {
            'accuracy':  accuracy_score(y_te, y_pred),
            'f1_macro':  f1_score(y_te, y_pred, average='macro'),
            'n_test':    len(y_te),
        }
        self.matriz_confusion = confusion_matrix(y_te, y_pred, labels=self.clases_)
        self.entrenado = True
        return self

    # ── Predicción: modelo ────────────────────────────────────

    def predecir_modelo(self, alcaldia: str, hora: int,
                        mes: int = 6, fin_semana: bool = False) -> dict[str, float]:
        """Probabilidad de cada tipo según el Naive Bayes. {tipo: prob} ordenado desc."""
        try:
            alc_enc = int(self.le_alc.transform([alcaldia])[0])
        except ValueError:
            alc_enc = 0
        x = np.array([[int(hora), int(mes), int(fin_semana), alc_enc]])
        proba = self.modelo.predict_proba(x)[0]
        d = {clase: float(p) for clase, p in zip(self.clases_, proba)}
        return dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True))

    # ── Predicción: baseline (frecuencias condicionadas) ──────

    def predecir_baseline(self, alcaldia: str, hora: int) -> dict[str, float]:
        """
        P(tipo | alcaldía, hora) leída de los datos. Si la celda exacta tiene
        pocos casos, cae al marginal de la alcaldía y, en último caso, al global.
        """
        conteos = self._celda(alcaldia, hora)
        total = conteos.sum()
        probs = (conteos + self.alpha) / (total + self.alpha * len(CATEGORIAS_7))
        d = {cat: float(probs[cat]) for cat in CATEGORIAS_7}
        return dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True))

    def _celda(self, alcaldia: str, hora: int) -> pd.Series:
        """Devuelve los conteos por tipo para (alcaldía, hora) con respaldo jerárquico."""
        vacio = pd.Series(0, index=CATEGORIAS_7)
        try:
            celda = self._t_ah.loc[(alcaldia, hora)].reindex(CATEGORIAS_7, fill_value=0)
        except KeyError:
            celda = vacio
        if celda.sum() >= MIN_CASOS_CELDA:
            return celda
        try:
            marg = self._t_a.loc[alcaldia].reindex(CATEGORIAS_7, fill_value=0)
        except KeyError:
            marg = vacio
        if marg.sum() > 0:
            return celda + marg          # mezcla celda + marginal de la alcaldía
        return celda + self._t_g          # último respaldo: distribución global

    # ── Comparación de ambos enfoques ─────────────────────────

    def predecir(self, alcaldia: str, hora: int,
                 mes: int = 6, fin_semana: bool = False) -> pd.DataFrame:
        """Tabla {tipo, prob_modelo, prob_baseline} ordenada por el modelo."""
        m = self.predecir_modelo(alcaldia, hora, mes, fin_semana)
        b = self.predecir_baseline(alcaldia, hora)
        filas = [{'tipo': t,
                  'prob_modelo':   m.get(t, 0.0),
                  'prob_baseline': b.get(t, 0.0)} for t in CATEGORIAS_7]
        return (pd.DataFrame(filas)
                  .sort_values('prob_modelo', ascending=False)
                  .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════
#  CARGA DE DATOS (parquet si existe, si no CSV)
# ══════════════════════════════════════════════════════════════

def cargar_df() -> pd.DataFrame:
    if RUTA_PARQUET.exists():
        return pd.read_parquet(RUTA_PARQUET)
    return pd.read_csv(RUTA_CSV)


# ══════════════════════════════════════════════════════════════
#  EXPORTACIÓN DE FIGURAS ESTÁTICAS (para el reporte escrito)
# ══════════════════════════════════════════════════════════════

def exportar_figuras(pred: "PredictorTipoAccidente",
                     ejemplos=(("Cuauhtémoc", 19), ("Iztapalapa", 8))) -> None:
    """Genera PNGs del predictivo en graficas/ usando matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    RUTA_GRAFICAS.mkdir(exist_ok=True)
    PALETA_PRINCIPAL = "#D85A30"
    PALETA_LEVE      = "#378ADD"

    # --- Matriz de confusión (normalizada por fila) ---
    cm = pred.matriz_confusion.astype(float)
    cm_norm = np.divide(cm, cm.sum(axis=1, keepdims=True),
                        out=np.zeros_like(cm), where=cm.sum(axis=1, keepdims=True) != 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(len(pred.clases_)))
    ax.set_yticks(range(len(pred.clases_)))
    ax.set_xticklabels(pred.clases_, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(pred.clases_, fontsize=8)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center',
                    fontsize=7, color='black' if cm_norm[i, j] < 0.5 else 'white')
    ax.set_title(f"Matriz de confusión — Naive Bayes\n"
                 f"accuracy={pred.metricas['accuracy']:.3f} · "
                 f"F1 macro={pred.metricas['f1_macro']:.3f}",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(RUTA_GRAFICAS / "P1_matriz_confusion.png", dpi=150)
    plt.close()

    # --- Comparación modelo vs baseline para casos de ejemplo ---
    for alc, hora in ejemplos:
        tabla = pred.predecir(alc, hora)
        fig, ax = plt.subplots(figsize=(9, 5))
        y = np.arange(len(tabla))
        ax.barh(y - 0.2, tabla['prob_modelo'] * 100, height=0.4,
                color=PALETA_PRINCIPAL, label='Modelo (Naive Bayes)')
        ax.barh(y + 0.2, tabla['prob_baseline'] * 100, height=0.4,
                color=PALETA_LEVE, label='Baseline (frecuencias)')
        ax.set_yticks(y); ax.set_yticklabels(tabla['tipo'], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Probabilidad (%)")
        ax.set_title(f"Tipo de accidente más probable — {alc}, {hora}:00 h",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(RUTA_GRAFICAS / f"P2_pred_{alc.split()[0]}_{hora}h.png", dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL  (python main.py --paso predictivo)
# ══════════════════════════════════════════════════════════════

def predecir() -> None:
    print("╔══════════════════════════════════════════╗")
    print("║   ANÁLISIS PREDICTIVO — VIALES CDMX      ║")
    print("╚══════════════════════════════════════════╝\n")

    df = cargar_df()
    print(f"[DATOS] {len(df):,} registros")

    print("[ENTRENA] Naive Bayes categórico + baseline de frecuencias...")
    pred = PredictorTipoAccidente().entrenar(df)

    print(f"\n  Accuracy : {pred.metricas['accuracy']:.3f}")
    print(f"  F1 macro : {pred.metricas['f1_macro']:.3f}")
    print(f"  Clases   : {pred.clases_}")

    print("\n  Ejemplo — Cuauhtémoc a las 19:00 h:")
    print(pred.predecir("Cuauhtémoc", 19).to_string(index=False,
          formatters={'prob_modelo': '{:.1%}'.format,
                      'prob_baseline': '{:.1%}'.format}))

    print("\n[FIGURAS] Exportando PNGs a graficas/...")
    exportar_figuras(pred)
    print("  ✓ P1_matriz_confusion.png, P2_pred_*.png")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    predecir()

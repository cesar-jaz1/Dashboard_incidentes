"""
main.py
=======
Pipeline OFFLINE del proyecto. Genera los artefactos del reporte escrito:

  limpieza.py    : limpia el CSV original → viales_limpio.csv + viales_limpio.parquet
  descriptivo.py : exporta las 6 gráficas descriptivas (PNG) a graficas/
  predictivo.py  : entrena el modelo y exporta sus gráficas (PNG) a graficas/

El TABLERO INTERACTIVO es aparte (no se corre desde aquí):
    streamlit run app.py

Uso:
    python main.py                    # corre los 3 pasos
    python main.py --paso limpieza    # solo limpieza
    python main.py --paso descriptivo # solo gráficas descriptivas
    python main.py --paso predictivo  # solo entrena + gráficas predictivas
"""

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent


# ── Verificación de archivos ──────────────────────────────

def verificar_csv_original() -> bool:
    ruta = BASE_DIR / "datos" / "inViales_2022_2024.csv"
    if not ruta.exists():
        print(f"\n[ERROR] No se encontró el archivo: {ruta}")
        print("        Coloca el CSV en la carpeta 'datos/' e intenta de nuevo.\n")
        return False
    return True


def verificar_datos_limpios() -> bool:
    """El descriptivo/predictivo necesitan el parquet o el CSV limpio."""
    parquet = BASE_DIR / "datos" / "viales_limpio.parquet"
    csv     = BASE_DIR / "datos" / "viales_limpio.csv"
    if not parquet.exists() and not csv.exists():
        print(f"\n[ERROR] No se encontró viales_limpio.parquet ni viales_limpio.csv")
        print("        Primero ejecuta: python main.py --paso limpieza\n")
        return False
    return True


# ── Pasos ─────────────────────────────────────────────────

def paso_limpieza() -> bool:
    print("\n" + "─" * 50)
    print("  PASO 1 — Limpieza de datos")
    print("─" * 50)
    if not verificar_csv_original():
        return False
    from limpieza import limpiar
    limpiar()
    return True


def paso_descriptivo() -> bool:
    print("\n" + "─" * 50)
    print("  PASO 2 — Análisis descriptivo (gráficas PNG)")
    print("─" * 50)
    if not verificar_datos_limpios():
        return False
    from descriptivo import analizar
    analizar()
    return True


def paso_predictivo() -> bool:
    print("\n" + "─" * 50)
    print("  PASO 3 — Análisis predictivo (modelo + gráficas PNG)")
    print("─" * 50)
    if not verificar_datos_limpios():
        return False
    from predictivo import predecir
    predecir()
    return True


# ── Pipeline completo ─────────────────────────────────────

def correr_todo() -> None:
    pasos = [
        ("limpieza",    paso_limpieza),
        ("descriptivo", paso_descriptivo),
        ("predictivo",  paso_predictivo),
    ]
    inicio_total = time.time()
    for nombre, funcion in pasos:
        inicio = time.time()
        if not funcion():
            print(f"\n[DETENIDO] El paso '{nombre}' falló. Revisa el error de arriba.\n")
            sys.exit(1)
        print(f"\n  ✓ Paso '{nombre}' completado en {time.time() - inicio:.1f}s")

    print("\n" + "═" * 50)
    print(f"  Pipeline completo en {time.time() - inicio_total:.1f}s")
    print(f"  Gráficas guardadas en: graficas/")
    print(f"  Para abrir el tablero: streamlit run app.py")
    print("═" * 50 + "\n")


# ── CLI ───────────────────────────────────────────────────

def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline offline de incidentes viales CDMX 2022–2024",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--paso",
        choices=["limpieza", "descriptivo", "predictivo"],
        default=None,
        help=("Paso específico a ejecutar (sin --paso se ejecutan los 3 en orden).\n"
              "El tablero interactivo se abre aparte: streamlit run app.py"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # consolas Windows (cp1252)
    except Exception:
        pass
    print("╔══════════════════════════════════════════════╗")
    print("║  ANÁLISIS DE INCIDENTES VIALES CDMX 2022–24  ║")
    print("╚══════════════════════════════════════════════╝")

    args = parsear_argumentos()
    if args.paso == "limpieza":
        ok = paso_limpieza()
    elif args.paso == "descriptivo":
        ok = paso_descriptivo()
    elif args.paso == "predictivo":
        ok = paso_predictivo()
    else:
        correr_todo()
        ok = True

    sys.exit(0 if ok else 1)

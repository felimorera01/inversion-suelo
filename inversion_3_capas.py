"""Inversión eléctrica 1D de tres capas para un arreglo Wenner superficial.

Ejecutar: python -m streamlit run inversion_3_capas.py
Pruebas matemáticas: python inversion_3_capas.py --verificar
Python >= 3.10. Dependencias: numpy scipy pandas plotly streamlit openpyxl libdlf.

Modelo: capas planas, homogéneas, isótropas, de extensión lateral infinita;
electrodos puntuales en superficie, corriente continua. La capa 3 es un
semiespacio: se estiman rho1, rho2, rho3, h1 y h2, nunca un espesor h3.
Es una extensión numérica de la interpretación estratificada, NO la aplicación
literal de las curvas de Sunde (dos capas), ni una certificación IEEE 80.

Referencias técnicas:
- Material docente: public_04_Sistemas_de_Puesta_a_Tierra-Diseño.pdf.
- scipy.optimize.least_squares: https://docs.scipy.org/doc/scipy/reference/
  generated/scipy.optimize.least_squares.html
- Filtro Hankel J0 de Key (2012), distribuido por libdlf:
  https://github.com/emsig/libdlf (coeficientes y atribución en la dependencia).
"""
from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

A = "a_m"
RHO = "rho_aparente_ohm_m"
R = "resistencia_ohm"
NOMBRES = ["rho1", "rho2", "rho3", "h1", "h2"]


def ejemplo() -> pd.DataFrame:
    """Datos transcritos de la columna RESISTIVIDAD APARENTE de la imagen.

    No se recalculan desde la resistencia: las columnas están redondeadas.
    """
    return pd.DataFrame({
        A: [0.31, 0.92, 1.52, 3.05, 4.57, 6.10, 9.15, 15.24,
            21.34, 27.44, 33.54, 39.63, 45.73],
        R: [109.38, 48.84, 30.40, 15.01, 9.42, 6.48, 3.52, 1.50,
            0.90, 0.64, 0.51, 0.42, 0.36],
        RHO: [209.5, 280.6, 290.9, 287.3, 270.5, 248.2, 202.2,
              143.6, 120.6, 110.3, 107.4, 104.5, 103.4],
    })


def validar(tabla: pd.DataFrame) -> pd.DataFrame:
    """Ignora filas vacías; rechaza filas incompletas, no numéricas o <= 0."""
    datos = tabla[[A, RHO]].copy().replace(r"^\s*$", np.nan, regex=True)
    datos = datos.dropna(how="all")
    for col in (A, RHO):
        # Admite coma decimal en celdas importadas, sin separadores de millares.
        datos[col] = pd.to_numeric(
            datos[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    if not np.isfinite(datos.to_numpy(dtype=float)).all():
        raise ValueError("Hay celdas incompletas o no numéricas. Revise ambas columnas.")
    if (datos.to_numpy() <= 0).any():
        raise ValueError("Todas las separaciones y resistividades deben ser mayores que cero.")
    if len(datos) < 6 or datos[A].nunique() < 6:
        raise ValueError("Ingrese al menos 6 separaciones distintas para ajustar 5 parámetros.")
    if len(datos) > 500:
        raise ValueError("Use como máximo 500 mediciones por inversión.")
    return datos.sort_values(A, kind="stable").reset_index(drop=True)


def completar_tabla(tabla: pd.DataFrame, magnitud: str) -> pd.DataFrame:
    """Construye a, R y rho_a usando la magnitud elegida como dato fuente.

    Wenner superficial: rho_a = 2*pi*a*R; inversa R = rho_a/(2*pi*a).
    No mezcla columnas originales redondeadas con columnas calculadas.
    """
    if magnitud not in (R, RHO):
        raise ValueError("Seleccione resistencia o resistividad aparente.")
    # Reutiliza la validación numérica, sin imponer unidades al dato temporal.
    base = validar(tabla[[A, magnitud]].rename(columns={magnitud: RHO}))
    if magnitud == R:
        base[R] = base[RHO]
        base[RHO] = 2 * np.pi * base[A] * base[R]
    else:
        base[R] = base[RHO] / (2 * np.pi * base[A])
    if not np.isfinite(base[[A, R, RHO]].to_numpy()).all():
        raise ValueError("La conversión excede el rango numérico. Revise las unidades.")
    return base[[A, R, RHO]]


@lru_cache(maxsize=1)
def filtro() -> tuple[np.ndarray, np.ndarray]:
    """Abscisas y pesos J0 del filtro digital de Hankel de 201 puntos."""
    from libdlf.hankel import key_201_2012
    base, j0, _j1 = key_201_2012()
    return base, j0


def transformada(lam: np.ndarray, modelo: np.ndarray) -> np.ndarray:
    """Recursión de resistividad transformada desde el semiespacio hacia arriba.

    T3 = rho3
    Ti = rhoi * (T(i+1) + rhoi*tanh(lambda*hi)) /
                  (rhoi + T(i+1)*tanh(lambda*hi))
    lambda tiene unidades 1/m; T tiene unidades ohm*m.
    """
    r1, r2, r3, h1, h2 = modelo
    t = np.full_like(lam, r3, dtype=float)
    for rho, h in ((r2, h2), (r1, h1)):
        u = np.tanh(lam * h)
        t = rho * (t + rho * u) / (rho + t * u)
    return t


def respuesta_wenner(a: np.ndarray, modelo: np.ndarray) -> np.ndarray:
    """Solución física directa mediante transformada de Hankel, no interpolación.

    F(r) = integral_0^inf T1(lambda)*J0(lambda*r) d(lambda)
    rho_a(a) = 2*a*[F(a)-F(2*a)].
    Se integra analíticamente rho1/r, y el filtro evalúa el resto:
    F(r) = rho1/r + sum_j w_j*[T1(b_j/r)-rho1]/r.
    Esto reproduce exactamente el semiespacio homogéneo y reduce cancelaciones.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    modelo = np.asarray(modelo, dtype=float)
    if modelo.shape != (5,) or not np.isfinite(modelo).all() or (modelo <= 0).any():
        raise ValueError("El modelo debe contener cinco parámetros positivos y finitos.")
    if not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError("Las separaciones deben ser positivas y finitas.")
    base, pesos = filtro()
    r = np.stack((a, 2 * a))
    t = transformada(base[None, None, :] / r[:, :, None], modelo)
    correccion = np.sum((t - modelo[0]) * pesos, axis=-1) / r
    return modelo[0] + 2 * a * (correccion[0] - correccion[1])


def rms_porcentual(observado: np.ndarray, calculado: np.ndarray) -> float:
    """RMS relativo: 100*sqrt(mean(((calculado-observado)/observado)**2))."""
    return float(100 * np.sqrt(np.mean(((calculado - observado) / observado) ** 2)))


@dataclass
class Resultado:
    modelo: np.ndarray
    calculado: np.ndarray
    rms: float
    intentos: pd.DataFrame
    convergio: bool
    mensaje: str
    condicion: float
    limites: list[str]


def invertir(datos: pd.DataFrame, rho_min: float = 1., rho_max: float = 10000.,
             h_min: float = 0.01, h_max: float = 1000., n_inicios: int = 24,
             progreso: Callable[[float], None] | None = None) -> Resultado:
    """Mínimos cuadrados relativos con múltiples puntos iniciales reproducibles.

    Se optimizan log10(rho1,rho2,rho3,h1,h2); así todos los parámetros son
    positivos y se equilibran sus escalas. El objetivo coincide con el RMS
    mostrado, sin confundirlo con un RMS de logaritmos. No impone orden entre
    resistividades. Multinicio reduce mínimos locales, pero no prueba unicidad.
    """
    datos = validar(datos)
    if not (0 < rho_min < rho_max and 0 < h_min < h_max):
        raise ValueError("Cada límite mínimo debe ser positivo y menor que su máximo.")
    a, obs = datos[A].to_numpy(), datos[RHO].to_numpy()
    lo = np.log10([rho_min] * 3 + [h_min] * 2)
    hi = np.log10([rho_max] * 3 + [h_max] * 2)
    rng = np.random.default_rng(80)
    inicial = np.log10([obs[0], np.max(obs), obs[-1], a[0], np.median(a)])
    inicios = [np.clip(inicial, lo + 1e-6, hi - 1e-6)]
    inicios.extend(rng.uniform(lo, hi, size=(n_inicios - 1, 5)))

    def residuos(x: np.ndarray) -> np.ndarray:
        return (respuesta_wenner(a, 10. ** x) - obs) / obs

    soluciones = []
    for i, x0 in enumerate(inicios):
        fit = least_squares(residuos, x0, bounds=(lo, hi), method="trf",
                            x_scale="jac", max_nfev=1800,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        soluciones.append(fit)
        if progreso:
            progreso((i + 1) / n_inicios)
    best = min(soluciones, key=lambda s: np.sum(s.fun ** 2))
    modelo = 10. ** best.x
    calc = respuesta_wenner(a, modelo)
    if not np.isfinite(calc).all() or (calc <= 0).any():
        raise ValueError("Respuesta numérica no física. Revise los límites y los datos.")
    filas = []
    for s in soluciones:
        filas.append(dict(zip(NOMBRES, 10. ** s.x),
                          RMS_porcentaje=100 * np.sqrt(np.mean(s.fun ** 2)),
                          convergio=bool(s.success)))
    limites = [n for n, x, l, u in zip(NOMBRES, best.x, lo, hi)
               if min(x - l, u - x) < 0.005 * (u - l)]
    return Resultado(modelo, calc, rms_porcentual(obs, calc),
                     pd.DataFrame(filas).sort_values("RMS_porcentaje"),
                     bool(best.success), str(best.message),
                     float(np.linalg.cond(best.jac)), limites)


def leer_archivo(archivo, separador: str, decimal: str) -> pd.DataFrame:
    """Lee CSV UTF-8 o Excel .xlsx; primera hoja y primera fila como encabezado."""
    contenido = io.BytesIO(archivo.getvalue())
    if archivo.name.lower().endswith(".xlsx"):
        return pd.read_excel(contenido, engine="openpyxl")
    return pd.read_csv(contenido, sep=separador, decimal=decimal, encoding="utf-8-sig")


def main() -> None:
    """Flujo GUI: cargar/editar -> validar -> invertir -> presentar/exportar."""
    import streamlit as st
    import plotly.graph_objects as go

    st.set_page_config(page_title="Suelo • Inversión 1D", page_icon="🌎", layout="wide")
    st.title("Resistividad del suelo · 3 capas")
    st.caption("Inversión 1D • Arreglo Wenner • Resistividades en Ω·m y espesores en m")
    st.info("1. Edite la tabla o cargue un archivo.  2. Revise las unidades.  "
            "3. Pulse Calcular Inversión. El ejemplo IEEE 80 ya está cargado.")
    with st.sidebar:
        st.header("Configuración")
        st.caption("Electrodos equiespaciados y profundidad de inserción despreciable "
                   "frente a a. No se modelan variaciones laterales.")
        n = st.slider("Puntos iniciales", 8, 64, 24, step=8)
        with st.expander("Límites de búsqueda", expanded=False):
            rmin = st.number_input("ρ mínimo (Ω·m)", value=1., min_value=0.001, format="%.3f")
            rmax = st.number_input("ρ máximo (Ω·m)", value=10000., min_value=0.002)
            hmin = st.number_input("h mínimo (m)", value=0.01, min_value=0.0001, format="%.4f")
            hmax = st.number_input("h máximo (m)", value=1000., min_value=0.001)
        st.caption("Los límites son restricciones de búsqueda, no datos geológicos. "
                   "El espesor de la tercera capa es infinito.")
    st.subheader("Datos de campo")
    origen = st.radio("Entrada", ["Ejemplo editable", "Tabla vacía", "Archivo CSV / Excel"], horizontal=True)
    tipo = st.radio("¿Qué valor desea introducir?",
                    ["Resistividad aparente (Ω·m)", "Resistencia (Ω)"], horizontal=True)
    magnitud = R if tipo == "Resistencia (Ω)" else RHO
    if origen == "Ejemplo editable":
        st.markdown("**Tabla original del ejercicio IEEE 80**")
        st.dataframe(ejemplo(), hide_index=True, width="stretch")
        st.caption("Ambas columnas conservan los valores de la imagen. Al estar "
                   "redondeados, pueden diferir de la conversión ρₐ = 2πaR. "
                   "El ajuste usará únicamente la magnitud seleccionada abajo.")
    st.caption("Wenner superficial: ρₐ = 2πaR. a en metros, R en Ω y ρₐ en Ω·m. "
               "Se supone profundidad de inserción despreciable frente a a.")
    tabla = ejemplo()[[A, magnitud]] if origen == "Ejemplo editable" else pd.DataFrame({A: [None]*6, magnitud: [None]*6})
    if origen == "Archivo CSV / Excel":
        archivo = st.file_uploader("Cargue CSV o Excel (.xlsx)", type=["csv", "xlsx"])
        if archivo is None:
            st.stop()
        c1, c2 = st.columns(2)
        sep = c1.selectbox("Separador CSV", [",", ";", "\t"])
        dec = c2.selectbox("Separador decimal CSV", [".", ","])
        try:
            bruto = leer_archivo(archivo, sep, dec)
            if len(bruto.columns) < 2:
                raise ValueError("No se encontraron dos columnas. Revise el separador CSV.")
            c1, c2 = st.columns(2)
            ca = c1.selectbox("Columna de separación (m)", bruto.columns, index=0)
            cr = c2.selectbox(f"Columna de {tipo.lower()}", bruto.columns,
                              index=list(bruto.columns).index(magnitud) if magnitud in bruto else 1)
            if ca == cr:
                raise ValueError("Seleccione columnas diferentes.")
            tabla = bruto[[ca, cr]].copy()
            tabla.columns = [A, magnitud]
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}")
            st.stop()
    st.markdown(f"**Datos editables: {tipo}**")
    st.caption("Puede agregar o borrar filas. En la tabla use punto decimal.")
    editada = st.data_editor(tabla, num_rows="dynamic", width="stretch",
                            key=f"tabla_{origen}_{magnitud}")
    st.download_button("Descargar CSV de ejemplo", ejemplo().to_csv(index=False),
                       "datos_ejemplo.csv", "text/csv")
    try:
        completa = completar_tabla(editada, magnitud)
        datos = completa[[A, RHO]]
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()
    st.markdown("**Tabla completa para el cálculo**")
    st.caption("Resistividad calculada a partir de la resistencia ingresada."
               if magnitud == R else
               "Resistencia equivalente calculada a partir de la resistividad ingresada.")
    st.dataframe(completa, hide_index=True, width="stretch")
    st.download_button("Descargar tabla completa", completa.to_csv(index=False),
                       "datos_completos.csv", "text/csv")
    if datos[A].duplicated().any():
        st.caption("Se conservarán las mediciones repetidas; cada fila tiene el mismo peso relativo.")
    # La firma evita mostrar resultados obsoletos después de modificar la entrada.
    firma = (completa.to_csv(index=False), magnitud, n, rmin, rmax, hmin, hmax)
    if st.button("Calcular Inversión", type="primary", width="stretch"):
        st.session_state.pop("resultado", None)
        barra = st.progress(0., text="Ajustando los cinco parámetros…")
        try:
            with st.spinner("Buscando el mejor ajuste desde distintos modelos iniciales…"):
                resultado = invertir(datos, rmin, rmax, hmin, hmax, n, barra.progress)
            st.session_state["resultado"] = (firma, resultado)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            st.error(f"No se completó la inversión: {exc}")
        finally:
            barra.empty()
    guardado = st.session_state.get("resultado")
    if guardado is None:
        st.stop()
    if guardado[0] != firma:
        st.info("La entrada cambió. Pulse Calcular Inversión para actualizar los resultados.")
        st.stop()
    resultado = guardado[1]
    r1, r2, r3, h1, h2 = resultado.modelo
    st.subheader("Modelo de mejor ajuste encontrado")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capa 1 · ρ₁", f"{r1:.3f} Ω·m", help=f"Espesor: {h1:.4f} m")
    c2.metric("Capa 2 · ρ₂", f"{r2:.3f} Ω·m", help=f"Espesor: {h2:.4f} m")
    c3.metric("Capa 3 · ρ₃", f"{r3:.3f} Ω·m", help="Semiespacio inferior")
    c4.metric("Error RMS relativo", f"{resultado.rms:.3f} %")
    c1, c2, c3 = st.columns(3)
    c1.metric("Espesor h₁", f"{h1:.4f} m")
    c2.metric("Espesor h₂", f"{h2:.4f} m")
    c3.metric("Espesor capa 3", "∞")
    capas = pd.DataFrame({"Capa": [1, 2, 3], "Resistividad (Ω·m)": [r1, r2, r3],
                           "Espesor (m)": [h1, h2, np.inf],
                           "Profundidad superior (m)": [0., h1, h1+h2],
                           "Profundidad inferior (m)": [h1, h1+h2, np.inf]})
    st.dataframe(capas, hide_index=True, width="stretch")
    st.caption("∞ = semiespacio. h₂ es el espesor de la segunda capa; "
               "su base se encuentra a h₁ + h₂.")
    if not resultado.convergio:
        st.warning("El mejor intento agotó el criterio de parada sin converger. " + resultado.mensaje)
    if resultado.limites:
        st.warning("Parámetros cercanos a los límites: " + ", ".join(resultado.limites) +
                   ". Revise si el intervalo de búsqueda es adecuado.")
    if resultado.condicion > 1e6:
        st.warning("Sensibilidad mal condicionada: algunos parámetros están poco determinados.")
    st.caption("El modelo no es necesariamente único. Un RMS bajo mide el ajuste, "
               "no demuestra que existan exactamente tres capas ni determina su incertidumbre.")
    a = datos[A].to_numpy()
    suave = np.geomspace(a.min(), a.max(), 240)
    st.subheader("Curva de mejor ajuste")
    st.markdown("**Línea azul gruesa: mejor ajuste calculado.** "
                "Los círculos naranjas representan las mediciones de campo.")
    curva = respuesta_wenner(suave, resultado.modelo)
    fig = go.Figure()
    fig.add_scatter(
        x=suave, y=curva, mode="lines", name="MEJOR AJUSTE · modelo de 3 capas",
        line=dict(color="#155EEF", width=5),
        hovertemplate="<b>Curva de mejor ajuste</b><br>a = %{x:.3f} m"
                      "<br>ρ aparente = %{y:.3f} Ω·m<extra></extra>")
    fig.add_scatter(
        x=a, y=datos[RHO], mode="markers", name="Datos de campo · mediciones",
        marker=dict(color="#F59E0B", size=11, symbol="circle",
                    line=dict(color="#78350F", width=1.5)),
        hovertemplate="<b>Medición de campo</b><br>a = %{x:.3f} m"
                      "<br>ρ aparente = %{y:.3f} Ω·m<extra></extra>")
    # En ejes logarítmicos Plotly recibe log10 en las coordenadas de anotación.
    # Se señala un punto real de la curva, sin desplazar ni alterar el ajuste.
    indice = int(0.60 * (len(suave) - 1))
    fig.add_annotation(
        x=float(np.log10(suave[indice])), y=float(np.log10(curva[indice])),
        xref="x", yref="y", text="<b>CURVA DE MEJOR AJUSTE</b>",
        showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#155EEF",
        ax=0, ay=-65, font=dict(size=14, color="#155EEF"),
        bgcolor="white", bordercolor="#155EEF", borderwidth=1, borderpad=7)
    fig.update_layout(
        template="plotly_white", height=560,
        title=dict(text=f"Ajuste de tres capas · RMS = {resultado.rms:.3f} %",
                   font=dict(size=20)),
        xaxis=dict(type="log", title="Separación a (m)", gridcolor="#E5E7EB"),
        yaxis=dict(type="log", title="Resistividad aparente (Ω·m)",
                   gridcolor="#E5E7EB"),
        legend=dict(orientation="h", yanchor="bottom", y=1.03,
                    xanchor="left", x=0, font=dict(size=14), itemsizing="constant"),
        margin=dict(t=135, b=65, l=75, r=35), hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    comparacion = completa.copy()
    comparacion["rho_calculada_ohm_m"] = resultado.calculado
    comparacion["error_relativo_porcentaje"] = 100*(resultado.calculado-datos[RHO])/datos[RHO]
    with st.expander("Mediciones, residuos y diagnóstico"):
        st.dataframe(comparacion, hide_index=True)
        st.write("Número de condición del Jacobiano:", f"{resultado.condicion:.3g}")
        st.write("Estado del optimizador:", resultado.mensaje)
        st.write("Resultados de todos los puntos iniciales (no son intervalos de confianza):")
        st.dataframe(resultado.intentos, hide_index=True)
    with st.expander("Matemática y alcance"):
        st.latex(r"\mathrm{RMS}(\%)=100\sqrt{\frac{1}{N}\sum_{i=1}^{N}"
                 r"\left(\frac{\rho_{a,i}^{calc}-\rho_{a,i}^{obs}}{\rho_{a,i}^{obs}}\right)^2}")
        st.write("Se minimiza la suma de cuadrados de los residuos relativos. "
                 "Cada medición tiene igual peso relativo. La solución directa utiliza "
                 "la recursión de resistividades y un filtro digital de Hankel J₀. "
                 "Sunde es un método gráfico de dos capas; esta aplicación resuelve "
                 "el problema físico de tres capas mediante mínimos cuadrados con límites.")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Descargar estratos", capas.to_csv(index=False), "estratos.csv", "text/csv")
    c2.download_button("Descargar ajuste", comparacion.to_csv(index=False), "ajuste.csv", "text/csv")
    c3.download_button("Descargar gráfico", fig.to_html(include_plotlyjs=True), "ajuste.html", "text/html")


def verificar() -> None:
    """Pruebas físicas independientes: homogéneo, serie de 2 capas e inversión."""
    a = np.geomspace(0.1, 100., 70)
    np.testing.assert_allclose(respuesta_wenner(a, [100,100,100,2,5]), 100., atol=1e-10)
    # Solución analítica de dos capas por imágenes, con rho2=rho3.
    # rho_a = rho1*[1+4*sum k^n*(1/sqrt(1+(2nh/a)^2)
    #                              -1/sqrt(4+(2nh/a)^2))].
    for r2 in (10., 1000.):
        r1, h = 100., 2.
        k = (r2-r1)/(r2+r1)
        n = np.arange(1, 1501)[:, None]
        z = 2*n*h/a
        exacto = r1*(1+4*np.sum(k**n*(1/np.sqrt(1+z*z)-1/np.sqrt(4+z*z)), axis=0))
        np.testing.assert_allclose(respuesta_wenner(a, [r1,r2,r2,h,8]), exacto, rtol=2e-4)
    real = np.array([120., 450., 65., 0.8, 7.])
    sintetico = pd.DataFrame({A:a, RHO:respuesta_wenner(a, real)})
    rec = invertir(sintetico, n_inicios=12)
    np.testing.assert_allclose(rec.modelo, real, rtol=0.01)
    print("OK: semiespacio homogéneo, series de dos capas e inversión sintética.")
    res = invertir(ejemplo())
    print("Ejemplo IEEE 80:", dict(zip(NOMBRES, res.modelo)))
    print(f"RMS relativo = {res.rms:.6f} %; convergencia = {res.convergio}")


if __name__ == "__main__":
    if "--verificar" in sys.argv:
        verificar()
    else:
        main()

# %% Importar librerias
from pathlib import Path 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA 
from scipy.stats import norm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox

# %% Direccion base de datos
Base_dir = Path(__file__).resolve().parents[1]
Datos_dir = Base_dir / "Datos"
Ruta_exp = Datos_dir / "ITCR colombia.csv"

# %% Cargar datos
df = pd.read_csv(Ruta_exp)

# %% Preparamos los datos
df.columns = ["Fecha", "ITCR_ipp", "ITCR_ipc"]
df["Fecha"] = pd.to_datetime(df["Fecha"])
df = df.set_index("Fecha")
df.index = pd.DatetimeIndex(df.index, freq="ME")
df= df.loc["2000":]

# %% Grafico serie de tiempo en nivel
sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
plt.plot(df["ITCR_ipc"], color="brown", linewidth=1.5)
plt.title("Serie de tiempo del ITCR (IPC)", fontsize=16)
plt.xlabel("Fecha")
plt.ylabel("indice")
plt.show()

# %%Revisamos la estacionariedad, y si toca, la corregimos
Resultado = adfuller(df["ITCR_ipc"], autolag="AIC")
print("ADF estadistico:", Resultado[0])
print("p-valor:", Resultado[1])
# ajustamos varianza y media
df["log_ipc"]=np.log(df["ITCR_ipc" ])
df["ipc_porcentual"]=df["log_ipc"].diff()

# %% dicky fuller despues de la transformacion
Resultado2 = adfuller(df["ipc_porcentual"].dropna(), autolag="AIC")
print("ADF estadistico (transformado):", Resultado2[0])
print("p-valor (transformado):", Resultado2[1]) 

# %% Grafica con var y media estabilizada
sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
plt.plot(df["ipc_porcentual"], color="brown", linewidth=1.5)
plt.title("ITCR (IPC) Var y media estabilizada", fontsize=16)
plt.xlabel("Fecha")
plt.ylabel("indice porcentual")
plt.show()

# %% Graficamos FAC y FACP
series = df["ipc_porcentual"].dropna()
fig, axes = plt.subplots(2, 1, figsize=(10,8))
plot_acf(series,lags=36, alpha=0.05, ax=axes[0],linewidth=1,markersize=0, title="FAC ITCR(IPC)")
plot_pacf(series,lags=36, alpha=0.05, method="ywm", ax=axes[1],linewidth=1,markersize=0, title="FACP ITCR(IPC)")
plt.show()

# %% Ralizamos pruebas aic y bic para determinar el modelo arima, y su tabla

# %% Estimación de los modelos
ordenes = [(1, 1, 0), (1, 1, 1), (0, 1, 1)]
nombres = [f"ARIMA({p},{d},{q})" for p, d, q in ordenes]
resultados = []
for orden in ordenes:
    res = ARIMA(df["log_ipc"], order=orden, trend="n").fit()
    resultados.append(res)
    print(orden, "AIC =", round(res.aic, 2), "BIC =", round(res.bic, 2))


# %% Tabla gráfica
def valor(res, nombre, tipo):
    """Devuelve coeficiente o error estándar formateado ('' si el parámetro no existe)."""
    if nombre not in res.params.index:
        return ""
    if tipo == "coef":
        return f"{res.params[nombre]:.3f}"
    return f"({res.bse[nombre]:.3f})"
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0.1, 1)
x_lab = 0.03
x_cols = [0.40, 0.62, 0.84]
x_ini, x_fin = 0.02, 0.98
# Regla superior y encabezado
ax.hlines(0.92, x_ini, x_fin, color="black", lw=1.3)
for x, nom in zip(x_cols, nombres):
    ax.text(x, 0.86, nom, ha="center", va="center", fontsize=12)
ax.hlines(0.80, x_ini, x_fin, color="black", lw=0.8)
# Filas de parámetros: (etiqueta, nombre del parámetro en statsmodels, posición y)
filas = [
    (r"$\hat{a}_1$",    "ar.L1", 0.73),
    (r"$\hat{\beta}_1$", "ma.L1", 0.57),
]
for etiqueta, par, y in filas:
    ax.text(x_lab, y, etiqueta, ha="left", va="center", fontsize=12)
    for x, res in zip(x_cols, resultados):
        ax.text(x, y,        valor(res, par, "coef"), ha="center", va="center", fontsize=12)
        ax.text(x, y - 0.07, valor(res, par, "se"),   ha="center", va="center", fontsize=12)
# Criterios de información
ax.text(x_lab, 0.40, "AIC", ha="left", va="center", fontsize=12)
ax.text(x_lab, 0.32, "BIC", ha="left", va="center", fontsize=12)
for x, res in zip(x_cols, resultados):
    ax.text(x, 0.40, f"{res.aic:.1f}", ha="center", va="center", fontsize=12)
    ax.text(x, 0.32, f"{res.bic:.1f}", ha="center", va="center", fontsize=12)
# Regla inferior y nota al pie
ax.hlines(0.26, x_ini, x_fin, color="black", lw=1.3)
ax.text(x_ini, 0.20, "Errores estándar entre paréntesis", ha="left", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("tabla_arima.png", dpi=300, bbox_inches="tight")
plt.show()


# %% Revisamos los residuos, autocorrelacion, jung box...
#Varianza, distribución y fac de los residuos
resultado_arima= ARIMA(df["log_ipc"], order=(0,1,1), trend="n").fit()
residuos = resultado_arima.resid.iloc[1:]
x = np.linspace(residuos.min(), residuos.max(), 200)
sns.set_style("whitegrid")
fig, axes= plt.subplots(2,2, figsize=(10,10))
plot_acf(residuos, lags=36, alpha=0.05, ax= axes[0,0], linewidth=1, markersize=0, title="FAC de residuos")
plot_acf(residuos**2, lags=36, alpha=0.05, ax=axes[0,1], linewidth=1, markersize=0, title="FAC de residuos^2")
axes[1,0].plot(residuos, color="brown", linewidth=0.7)
axes[1,0].set_title("Residuos", fontsize=12)
axes[1,0].set_xlabel("Fecha")
axes[1,0].set_ylabel("indice porcentual")
sns.histplot(residuos, bins=40, stat="density", kde=True, color="steelblue", ax=axes[1,1])
axes[1, 1].plot(x, norm.pdf(x, residuos.mean(), residuos.std()), color="red", linestyle="--", label="Normal")
plt.show()

#pruebas formales
plt.rcParams["mathtext.fontset"] = "cm"

# %% Pruebas de diagnóstico de residuales
ordenes = [(1, 1, 0), (1, 1, 1), (0, 1, 1)]
nombres = [f"ARIMA({p},{d},{q})" for p, d, q in ordenes]

diag = []
for (p, d, q) in ordenes:
    res = ARIMA(df["log_ipc"], order=(p, d, q), trend="n").fit()
    residuos = res.resid.iloc[d:]          # se descartan los primeros d residuos (inicialización)

    jb_stats, jb_p, asimetria, curtosis = jarque_bera(residuos)
    arch_p = {k: het_arch(residuos, nlags=k)[1] for k in [1, 2, 5]}
    lb = acorr_ljungbox(residuos, lags=[5, 10, 20], model_df=p + q)
    lb_p = lb["lb_pvalue"]

    diag.append([jb_p, arch_p[1], arch_p[2], arch_p[5], lb_p.loc[5], lb_p.loc[10], lb_p.loc[20]])

    print(f"ARIMA({p},{d},{q})")
    print("  JB p-valor:", round(jb_p, 3))
    print("  ARCH p-valores:", {k: round(float(v), 3) for k, v in arch_p.items()})
    print("  LB p-valores:", {k: round(float(v), 3) for k, v in lb_p.items()})

# %% Tabla gráfica
fig, ax = plt.subplots(figsize=(11, 4))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0.1, 1)

x_ini, x_fin = 0.01, 0.99
x_lab = 0.02
x_cols = [0.27, 0.40, 0.50, 0.60, 0.73, 0.83, 0.93]
sub = ["JB",
       r"$A_{(1)}$", r"$A_{(2)}$", r"$A_{(5)}$",
       r"$LB_{(5)}$", r"$LB_{(10)}$", r"$LB_{(20)}$"]

# Regla superior y encabezados agrupados
ax.hlines(0.95, x_ini, x_fin, color="black", lw=1.3)
ax.text(x_cols[0], 0.89, "Normalidad", ha="center", va="center", fontsize=13)
ax.text(x_cols[2], 0.89, "Efectos ARCH", ha="center", va="center", fontsize=13)
ax.text(x_cols[5], 0.89, "Correlación Serial", ha="center", va="center", fontsize=13)
ax.hlines(0.84, x_cols[1] - 0.05, x_cols[3] + 0.05, color="black", lw=0.6)
ax.hlines(0.84, x_cols[4] - 0.05, x_cols[6] + 0.05, color="black", lw=0.6)

# Subencabezados
ax.text(x_lab, 0.78, "Modelo", ha="left", va="center", fontsize=13)
for x, s in zip(x_cols, sub):
    ax.text(x, 0.78, s, ha="center", va="center", fontsize=13)
ax.hlines(0.72, x_ini, x_fin, color="black", lw=0.8)

# Filas de resultados
ys = [0.64, 0.55, 0.46]
for y, nom, fila in zip(ys, nombres, diag):
    ax.text(x_lab, y, nom, ha="left", va="center", fontsize=13)
    for x, v in zip(x_cols, fila):
        ax.text(x, y, f"{v:.3f}", ha="center", va="center", fontsize=13)

# Regla inferior y notas
ax.hlines(0.39, x_ini, x_fin, color="black", lw=1.3)
ax.text(x_ini, 0.33,
        r"Las cifras corresponden a valores-p. JB es la prueba Jarque-Bera, "
        r"$A_{(i)}$ es la prueba ARCH",
        ha="left", va="center", fontsize=9.5)
ax.text(x_ini, 0.27,
        r"de orden $i^{\mathbf{th}}$, $LB_{(i)}$ denota la prueba Ljung-Box de $i^{\mathbf{th}}$ orden.",
        ha="left", va="center", fontsize=9.5)

plt.tight_layout()
plt.savefig("tabla_diagnostico.png", dpi=300, bbox_inches="tight")
plt.show()
#pronostico
# %%
# %% Modelo con errores estándar (por defecto)
resultado_arima = ARIMA(df["log_ipc"], order=(0, 1, 1), trend="n").fit()

tabla_coef = pd.DataFrame({
    "Coef": resultado_arima.params,
    "Error estándar": resultado_arima.bse,
    "z": resultado_arima.tvalues,
    "p-valor": resultado_arima.pvalues,
    "IC 95% inf": resultado_arima.conf_int(alpha=0.05).iloc[:, 0],
    "IC 95% sup": resultado_arima.conf_int(alpha=0.05).iloc[:, 1],
}).round(4)
print(tabla_coef)

# %% Pronóstico a 10 periodos
pron = resultado_arima.get_forecast(steps=10)
media_log = pron.predicted_mean
intervalo_log = pron.conf_int(alpha=0.05)

# Volver de logaritmos a la escala del índice
tabla_pron = pd.DataFrame({
    "Pronóstico": np.exp(media_log),
    "Límite inferior (95%)": np.exp(intervalo_log.iloc[:, 0]),
    "Límite superior (95%)": np.exp(intervalo_log.iloc[:, 1]),
}).round(2)
print(tabla_pron)

# %% Gráfico del pronóstico
historia = df["ITCR_ipc"].loc["2015":]       # últimos años, para que se aprecie

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(historia, color="brown", linewidth=1.5, label="Observado")
ax.plot(tabla_pron["Pronóstico"], color="steelblue", linewidth=2, label="Pronóstico")
ax.fill_between(tabla_pron.index,
                tabla_pron["Límite inferior (95%)"],
                tabla_pron["Límite superior (95%)"],
                color="steelblue", alpha=0.25, label="Intervalo de predicción 95%")
ax.set_title("Pronóstico del ITCR (IPC) a 10 meses - ARIMA(0,1,1)", fontsize=14)
ax.set_xlabel("Fecha")
ax.set_ylabel("Índice")
ax.legend()
plt.tight_layout()
plt.show()
# %%


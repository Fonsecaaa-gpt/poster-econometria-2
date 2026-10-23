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

#Revisamos los datos con jupyter(?)
#print(df.columns) -> Nombres dde las columnas

# %% Preparamos los datos
df.columns = ["Fecha", "ITCR_ipp", "ITCR_ipc"]
df["Fecha"] = pd.to_datetime(df["Fecha"])
df = df.set_index("Fecha")

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
plot_acf(series,lags=36, alpha=0.05, ax=axes[0],linewidth=1,markersize=0, title="FAC de IPC")
plot_pacf(series,lags=36, alpha=0.05, method="ywm", ax=axes[1],linewidth=1,markersize=0, title="FACP de IPC")
plt.show()

# %% Ralizamos pruebas aic y bic para determinar el modelo arima, y su tabla
for orden in [(1,1,0), (1,1,1), (0,1,1)]:
    resultado_modelo_arima=ARIMA(df["log_ipc"], order=orden, trend="n").fit()
    print(orden, "AIC = ",round(resultado_modelo_arima.aic,2), "BIC = ", round(resultado_modelo_arima.bic,2))


#Aqui va la tabla


# %% para ARIMA(0,1,1)
resultado_arima= ARIMA(df["log_ipc"], order=(0,1,1), trend="n").fit()
print("coeficientes estimados = ", resultado_arima.params, "Errores estandar = ",resultado_arima.bse, "p-valor = ",resultado_arima.pvalues)
#tabla


# %% Revisamos los residuos, autocorrelacion, jung box...
#Varianza, distribución y fac de los residuos
residuos = resultado_arima.resid.iloc[1:]
x = np.linspace(residuos.min(), residuos.max(), 200)
sns.set_style("whitegrid")
fig, axes= plt.subplots(2,2, figsize=(10,10))
plot_acf(residuos, lags=36, alpha=0.05, ax= axes[0,0], linewidth=1, markersize=0, title="FAC de residuos")
plot_acf(residuos**2, lags=36, alpha=0.05, ax=axes[0,1], linewidth=1, markersize=0, title="FAC de residuos^2")
axes[1,0].plot(residuos, color="brown", linewidth=1.5)
axes[1,0].set_title("Residuos", fontsize=12)
axes[1,0].set_xlabel("Fecha")
axes[1,0].set_ylabel("indice porcentual")
sns.histplot(residuos, bins=40, stat="density", kde=True, color="steelblue", ax=axes[1,1])
axes[1, 1].plot(x, norm.pdf(x, residuos.mean(), residuos.std()), color="red", linestyle="--", label="Normal")

#pruebas formales
jb_stats, jb_p, asimetria, curtosis = jarque_bera(residuos)
arch_p = {k: het_arch(residuos, nlags=k)[1] for k in [1, 2, 5]}
lb = acorr_ljungbox(residuos, lags=[5,10,20], model_df=1)
lb_p = lb ["lb_pvalue"]
print("JB p-valor:", round(jb_p, 3))
print("ARCH p-valores:", {k: round(float(v), 3) for k, v in arch_p.items()})
print("LB p-valores:", {k: round(float(v), 3) for k, v in lb_p.items()})


# %%

# %% Importar librerias
from pathlib import Path 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA 

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
df["log_ipc"]= np.log(df["ITCR_ipc"])
print("ADF estadistico:", Resultado[0])
print("p-valor:", Resultado[1])
# ajustamos varianza y media
df["log_ipc"]=np.log(df["ITCR_ipc" ])
df["ipc_porcentual"]=df["log_ipc"].diff().dropna()

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
   
   
   
    #para ARIMA(0,1,1)
Resultado_arima= ARIMA(df["log_ipc"], order=(0,1,1), trend="n").fit()
print("coeficientes estimados = ", Resultado_arima.params, "Errores estandar = ",Resultado_arima.bse, "p-valor = ",Resultado_arima.pvalues)
    #tabla
    




# %% Revisamos los residuos, autocorrelacion, fac y facp, jung box...
#FAC Y FACP de los residuos
residuos=Resultado_arima.resid[1:]
fig, axes= plt.subplots(2,1, figsize=(10,8))
plot_acf(residuos, lags=36, alpha=0.05, ax= axes[0], linewidth=1, markersize=0, title="FAC de residuos")
plot_pacf(residuos, lags=36, alpha=0.05, ax=axes[1], linewidth=1, markersize=0, method="ywm", title="FACP de residuos")
#fac de residuos ^2
# %%
plot_acf(residuos**2, lags=36, alpha=0.05, linewidth=1, markersize=0, title="FAC de residuos^2")

# %%

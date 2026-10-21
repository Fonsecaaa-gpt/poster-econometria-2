# %% Importar librerias
from pathlib import Path 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller 


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
plt.title("Serie de tiempo del ITCR (IPP - IPC)", fontsize=16)
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
df["ipc_dif"]=df["log_ipc"].diff().dropna()

# %% dicky fuller despues de la transformacion
Resultado2 = adfuller(df["ipc_dif"].dropna(), autolag="AIC")
print("ADF estadistico (transformado):", Resultado2[0])
print("p-valor (transformado):", Resultado2[1]) 

# %%
# %% Grafica con var y media estabilizada
sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
plt.plot(df["ipc_dif"], color="brown", linewidth=1.5)
plt.title("ITCR (IPC) Var y media estabilizada", fontsize=16)
plt.xlabel("Fecha")
plt.ylabel("indice")
plt.show()

#si si esta bien, creeria que toca sacar fac y facp, y quedaria un arima (p,1,q)
# %%

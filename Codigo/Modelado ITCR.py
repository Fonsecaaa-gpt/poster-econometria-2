# %% Importar librerias
from pathlib import Path 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
plt.plot(df["ITCR_ipp"], color="darkblue", linewidth=1.5)
plt.plot(df["ITCR_ipc"], color="brown", linewidth=1.5)
plt.title("Serie de tiempo del ITCR (IPP - IPC)", fontsize=16)
plt.xlabel("Fecha")
plt.ylabel("indice")
plt.show()


# %%

# %% Importar librerias
from pathlib import Path 
import pandas as pd
import matplotlib.pyplot as plt

# %% Direccion base de datos
Base_dir = Path(__file__).resolve().parents[1]
Datos_dir = Base_dir / "Datos"
Ruta_exp = Datos_dir / "ITCR colombia.csv"

# %% Cargar datos
df = pd.read_csv(Ruta_exp)

#Revisamos los datos con jupyter
#print(df.columns) -> Nombres dde las columnas

# %% Preparamos los datos
df.columns = ["Fecha", "ITCR_ipp", "ITCR_ipc"]
# %%

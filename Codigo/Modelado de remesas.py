# %%
import pandas as pd
from pathlib import Path 

Base_dir = Path(__file__).resolve().parents[1]
Datos_dir = Base_dir / "Datos"
Ruta_exp = Datos_dir / "Tasas de interés de política monetaria.csv"

# %% Cargar datos
df = pd.read_csv(Ruta_exp)
# %%

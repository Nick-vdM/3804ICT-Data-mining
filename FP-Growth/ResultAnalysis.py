import matplotlib.pyplot as plt
import pandas as pd

df_library = pd.read_csv("Library.csv")
df_implementation = pd.read_csv("Implementation.csv")

ax = df_library.plot()
df_implementation.plot(ax=ax)
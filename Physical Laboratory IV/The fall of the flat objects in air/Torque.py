import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
import plotly.express as px

path = r"/workspaces/Physical-Labratory/Physical Laboratory IV/The fall of the flat objects in air/"

df = pd.read_excel(path + "moment síly.xlsx", sheet_name='Torque')

l = 4.1e-2  # m
g = 9.81  # m/s^2
df['m'] = df['m'] * 1e-3  # kg

df['M'] = l*g*cos(deg2rad(df['alpha']+10))*df['m']  # N*m


fig, ax = plt.subplots(figsize=(16, 9))
ax.scatter(df['alpha'], df['M']*1e3, color='blue', label='Torque vs Angle', s=50)
ax.set_xlabel(fr'Angle $\alpha$ [$^\circ$]', fontsize=14)
ax.set_ylabel(fr'Torque $M$ [mN*m]', fontsize=14)
ax.set_title('Torque vs Angle', fontsize=20, pad=10, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=13)
ax.legend(fontsize=14)
ax.grid(alpha=0.6)

fig.savefig(path + "Torque.png", dpi=300, bbox_inches='tight')
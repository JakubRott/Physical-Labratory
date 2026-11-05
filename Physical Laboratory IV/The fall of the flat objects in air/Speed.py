import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import pandas as pd

path = r"/workspaces/Physical-Labratory/Physical Laboratory IV/The fall of the flat objects in air/"

top = pd.read_excel(path + 'speed.xlsx', sheet_name='Navetr')
bot = pd.read_excel(path + 'speed.xlsx', sheet_name='Zavetr')

# data: top side (gray line) and underside, same x convention [cm], speeds in m/s
x_top = #np.array([8.35-10,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8])
v_top = #np.array([4.8,5.2,np.mean([5.2,5.9]),5.6,5.5,5.8,5.96,5.96,5.92,5.89,5.84,5.66,5.59,5.5,5.37,5.29,5.28])
x_bot = #-1*np.array([-1,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8])  # cm
v_bot = #np.array([5.13,4.81,4.71,4.47,4.42,4.2,4.11,4,3.92,3.83,3.8,3.7,3.71,3.67,3.7]) # m/s
# if the underside x values are negative only because of the sign convention
# and you want them on the same x range, use np.abs(...) here

h, angle, x0 = 1.7, 30, 0              # cm, deg, cm
vmax = max(v_top.max(), v_bot.max())
s = 5 / vmax                           # cm of plot height per m/s (visual scale only)

# plane (your two points, minus sign removed from the second y)
line = np.array([
    [5*np.cos(np.deg2rad(180-angle)), 5*np.cos(np.deg2rad(360-angle))],
    [5*np.sin(np.deg2rad(180-angle)), 5*np.sin(np.deg2rad(360-angle))]
])

xs = np.concatenate([x_top, x_bot, x0 + line[0]])
xmin, xmax = xs.min() - 1, xs.max() + 1
ytop = h + s*vmax + 1
ybot = -ytop

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(xmin, xmax); ax.set_ylim(ybot, ytop)
ax.set_aspect('equal')                 # 1 cm in x = 1 cm in y, so 31° looks like 31°

# measurement lines
ax.axhline( h, color='gray', lw=1, ls='--')
ax.axhline(-h, color='gray', lw=1, ls='--')

# data: top goes up from +h, underside goes down from -h
ax.scatter(x_top,  h + s*v_top, color='tab:red',  zorder=3, label='Top side')
ax.scatter(x_bot, -h - s*v_bot, color='tab:blue', zorder=3, label='Underside (x measured oppositely)')

# geometry
ax.axhline(0, color='black', lw=1, linestyle='-.')                         # middle of the plane
ax.plot(x0 + line[0], line[1], color='black', lw=3)        # the plane
R = 2.5
ax.add_patch(Arc((x0, 0), 2*R, 2*R, theta1=-angle, theta2=0, color='black', lw=1.5))
a = np.deg2rad(-angle/2)
ax.text(x0 + 1.25*R*np.cos(a), 1.25*R*np.sin(a), f'{angle}°', va='center')

# speed tick labels: positive on both sides, zero on the respective measurement line
tv = np.linspace(0, vmax, 4)
ticks = np.concatenate([-h - s*tv[::-1], h + s*tv])
labels = [f'{t:.2g}' for t in tv[::-1]] + [f'{t:.2g}' for t in tv]
ax.set_yticks(ticks); ax.set_yticklabels(labels)

ax.set_xlabel('x [cm]')
ax.set_ylabel('v [m/s]  (0 on each gray line)')
ax.legend(loc='lower right', fontsize=10)

# 1.7 cm dimension annotations (between the black center line and each gray line)
xa = xmin + 1                      # x position of the arrows; shift if it collides with data
for y_end in (h, -h):
    ax.annotate('', xy=(xa, 0), xytext=(xa, y_end),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(xa + 0.3, y_end/2, f'{h} cm', va='center', ha='left')

# wind direction: from negative to positive x
y_wind = ytop - 0.8                          # near the top edge, above the data
ax.annotate('', xy=(xmin + 5, y_wind), xytext=(xmin + 1, y_wind),
            arrowprops=dict(arrowstyle='-|>', color='tab:green', lw=2.5, mutation_scale=20))
ax.text(xmin + 3, y_wind + 0.1, 'Air Flow', color='tab:green', ha='center', va='bottom')

y_wind2 = ybot + 0.8
ax.annotate('', xy=(xmin + 5, y_wind2), xytext=(xmin + 1, y_wind2),
            arrowprops=dict(arrowstyle='-|>', color='tab:green', lw=2.5, mutation_scale=20))
ax.text(xmin + 3, y_wind2 + 0.1, 'Air Flow', color='tab:green', ha='center', va='bottom')




fig.savefig(path +'speed_both_sides.png', dpi=300, bbox_inches='tight')
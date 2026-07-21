import glob
import numpy as np
import scipy.io as sio
from svgpathtools import svg2paths

svg_files = sorted(glob.glob("i/*.svg"))
num_points = 100
custom_demos = []

for file_path in svg_files:
    paths, _ = svg2paths(file_path)

    # Combine all path segments of the letter
    full_x, full_y = [], []
    for path in paths:
        pts = [path.point(t) for t in np.linspace(0, 1, 30)]
        full_x.extend([p.real for p in pts])
        full_y.extend([-p.imag for p in pts])

    # Interpolate to exactly 100 points
    t_orig = np.linspace(0, 1, len(full_x))
    t_new = np.linspace(0, 1, num_points)

    x_interp = np.interp(t_new, t_orig, full_x)
    y_interp = np.interp(t_new, t_orig, full_y)

    # Center the trajectory at the starting point (0,0)
    x_interp -= x_interp[0]
    y_interp -= y_interp[0]

    pos = np.vstack([x_interp, y_interp])
    vel = np.gradient(pos, axis=1)

    custom_demos.append(
        {"pos": pos, "vel": vel, "t": np.arange(1, num_points + 1)}
    )

# Save in MATLAB format
demos_array = np.empty((1, len(custom_demos)), dtype=object)
for i, d in enumerate(custom_demos):
    demos_array[0, i] = d

sio.savemat("i.mat", {"demos": demos_array})
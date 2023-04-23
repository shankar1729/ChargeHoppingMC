#!/usr/bin/env python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

def plotTrajectoryFromFile(fname, dir_names="xyz"):
    trajectory = np.loadtxt(fname)
    fname_out = os.path.splitext(fname)[0] + '.png'
    plotTrajectory(trajectory, fname_out, dir_names)
    plt.show()


def plotTrajectory(trajectory, fname, dir_names):
    if len(trajectory.shape) == 1:  # structured case
        trajectory = np.array(trajectory.tolist(), dtype=float)  # 2D array
    iElectron = trajectory[:, 0]
    nElectrons = 1 + int(iElectron.max())
    print(f"Read trajectory with {len(iElectron)} hops and {nElectrons} electrons")
    
    # Extract segments
    t = trajectory[:, 1]
    sort_index = np.lexsort((t, iElectron))
    trajectory = trajectory[sort_index]
    iElectron, t, x, y, z = trajectory.T
    maxHopDistance = 5.  # break segments at jumps longer than this
    i0 = np.where(np.logical_and(
        iElectron[:-1] == iElectron[1:],
        np.hypot(x[1:] - x[:-1], y[1:] - y[:-1]) < 5.
    ))[0]  # starting indices of segments
    
    # Sort segments by y for depth-ordered plotting:
    y_mid = 0.5*(y[i0] + y[i0 + 1])
    sort_index = y_mid.argsort()[::-1]  # highest y first (made deepest)
    i0 = i0[sort_index]
    y_mid = y_mid[sort_index]
    points = np.array((z, x)).T
    points += np.random.randn(*points.shape)  # minimize aliasing
    segments = np.concatenate((points[i0, None], points[i0 + 1, None]), axis=1)
    
    # Transparency color map:
    ref_color = colors.to_rgb("red")
    alpha_max = 0.3
    cmap = colors.LinearSegmentedColormap.from_list(
        'custom_cmap', [ref_color + (alpha_max,), ref_color + (0.0,)]
    )

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.add_collection(LineCollection(segments, array=y_mid, cmap=cmap, lw=0.5))
    plt.xlim(z.min(), z.max())
    plt.ylim(x.min(), x.max())
    plt.xlabel(f'{dir_names[2]} [nm]')
    plt.ylabel(f'{dir_names[0]} [nm]')
    plt.savefig(fname, bbox_inches='tight')


if __name__=="__main__":
	assert len(sys.argv) == 2
	plotTrajectoryFromFile(sys.argv[1])

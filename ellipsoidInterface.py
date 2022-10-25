import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
from ellipsoid import Ellipsoid
from common import printDuration


def main():
	calc = EllipsoidInterfaceCalculation("structure.mat")
	calc.visualize_geometry("structure.png")


class EllipsoidInterfaceCalculation:
	
	def __init__(self, filename):
		"""Initialize calculation parameters from a mat filename."""
		mat = loadmat('structure.mat')
		self.aspect_ratio = float(mat['aspect_ratio'])
		self.centers = np.array(mat['centers'])
		self.axes = np.array(mat['axes'])
		self.L = np.array(mat['L'], dtype=float).flatten()  # box size
		self.h = np.array(mat['h'], dtype=float).flatten()  # grid spacing
		self.a = float(mat['a'])  # semi-major axis length (polar radius)
		self.b = self.a / self.aspect_ratio
		self.interface_thickness = np.array(
			mat['interface_thickness'], dtype=float
		).flatten()
		self.n_layers = len(self.interface_thickness) + 2
		self.epsilon = np.array(mat['epsilon'], dtype=float)
		assert self.epsilon.shape[1] == self.n_layers
		
		# Enforce constraints on data:
		self.axes *= (
			1.0 / np.linalg.norm(self.axes, axis=1)[:, None]
		)  # make sure these are unit vectors
		self.S = np.round(self.L / self.h).astype(int)  # grid shape
		self.h = self.L / self.S  # updated to be commensurate with L
		self.h_avg = np.prod(self.h) ** (1./3)  # average spacing
		
		# Create geometry evaluator
		self.n_max = self.interface_thickness.sum() + 4 * self.h_avg
		self.ellipsoid = Ellipsoid(self.a, self.b, self.n_max)

	def get_normal_distance(self, r1d):
		"""
		Calculate closest normal distance to any particle for positions
		specified by the (ij-indexed) meshgrid of 1D arrays x, y and z in r1d.
		(Periodic boundary conditions applied in the minimum image convention.)
		Output will be between -min(a, b) and self.n_max.
		"""
		r_shape = tuple(len(x) for x in r1d)
		n = np.full(r_shape, self.n_max)
		focus_dist = max(self.a, self.b) - self.b
		bbox_radius = self.b + self.n_max
		print('Setting normal distance:', end=' ', flush=True)
		particle_interval = int(np.ceil(0.02 * len(self.centers)))
		for i_particle, (center, axis) in enumerate(
			zip(self.centers, self.axes)
		):
			r_sq = []
			z = []
			bbox_sel = []
			for i_dir, (xi, center_i, axis_i, Li) in enumerate(
				zip(r1d, center, axis, self.L)
			):
				# Wrap coordinates around center (minimum-image convention):
				xi = (xi - center_i) / Li  # wrt to center in fractional coords
				xi -= np.floor(0.5 + xi)  # wrap to [-0.5, 0.5)
				xi *= Li  # back to Cartesian coordinates
				# Limit to bounding box:
				bbox_i = abs(axis_i) * focus_dist + bbox_radius
				bbox_sel_i = np.where(abs(xi) <= bbox_i)[0]
				# Put along appropriate dimension for broadcasting:
				out_shape = [1, 1, 1]
				out_shape[i_dir] = -1
				xi = xi[bbox_sel_i].reshape(out_shape)
				r_sq.append(np.square(xi))
				z.append(xi * axis_i)  # term in dot-product below
				bbox_sel.append(bbox_sel_i.reshape(out_shape))
			z = abs(sum(z))  # broadcasted dot-product r1d @ axis
			rho = np.sqrt(sum(r_sq) - np.square(z))  # broadcasted sqrt(r^2-z^2)
			bbox_sel = tuple(bbox_sel)
			# Compute normal coordinate:
			n[bbox_sel] = self.ellipsoid.minimum_normal_coordinate(
				rho, z, n[bbox_sel]
			)
			# Report progress:
			if (i_particle + 1) % particle_interval == 0:
				progress_pct = (i_particle + 1) * 100.0 / len(self.centers)
				print(f'{progress_pct:.0f}%', end=' ', flush=True)
		print('done.\n')
		return n
	
	def map_property(self, n, prop):
		"""
		Map property `prop` specified for each material component
		(with length `n_layers`) onto each point with normal coordinate `n`.
		The output will be smoothed to be just resolvable at h_avg spacing.
		"""
		# Create interpolation function:
		dn = 0.1 * self.h_avg
		sigma = 0.5 * self.h_avg
		n_grid = np.arange(
			-min(self.a, self.b) - 4*sigma, self.n_max + 4*sigma + dn, dn
		)
		prop_grid = np.full(n_grid.shape, prop[0], dtype=float)  # filler
		n_cut = 0.
		for thickness, prop_layer in zip(self.interface_thickness, prop[1:-1]):
			prop_grid[n_grid >= n_cut] = prop_layer  # interface layers
			n_cut += thickness
		prop_grid[n_grid >= n_cut] = prop[-1]  # matrix
		prop_grid = gaussian_filter1d(prop_grid, sigma/dn)  # smooth ~ h_avg
		# Interpolate:
		return np.interp(n, n_grid, prop_grid)

	def visualize_geometry(self, filename):
		"""Output visualization of geometry to filename."""
		r1d = tuple(np.arange(Si)*hi for Si, hi in zip(self.S, self.h))
		mask = self.map_property(
			self.get_normal_distance(r1d), np.arange(self.n_layers)
		)
		n_panels = 4
		fig, axes = plt.subplots(
			n_panels, n_panels, sharex=True, sharey=True,
			figsize=(n_panels*2, n_panels*2),
		)
		plt.subplots_adjust(hspace=0.1, wspace=0.1)
		for i_ax, ax in enumerate(axes.flatten()):
			iz = (i_ax * self.S[2]) // (n_panels ** 2)
			z = iz * self.h[2]
			plt.sca(ax)
			plt.axis("off")
			plt.imshow(
				mask[:, :, iz].T, vmin=0, vmax=self.n_layers-1, origin="lower"
			)
			plt.text(
				0.5, 0.99, f"{z = :.1f}", transform=ax.transAxes,
				ha='center', va='bottom', fontsize="small",
			)
		plt.savefig(filename, bbox_inches='tight', dpi=150)


if __name__ == "__main__":
	main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
from ellipsoid import Ellipsoid


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

	def get_normal_distance(self, r):
		"""
		Calculate closest normal distance to any particle for positions r
		(periodic wrapping in final length-3 dimension of r handled).
		Output will be between -min(a, b) and self.n_max.
		"""
		n = np.full(r.shape[:-1], self.n_max)
		print('Setting normal distance:', end=' ', flush=True)
		particle_interval = int(np.ceil(0.02 * len(self.centers)))
		for i_particle, (center, axis) in enumerate(
			zip(self.centers, self.axes)
		):
			# Map r onto cylindrical coordinates about axis of particle:
			dr = (r - center) / self.L  # separation in fractional coordinates
			dr -= np.floor(0.5 + dr)  # wrap to [-0.5, 0.5)
			dr *= self.L  # separation with minimum-image convention wrapping
			z = abs(dr @ axis)  # axial coordinate in frame of particle
			rho = np.sqrt(np.linalg.norm(dr, axis=-1)**2 - z**2)  # cyl. coord.
			# Compute normal coordinate:
			sel, n_sel = self.ellipsoid.normal_coordinate(rho, z)
			n[sel] = np.minimum(n[sel], n_sel)
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
		grids1D = tuple(np.arange(Si)*hi for Si, hi in zip(self.S, self.h))
		r = np.array(
			np.meshgrid(*grids1D, indexing='ij')
		).transpose((1, 2, 3, 0))  # bring Cartesian direction to end
		mask = self.map_property(
			self.get_normal_distance(r), np.arange(self.n_layers)
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

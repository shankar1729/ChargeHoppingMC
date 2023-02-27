import numpy as np
from scipy.interpolate import RectBivariateSpline


class Ellipsoid:
	"""Efficient calculator of point-ellipsoid distance with caching."""
	
	def __init__(self, a, b, n_max):
		"""
		Setup to calculate normal distances of points up to
		n_max away from an ellipsoid with semi-axes a and b.
		"""
		self.a = a
		self.b = b
		self.n_max = n_max
		n_points_max = 200  # max. resolution of cache in any direction
		dr = (max(a, b) + n_max) / n_points_max
		x = np.linspace(0., a + n_max, 1 + int(np.ceil((a + n_max) / dr)))
		y = np.linspace(0., b + n_max, 1 + int(np.ceil((b + n_max) / dr)))
		x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
		n = ellipse_normal_coordinate(a, b, x_mesh, y_mesh, n_iter=20)
		self.n = RectBivariateSpline(x, y, n, kx=1, ky=1)  # linear interpolator
		
	def minimum_normal_coordinate(self, rho, z, n):
		"""
		Update n with minimum(n, normal distance) of points (rho, z) in
		cylindrical coordinates, where z is along the ellipsoid a-axis.
		Note that points with n >= n_max are not modified.
		(The dimensions of rho, z and n should match.)
		Returns the input n (which is modified in place).
		"""
		# Select bounding box:
		sel = np.where(np.logical_and(
			z <= self.a + self.n_max, rho <= self.b + self.n_max,
		))
		# Update n:
		n[sel] = np.minimum(n[sel], self.n.ev(z[sel], rho[sel]))
		return n


def ellipse_normal_coordinate(a, b, x, y, n_iter=10):
	"""
	Calculate normal coordinate of point (x, y) with respect to the ellipse
	centered at (0, 0) with semi-major and semi-minor lengths, a and b. The
	result is <0, =0 and >0 for points inside, on and outside the ellipse.
	Here, n_iter controls number of iterations of the numerical solution.
	"""
	theta = ellipse_normal_point(a, b, x, y, n_iter)
	c, s = np.cos(theta), np.sin(theta)
	return (abs(b*x*c) + abs(a*y*s) - abs(a*b)) / np.hypot(a*s, b*c)


def ellipse_normal_point(a, b, x, y, n_iter=10):
	"""
	Return theta such that the line connecting (a cos(theta), b sin(theta))
	and (x, y) is normal to the ellipse centered at the origin with semi-major
	and semi-minor lengths, a and b. This requires an iterative solution,
	with number of iterations limited by n_iter.
	"""
	# This amounts to solving for theta in:
	# A sin(theta) + B cos(theta) + C sin(theta) cos(theta) = 0,
	# where A = a x, B = -b y, C = b^2 - a^2
	A = abs(a * x)  # map all to first quadrant
	B = -abs(b * y)
	C = b**2 - a**2
	theta = np.full(x.shape, 0.25*np.pi)
	fp_sq = 1E-9 ** 2  # floor on (df/dtheta)^2 for safe step
	print('Normal solve (MAE):', end=' ', flush=True)
	for i_iter in range(n_iter):
		c, s = np.cos(theta), np.sin(theta)
		f = A*s + B*c + C*s*c
		f_prime = A*c - B*s + C*(c*c - s*s)  # df/dtheta
		theta -= f * f_prime/(f_prime**2 + fp_sq)   # Safe Newton-Raphson update
		theta = np.clip(theta, 0., 0.5*np.pi)  # keep in first quadrant
		print(f"{abs(f).mean():.0e}", end= ' ', flush=True)
	print('done.', flush=True)
	return theta


#---------- Test code ----------
if __name__ == "__main__":
	import matplotlib.pyplot as plt
	ellipsoid = Ellipsoid(a=8., b=1., n_max=1.)
	z, rho = np.meshgrid(np.linspace(0, 10, 1000), np.linspace(0, 3, 300))
	n = np.full(z.shape, ellipsoid.n_max)
	ellipsoid.minimum_normal_coordinate(rho, z, n)
	n_mag = 1.5
	plt.imshow(
		n, extent=(z.min(), z.max(), rho.min(), rho.max()), origin='lower',
		cmap='RdBu', vmin=-n_mag, vmax=+n_mag,
	)
	plt.xlabel("$z$")
	plt.ylabel(r"$\rho$")
	plt.colorbar()
	plt.show()

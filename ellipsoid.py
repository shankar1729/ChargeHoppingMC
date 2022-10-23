import numpy as np


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

	x, y = np.meshgrid(np.linspace(-9, 9, 9001), np.linspace(-2, 2, 2001))
	a, b = 8., 1.
	n = ellipse_normal_coordinate(a, b, x, y)
	n_mag = 1.
	plt.imshow(
		n, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower',
		cmap='RdBu', vmin=-n_mag, vmax=+n_mag,
	)
	plt.colorbar()
	plt.show()

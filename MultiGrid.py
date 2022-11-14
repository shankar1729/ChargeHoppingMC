import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags


class MultiGrid:
	"""Recursive implementation of a geometric multi-grid method."""
	
	def __init__(self, A, S, subtract_mean, Ndirect=1000, top_level=True):
		"""
		Construct multi-grid heirarchy for matrix A with grid dimensions S.
		(A must be a sparse square matrix with side prod(S).)
		If subtract_mean, project out the mean value of result.
		Resort to a direct solution once matrix size drops below Ndirect.
		"""
		N = np.prod(S)
		assert A.shape[0] == N
		if top_level:
			print("\tMultigrid:", end=" ", flush=True)
		
		if N <= Ndirect:
			self.invA = np.linalg.pinv(A.toarray())
			self.next_grid = None
			print(f"{S}.", flush=True)
		else:
			# Select coarse grid points:
			index = np.arange(N).reshape(S)
			index_coarse = index[(slice(None, None, 2),) * len(S)]
			Scoarse = index_coarse.shape
			print(f"{S} ->", end=" ", flush=True)
			
			# Construct interpolation = transpose(coarsening) operator:
			absA = csr_matrix(abs(A))
			absA.sum_duplicates()
			absA.eliminate_zeros()
			interp = absA[:, index_coarse.flatten()]
			Ncoarse = interp.shape[1]
			
			while(True):
				# Normalize non-zero rows to unit sum:
				row_sum = interp.sum(axis=1).view(np.ndarray).flatten()
				threshold = 1E-12 * row_sum.max()
				done = np.where(row_sum > threshold)[0]
				interp_done = diags(1.0 / row_sum[done]) @ interp[done]
				
				# Determine pending rows:
				pending = np.where(row_sum <= threshold)[0]
				if not len(pending):
					interp = interp_done
					break
				
				# Set pending rows by interpolating from ready ones
				interp_pending = absA[pending][:, done] @ interp_done
				
				# Combine pieces of interp:
				interp_done = coo_matrix(interp_done)
				interp_pending = coo_matrix(interp_pending)
				interp = csr_matrix((
					np.concatenate((interp_done.data, interp_pending.data)),
					(
						np.concatenate((
							done[interp_done.row], pending[interp_pending.row]
						)),
						np.concatenate((interp_done.col, interp_pending.col)),
					)
				), shape=(N, Ncoarse))
			
			# Finalize interpolation matrix and cleanup temporary matrices:
			del absA
			self.interp = csr_matrix(interp)
			del interp
			
			# Store pieces for relaxation operator (approximate A inverse):
			self.A = A
			self.Dinv = 1.0 / A.diagonal()
			self.subtract_mean = subtract_mean
			
			# Construct coarse-grid matrix:
			Acoarse = self.interp.T @ (A @ self.interp)
			self.next_grid = MultiGrid(Acoarse, Scoarse, False, Ndirect, False)

	def Vcycle(self, rhs):
		if self.next_grid is None:
			return self.invA @ rhs  # Exact inverse on coarsest grid
		
		Dinv_rhs = self.Dinv * rhs  # Pre-relaxtion applied to phi = 0
		r = rhs - self.A @ Dinv_rhs  # Compute residual
		phi = Dinv_rhs + self.interp @ self.next_grid.Vcycle(self.interp.T @ r)
		phi += Dinv_rhs - self.Dinv * (self.A @ phi)  # Post-relaxation
		if self.subtract_mean:
			phi -= phi.mean()
		return phi

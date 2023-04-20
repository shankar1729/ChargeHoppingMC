import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat, savemat
from ellipsoid import Ellipsoid
from Poisson import Poisson


def main():
    # Get command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dielectric",
        help="compute dielectric tensor",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--mobility",
        help="compute carrier mobility",
        action="store_true",
    )
    args = parser.parse_args()
    compute_both = not (args.mobility or args.dielectric)  #default is both

    # Load inputs:
    calc = EllipsoidInterfaceCalculation("structure.mat")
    calc.visualize_geometry("structure")

    # Calculate mobility if requested:
    if args.mobility or compute_both:
        run_mobility(calc)

    # Calculate epsilon if requested:
    if args.dielectric or compute_both:
        run_epsilon(calc)


def run_epsilon(calc):
    
    # Calculate and report dielectric tensor for each frequency:
    sort_index = calc.freq.argsort()[::-1]  # solve from high to low frequency
    epsilon_arr = np.zeros((len(sort_index), 7), dtype=calc.epsilon.dtype)
    for i_freq, freq, epsilon in zip(
        sort_index, calc.freq[sort_index], calc.epsilon[sort_index]
    ):
        print(f'\nCalculating for {freq = :g} material epsilons = {epsilon}:')
        epsilon_eff = calc.get_epsilon_eff(epsilon)
        epsilon_arr[i_freq] = np.array([
            np.trace(epsilon_eff)/3,  # Avg
            epsilon_eff[0, 0], # XX
            epsilon_eff[1, 1], # YY
            epsilon_eff[2, 2], # ZZ
            epsilon_eff[1, 2], # YZ
            epsilon_eff[2, 0], # ZX
            epsilon_eff[0, 1], # XY
        ])
        print(f'Effective epsilon:\n{epsilon_eff}')

    # Additionally save results in matlab format:
    savemat(
        "epsilon.mat",
        {
            "freq": calc.freq,  # relayed from input; not used in calculation
            "epsilon": epsilon_arr,
            "column_names": ["Avg", "XX", "YY", "ZZ", "YZ", "ZX", "XY"],
        }
    )


def run_mobility(calc):
    print("Running mobility")
    calc.get_mobility(2)


class EllipsoidInterfaceCalculation:
    
    def __init__(self, filename):
        """Initialize calculation parameters from a mat filename."""
        mat = loadmat('structure.mat')
        self.molecule = str(mat["molecule"][0]) if ("molecule" in mat) else None 
        self.aspect_ratio = float(mat['aspect_ratio'])
        self.centers = np.array(mat['centers'])
        self.axes = np.array(mat['axes'], dtype=float)
        self.L = np.array(mat['L'], dtype=float).flatten()  # box size
        self.h = np.array(mat['h'], dtype=float).flatten()  # grid spacing
        self.a = float(mat['a'])  # semi-major axis length (polar radius)
        self.b = self.a / self.aspect_ratio
        self.interface_thickness = np.array(
            mat['interface_thickness'], dtype=float
        ).flatten()
        self.n_layers = len(self.interface_thickness) + 2
        self.epsilon = np.array(mat['epsilon'])
        self.freq = np.array(mat.get('freq', range(len(self.epsilon)))).flatten()
        assert self.epsilon.shape[1] == self.n_layers
        self.phi0 = None  # Initial guesses for Poisson solve
        
        # Enforce constraints on data:
        self.axes *= (
            1.0 / np.linalg.norm(self.axes, axis=1)[:, None]
        )  # make sure these are unit vectors
        self.S = np.round(self.L / self.h).astype(int)  # grid shape
        self.h = self.L / self.S  # updated to be commensurate with L
        self.h_avg = np.prod(self.h) ** (1./3)  # average spacing
        
        # Compute normal distances for simulation mesh
        self.n_max = self.interface_thickness.sum() + 4 * self.h_avg
        self.ellipsoid = Ellipsoid(self.a, self.b, self.n_max)
        r1d = tuple(np.arange(Si)*hi for Si, hi in zip(self.S, self.h))
        self.n = self.get_normal_distance(r1d)

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
        print('done.')
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
        prop_grid = np.full(n_grid.shape, prop[0])  # filler
        n_cut = 0.
        for thickness, prop_layer in zip(self.interface_thickness, prop[1:-1]):
            prop_grid[n_grid >= n_cut] = prop_layer  # interface layers
            n_cut += thickness
        prop_grid[n_grid >= n_cut] = prop[-1]  # matrix
        prop_grid = gaussian_filter1d(prop_grid, sigma/dn)  # smooth ~ h_avg
        # Interpolate:
        return np.interp(n, n_grid, prop_grid)

    def visualize_geometry(self, filename_prefix):
        """Output visualization of geometry to filename."""
        mask = self.map_property(self.n, np.arange(self.n_layers, dtype=float))
        self.visualize_field(
            filename_prefix, mask, vmin=0, vmax=self.n_layers - 1
        )

    def visualize_field(self, filename_prefix, value, vmin=None, vmax=None):
        """Output visualization of scalar field `value` to filename."""
        n_panels = 4
        for proj_dir, proj_name in enumerate("xyz"):
            fig, axes = plt.subplots(
                n_panels, n_panels, sharex=True, sharey=True,
                figsize=(n_panels*2, n_panels*2),
            )
            plt.subplots_adjust(hspace=0.1, wspace=0.1)
            for i_ax, ax in enumerate(axes.flatten()):
                i_proj = (i_ax * self.S[proj_dir]) // (n_panels ** 2)
                proj = i_proj * self.h[proj_dir]
                index = [slice(None)] * 3
                index[proj_dir] = i_proj
                plt.sca(ax)
                plt.axis("off")
                plt.imshow(
                    value[tuple(index)].T, vmin=vmin, vmax=vmax, origin="lower"
                )
                plt.text(
                    0.5, 0.99, f"${proj_name}$ = {proj:.1f}",
                    ha='center', va='bottom', fontsize="small",
                    transform=ax.transAxes, 
                )
            plt.savefig(
                f'{filename_prefix}_{proj_name}.png',
                bbox_inches='tight',
                dpi=150,
            )

    def get_epsilon_eff(self, epsilon):
        """Return effective dielectric tensor for specificied
        set of dielectric constants of each layer."""
        epsInv = self.map_property(self.n, 1.0 / epsilon)
        epsEff, self.phi0 = Poisson(self.L, epsInv).computeEps(phi0=self.phi0)
        return epsEff

    def get_mobility(self, i_dir):
        """Compute carrier mobility by Monte Carlo simulations.
        Here, `i_dir` is 0-based index of non-periodic direction with field.
        """
        assert self.molecule is not None  # Need to know for trap distribution
        
        # Create trap distributions for each type at each spatial location:
        E0 = np.zeros((np.prod(self.S), 3))
        # --- filler
        trapDepthFiller = 1.1
        # --- matrix
        trapDepthSigma = 0.224  # from previous paper
        trapDepthMatrix = np.random.randn(len(E0)) * trapDepthSigma
        # --- functional group
        trapHistogram = {
            "thiophene": [1, 2, 2, 3, 5, 2],
            "terthiophene": [2, 5, 5, 4, 3, 3, 1, 3, 2, 1],
            "ferrocene": [8, 4, 5, 7, 2, 2, 3, 0, 1, 1],
        }  # from DFT calculations, trap counts within constant-width bins 
        trapBinWidth = 0.2  # in eV
        trapCDF = np.cumsum([0] + trapHistogram[self.molecule]).astype(float)
        trapCount = trapCDF[-1]  # in the DFT, this is for eff volume ~ 15 nm^3
        trapCDF *= 1.0 / trapCount  # normalize PDF to 1
        trapBins = trapBinWidth * np.arange(len(trapCDF))
        trapDepthMol = np.interp(np.random.rand(len(E0)), trapCDF, trapBins)
        # --- account for functional group only in part of extrinsic interface
        dftVolume = 15.0
        voxelVolume = np.prod(self.h)
        trapProb = trapCount / (trapCount + dftVolume / voxelVolume)
        sel = np.where(np.random.rand(len(E0)) > trapProb)[0]
        trapDepthMol[sel] = trapDepthMatrix[sel]
        
        # Create energy landscape using mask:
        mask = self.map_property(
            self.n, np.arange(self.n_layers, dtype=float)
        ).flatten()
        E0 = -trapDepthMatrix
        sel_mol = np.where(mask < 1.5)[0]  # select upto extrinsic interface
        E0[sel_mol] = -trapDepthMol[sel_mol]
        sel_filler = np.where(mask < 0.5)[0]  # select filler alone
        E0[sel_filler] = -trapDepthFiller
        E0 = E0.reshape(self.S)
        if i_dir == 2:  # Visualize trap landscape for one of the directions
            self.visualize_field("energy", E0, vmin=E0.min(), vmax=E0.max())
        exit()

        #--- calculate polymer internal DOS
        Epoly = params["dosMu"] + params["dosSigma"]*np.random.randn(*S)
        #--- calculate electric field contributions and mask:
        Ez = params["Efield"]
        mask = params["mask"]
        print('Solving Poisson equation:')
        phi = PeriodicFD(L, mask, epsNP, epsBG, [False,False,True]).solve([0,0,Ez], shouldPlotNP)
        #--- combine field and DOS contributions to energy landscape:
        return phi + np.where(mask>0.5, params["trapDepthNP"], Epoly)

if __name__ == "__main__":
    main()

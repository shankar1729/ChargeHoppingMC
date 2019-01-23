# Charge Hopping Simulation using Monte Carlo method

- A graph of connected local minima in the energy landscape is constructed to avoid unnecessary hopping events in the local minima.
- Coulomb interaction is considered between all electrons, and it is updated at every hop. The update routine only alters the pairs involving the moving electron.
- Memory optimization is also done.
- Multiple MC simulations are carried out parallely to improve the statistics
- Simulation stops when one of the electrons reaches the other end
- An upper cap on the number of hops for each electron to stop the simulation without waiting for any of them to reach the other end

Two approaches have been implemented, namely
1. Grid-based method (`CarrierHoppingMC.py`)
2. Graph-based method (`MinimaHoppingMC.py`)

## How to run

The two Python files mentioned above have a list of parameters (with descriptive comments) near the end of file. Change them before running the simulation using either `python CarrierHoppingMC.py` or `python MinimaHoppingMC.py`

## Output

An ASCII file named `trajectory.dat` contains information about each electron hop.
It has five columns corresponding to:
- electron ID (electrons over parallel runs have unique IDs)
- time of hop (in seconds)
- 3 columns for coordinates of electron after the hop (in nm)

## Post-processing

`analyze.py` post-process the output file to:
- plot trajectories
- calculate and plot average velocity and displacement in z-dir
- exports figures in PDF files and calculated values in .dat files

# Charge Hopping Simulation using Monte Carlo method

- A graph of connected local minima in the energy landscape is constructed to avoid unnecessary hopping events in the local minima.
- Coulomb interaction is considered between all electrons, and it is updated at every hop. The update routine only alters the pairs involving the moving electron.
- Memory optimization is also done.
- Multiple MC simulations are carried out parallely to improve the statistics
- Simulation stops when one of the electrons reaches the other end
- An upper cap on the number of hops for each electron to stop the simulation without waiting for any of them to reach the other end

## TODO

- [ ] Nano particle - various cluster shapes
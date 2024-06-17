The trinomial.py file contains 2 classes, that compute the trinomial tree for the Hull-White and Black-Karasinski.
The trinomial class is the parent class, that through inheritance develops 2 other classes.

The Hull-White is computed using the analytical solution for the alpha parameter:
- $\alpha_m = \frac{\ln \left(\sum_{j=-n_m}^{n_m} Q_{m,j} e^{-j \Delta R \Delta t} \right) - \ln P_{m+1}}{\Delta t}$



- Yield Interpolation function
  

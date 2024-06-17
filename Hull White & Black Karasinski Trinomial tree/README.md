The trinomial.py file contains 2 classes, that compute the trinomial tree for the Hull-White and Black-Karasinski.
The trinomial class is the parent class, that through inheritance develops 2 other classes.

The Hull-White is computed using the analytical solution for the alpha parameter:
- alpha_m = (log(sum(Q_m,j * exp(-j * Delta_R * Delta_t) for j in range(-n_m, n_m + 1))) - log(P_m_plus_1)) / Delta_t


- Yield Interpolation function
  

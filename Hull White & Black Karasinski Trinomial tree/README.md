The trinomial.py file contains 2 classes, that compute the trinomial tree for the Hull-White and Black-Karasinski.
The trinomial class is the parent class, that through inheritance develops 2 other classes.

The Hull-White is computed using the analytical solution for the alpha parameter:
- The formula for αₘ is:

αₘ = (ln (∑ⱼ₌₋ₙₘ ᵠₘⱼ ⋅ e⁻ʲ ⋅ ΔR ⋅ Δt) - ln Pₘ₊₁) / Δt



- Yield Interpolation function
  

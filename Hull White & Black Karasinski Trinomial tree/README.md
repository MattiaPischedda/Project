The trinomial.py file contains 2 classes, that compute the trinomial tree for the Hull-White and Black-Karasinski.
The trinomial class is the parent class, that through inheritance develops 2 other classes.
The trinomial discretizes the continuous stochastic process:
- Hull White: $dr = a(b -r)dt + \sigma \, dZ$
- Black Karasinski: $d \ln(r) = [\theta(t) - a \ln(r)] \, dt + \sigma \, dZ$


The mean reverting process is obtained through a change in the branch of the tree.
![branching](https://github.com/MattiaPischedda/Project/assets/154690956/2cd91864-3bc0-4823-b5c1-9a7b19107c3d)


The Hull-White is computed using the analytical solution for the alpha parameter:
- $\alpha_m = \frac{\ln \left( \sum_{j=-n_m}^{n_m} Q_{m,j} e^{-j \Delta R \Delta t} \right) - \ln P_{m+1}}{\Delta t}$

  
while for the Black Karasinski, I had to resort to numerical procedure. The default methodology is Newton-Raphson, but specifying the optimization_type, also the Brent optimization can be used. The method is also able to handle a possible mistake, raising a value error.

Each class can compute different securities:
- traditional fixed coupon bonds.
- Callable and Putable bonds, with the possibility of inserting a Conversion Period.
- Interest Rate Swap
- Bermudian/American/European Swaption.

Additional methods are inserted to display the trees of the rate and the securities more pleasantly.
![finaltree](https://github.com/MattiaPischedda/Project/assets/154690956/acf3f894-5010-4178-9e2d-a7414402a433)
![swap](https://github.com/MattiaPischedda/Project/assets/154690956/c53d8d96-dff3-4ae7-94e3-4dc54d6b7ad2)




The trinomial tree is designed to match specifically the time step of the term structure provided. For this reason, a helper function named Yield Interpolation function is created, in order to interpolate using a Cubic Spline algorithm. The function is modelled to the term structure data provided by the ECB website. 
![months interpolation](https://github.com/MattiaPischedda/Project/assets/154690956/cf42b89d-41fa-409c-a257-3473262f07b2)
![1year interpolation](https://github.com/MattiaPischedda/Project/assets/154690956/7e34e321-d82e-49c5-92d0-188458d6cdb5)




  

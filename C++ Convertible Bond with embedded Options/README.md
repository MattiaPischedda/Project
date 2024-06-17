The convertible_bond.cpp file is an implementation of a Callable bond, a Putable bond, and a Convertible bond with/without an embedded option.

The rate tree is computed according to some statistical relationship with the stock. In the code, we have r = 0.16 - 0.001*S, but it can be modified accordingly.

To the bond can be added a call/put feature that allows the issuer/holder to redeem/sell the bond prior to maturity. It is also possible to insert an option protection period.

The resulting matrix is of the type:

![Screenshot 2024-06-17 214823](https://github.com/MattiaPischedda/Project/assets/154690956/f5adfddc-dd92-4562-b5fa-e04b320b1435)


Fabozzi, F. J. et al. (2002). Interest rate, term structure, and valuation modelling, volume 5. John Wiley & Sons

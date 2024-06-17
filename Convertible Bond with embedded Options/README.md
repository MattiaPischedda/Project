The convertible_bond.cpp file is an implementation of a Callable bond, a Putable bond, and a Convertible bond with/without an embedded option.

The rate tree is computed according to some statistical relationship with the stock. In the code, we have r = 0.16 - 0.001*S, but it can be modified accordingly.

The resulting matrix is of the type:

![image](https://github.com/MattiaPischedda/Project/assets/154690956/87d7b731-6006-4b06-8cbb-bd8966c3c9ae)



- Fabozzi, F. J. et al. (2002). Interest rate, term structure, and valuation modelling, volume 5. John Wiley &
Sons

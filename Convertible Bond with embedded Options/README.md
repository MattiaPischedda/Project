The convertible_bond.cpp file is an implementation of a Callable bond, a Putable bond, and a Convertible bond with/without an embedded option.

The rate tree is computed according to some statistical relationship with the stock. In the code, we have r = 0.16 - 0.001*S, but it can be modified accordingly.

The resulting matrix is of the type:
The bond price is: 1123.6
      1123.6        1100     1112.22      1222.9     1344.59      1478.4     1625.52     1787.28     1965.14      2160.7     2375.72     2712.14
           0        1100        1100        1100     1112.22      1222.9     1344.59      1478.4     1625.52     1787.28     1965.14      2260.7
           0           0     1086.61        1100        1100        1100     1112.22      1222.9     1344.59      1478.4     1625.52     1887.28
           0           0           0     1055.56     1082.65        1100        1100        1100     1112.22      1222.9     1344.59      1578.4
           0           0           0           0     1020.13     1046.96     1070.08     1087.67        1098        1100     1112.22      1322.9
           0           0           0           0           0     991.309     1013.95     1032.06     1043.37     1045.33     1035.37     1111.55
           0           0           0           0           0           0      972.38     992.651     1008.14     1016.49     1014.85        1100
           0           0           0           0           0           0           0     962.007     981.149      995.48     1002.69        1100
           0           0           0           0           0           0           0           0     959.593     978.609     992.842        1100
           0           0           0           0           0           0           0           0           0     964.986     984.843        1100
           0           0           0           0           0           0           0           0           0           0     978.323        1100
           0           0           0           0           0           0           0           0           0           0           0        1100


- Fabozzi, F. J. et al. (2002). Interest rate, term structure, and valuation modelling, volume 5. John Wiley &
Sons

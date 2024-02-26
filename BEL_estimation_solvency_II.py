"""
 This code is a project I did for the final exam of Insuranc-Solvency II of my MsC in Quantitative Finance.
 The aim is to compute the Best Estimate Liabilities, using the Best of Policies and Cliquet methods.
 
Note:

- The short rate model has been modeled using 1-factor Hull_White

- In order to compute the stochastic discount factor, I have discretized the value of r(t), using the cumulative sum function np.cumsum, 
     - Then discount the value to find the SDF 
     
- The revaluation method are: Best of Policies and Cliquet
 
- I supposed that each simulation had the same weight in the Expectation
    
"""

import random
import numpy as np
import math
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark') #Note -> if the style name is different and does not run, just remove this line



class BEL:
    
    
    value_alpha = [-0.005813556   ,-0.005839336   ,-0.005561832   ,-0.004823830   ,-0.003866221   ,-0.002706620   ,-0.001322801    ,0.000157949    ,0.001691744   ,0.003288021   ,0.004389697    ,0.006062678   ,0.007268967   ,0.007869954   ,0.009404967   ,0.009311137   ,0.008198868   ,0.008507299   ,0.010463014   ,0.014136915   ,0.018926121,   0.024014002   ,0.028557765   ,0.032367439   ,0.035921490   ,0.039014895   ,0.041622654   ,0.043929496   ,0.046108635   ,0.048145449   ,0.038169211 ]
    yield_curve = [ 1.000000, 1.005814, 1.011844, 1.017450, 1.022511, 1.026823, 1.030284, 1.032656, 1.034078, 1.034392, 1.033711, 1.033032, 1.030245, 1.028509, 1.025975, 1.022619, 1.021850, 1.022361, 1.022585, 1.020743, 1.015119, 1.005264, 0.991240, 0.974359, 0.955008, 0.933648, 0.911036, 0.887501, 0.863385, 0.838797, 0.814083 ]
    
    def __init__(self, a = 0.03, sigma = 0.01, dt = 1):
        self.a = a          #mean reversion parameter
        self.sigma = sigma  #volatility 
        self.dt = dt        #interval of time
        self.alpha = None
        
        #size of the array
        self.row = 31
        self.columns = 1000
        
        
    
    def rate_simulation(self):
    
        
        #create the arrays
        self.alpha = np.array(BEL.value_alpha)
        self.uniform_number = np.zeros((self.row+1, self.columns)) #31+1, so the first row is 0
        self.normal_number = np.zeros((self.row+1, self.columns)) #31+1, so the first row is 0
        self.x_values = np.zeros((self.row+1, self.columns)) #31+1, so the first row is 0
        self.r_values = np.zeros((self.row, self.columns))
        
    
        #make the simulation
        for i in range(self.row):
            for j in range(self.columns): 
                self.uniform_number[i+1][j] = random.random()  #random number between 0 and 1
                self.normal_number[i+1][j] = norm.ppf(self.uniform_number[i+1][j]) #normal ppf obtained using the uniform_number
                
                #The last row of the matrix x_values is computed but not used for r_values
                self.x_values[i+1][j] = self.x_values[i][j] * math.exp(-self.a * self.dt) + self.sigma * math.sqrt((1 / (2 * self.a)) * (1 - math.exp(-2 * self.a * self.dt))) * self.normal_number[i+1][j]
                self.r_values[i][j] = self.alpha[i] + self.x_values[i][j]  #x_values row 32 is created but not used in r_values because the last is x_values[30](row 31)
         
        
    
        return self.r_values
    

        
    def plot_rate_simulation(self):
        
        #calling the function
        result = self.rate_simulation()
        
        #plot the Monte Carlo Simulation 
        plt.figure(figsize=(25,15))
        plt.title(r'$r_{t}$ Monte Carlo Simulation', fontsize=30)
        plt.ylabel('Rate',fontsize= 20)
        plt.xlabel('Year',fontsize = 20)
        plt.grid(True)
        plt.plot(result)
        plt.show()
        sns.distplot(self.r_values,color='firebrick')
        plt.title(r"$r_{t}$ Distribution",fontsize=15)
        plt.grid(True)
        plt.axvline(x=self.r_values.mean(), linestyle='--', color='blue',linewidth=0.7,alpha=.7)
        
        # Calculate statistics
        mean_val = np.round(np.mean(self.r_values), 4)
        std_val = np.round(np.std(self.r_values), 4)
    

        # Create a box with statistics
        box_text = f"Mean: {mean_val}\nStd: {std_val}"

        # Add the box to the upper right
        plt.text(0.95, 0.95, box_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.show()
     
     
        
    #Stochastic discount factor
    def sdf_simulation(self):
        
        #Generate the stochastic discount factor
        self.rate_simulation()
        
        # Compute the cumulative short rate
        self.cum_short_rate = np.zeros((self.r_values.shape))
        for i in range(self.r_values.shape[1]):
            self.cum_short_rate[:, i] = np.cumsum(self.r_values[:, i], axis=0)    #Discretize the values using cumsum
            
        
        # Compute the SDF paths
        self.sdf_values = np.zeros((self.r_values.shape))
        for i in range(self.r_values.shape[0]):
            for j in range(self.r_values.shape[1]):
                self.sdf_values[i][j] = np.exp(-(self.cum_short_rate[i][j])) #discount the value of the integral
                

        return self.sdf_values
     
    
    def ESG(self):
        
        self.rate_simulation()
        self.yield_curve = np.array(BEL.yield_curve)
        
        #array for computation Px(t,T)
        self.H = (1/self.a) * (1- (np.exp(-self.a * (self.row-1) )))
        self.G = np.zeros((self.row, self.columns))
        self.P = np.zeros((self.row, self.columns))
        
        #array for computation of Px(0,t)
        self.H0 = np.zeros(self.row)
        self.G0 = np.zeros((self.row,self.columns))
        self.P0 = np.zeros((self.row, self.columns))
        
        
        self.ZCB = np.zeros((self.row, self.columns))      #ZCB, no pull to par effect
        self.Ra = np.zeros((self.row, self.columns))     #return of the segregated fund related to the period [Ti−1,Ti]
        
        for i in range(self.row):
            for j in range(self.columns):
                
                self.G[i][j] = np.exp( (-(self.sigma**2) / (2*(self.a**2))) * (self.H - (self.row-1)) - ((self.sigma**2) / (4*self.a)) * (self.H**2) )
                self.P[i][j] = self.G[i][j] * np.exp(-self.H * self.x_values[i][j])
                
                self.H0[i] = (1/self.a) * (1- (np.exp(-self.a * (i+1)))) 
                self.G0[i][j] = np.exp( (-(self.sigma**2) / (2*(self.a**2))) * (self.H0[i] - (i+1)) - ((self.sigma**2) / (4*self.a)) * (self.H0[i]**2) )
                self.P0[i][j] = self.G0[i][j] * np.exp(-self.H0[i] * self.x_values[i+1][j])
                
                self.ZCB[i][j] = (self.yield_curve[0] / self.yield_curve[i]) * (self.P0[i][j] / self.P[0][j]) * self.P[i][j] 
        
                
        for  i in range(self.row-1):
            for j in range(self.columns):
                
                self.Ra[i+1][j] = np.power((1/self.ZCB[i+1][j]), (1/30))-1 #ZCB[i+1] because it select the same time period t=1 for the formula of the rate of return
        
        return self.Ra
        
        
        
    def BEL_computation_bop(self,theta=0.005, g = 0.005, qx = 0.001, fj =0.05): #best of policy
        
        #REVALUATION METHOD: BEST OF POLICIES
        
        #calling the other methods
        self.sdf_simulation()
        self.ESG()
        
        self.theta = theta  #minimum return retained by the insurer
        self.g = g          #guaranteed minimum interest rate
        self.qx = qx        #probability that an individual aged x dies with in the period
        self.fj = fj        #probability that the contract of the model is surrendered

        #maybe we need to increase to 1 the rows
        self.Ci_t = np.zeros((self.row, self.columns))  
        self.Cg_t = np.zeros((self.row, self.columns))
        self.C_t = np.zeros((self.row, self.columns))  #Policy's benefit accrued at time Ti
        
        self.Ci_t[0] = 100 #initialize the first row with 100
        self.Cg_t[0] = 100 #initialize the first row with 100
        self.C_t[0] = 100 #initialize the first row with 100

        self.N_t = np.zeros((self.row, self.columns))  #number of contracts in force at each date
        self.N_t[0] = 100 #initialize the first row with 100

        self.C_q = np.zeros((self.row, self.columns))  #stream of benefits in case of death
        self.C_f = np.zeros((self.row, self.columns))  #stream of benefits in case of surrender
        self.C_j = np.zeros((self.row, self.columns))  #stream of benefits in case of maturity of the contract - { C_t*N_t if maturity, 0 otherwise}



        for i in range(self.row-1):
            for j in range(self.columns):
                
                self.Ci_t[i+1][j] = self.Ci_t[i][j]*(1 + self.Ra[i+1][j] - self.theta) #Ra start from the secon row, Ci_t[i][j] take the previous value
                self.Cg_t[i+1][j] = self.Cg_t[i][j]*(1 +self. g) 
                self.C_t[i+1][j] = max(self.Ci_t[i+1][j], self.Cg_t[i+1][j]) #max btw Ci and Cg
                    
                self.N_t[i+1][j] = self.N_t[i][j]*(1 - self.qx - self.fj) 
                    
                self.C_q[i+1][j] = self.N_t[i][j] * self.qx * self.C_t[i][j]
                self.C_f[i+1][j] = self.N_t[i][j] * self.fj * self.C_t[i][j] 
                    
                    
        self.C_j[self.row-1] = self.C_t[self.row-1]* self.N_t[self.row-1] #matrix multiplication, values all 0 except for last row 
    
    
        self.value_BEL = np.zeros((self.row, self.columns))
        for i in range(self.row-1):
            for j in range(self.columns):                                        
                self.value_BEL[i+1][j] = self.C_q[i+1][j] + self.C_f[i+1][j] + self.C_j[i+1][j] #start from second row, the first is empty -> for C_j only the last row has values
          

        self.final_BEL = np.array( np.sum( self.sdf_values * self.value_BEL, axis = 0)) #sum product, one value for each column
        self.final_BEL = self.final_BEL.mean()
        print(f"The Value of the BEL using the best of policies is: {self.final_BEL}")
        
        return self.final_BEL
    
    
    
    
    def BEL_computation_cliquet(self,theta=0.005, g = 0.005, qx = 0.001, fj =0.05, eta=0.8, h = 0):
        
        #REVALUATION METHOD: CLIQUET
        
        #calling the other methods
        self.sdf_simulation()
        self.ESG()
        
        self.h = h          #technical interest rate
        self.eta = eta      #participation coefficient (0,1]
        self.theta = theta  #minimum return retained by the insurer
        self.g = g          #guaranteed minimum interest rate
        self.qx = qx        #probability that an individual aged x dies with in the period
        self.fj = fj        #probability that the contract of the model is surrendered

        
        self.C_t = np.zeros((self.row, self.columns)) #Policy's benefit accrued at time Ti
        self.delta = np.zeros((self.row, self.columns))
        
        self.C_t[0] = 100 #initialize the first row with 100

        self.N_t = np.zeros((self.row, self.columns))  #number of contracts in force at each date
        self.N_t[0] = 100 #initialize the first row with 100

        self.C_q = np.zeros((self.row, self.columns))  #stream of benefits in case of death
        self.C_f = np.zeros((self.row, self.columns))  #stream of benefits in case of surrender
        self.C_j = np.zeros((self.row, self.columns))  #stream of benefits in case of maturity of the contract - { C_t*N_t if maturity, 0 otherwise}



        for i in range(self.row-1):
            for j in range(self.columns):
                
                self.delta[i+1][j] = (max(min(self.eta* self.Ra[i+1][j], self.Ra[i+1][j] - self.theta), self.g) - self.h  )/ (1 + self.h) 
                self.C_t[i+1][j] = self.C_t[i][j]*(1 + self.delta[i+1][j]) 
                    
                self.N_t[i+1][j] = self.N_t[i][j]*(1 - self.qx - self.fj) 
                    
                self.C_q[i+1][j] = self.N_t[i][j] * self.qx * self.C_t[i][j]
                self.C_f[i+1][j] = self.N_t[i][j] * self.fj * self.C_t[i][j] 
                 
                    
        self.C_j[self.row-1] = self.C_t[self.row-1]* self.N_t[self.row-1] #matrix multiplication, values all 0 except for last row 
    
    
        self.value_BEL = np.zeros((self.row, self.columns))
        for i in range(self.row-1):
            for j in range(self.columns):                                        
                self.value_BEL[i+1][j] = self.C_q[i+1][j] + self.C_f[i+1][j] + self.C_j[i+1][j] #start from second row, the first is empty -> for C_j only the last row has values
          

        self.final_BEL = np.array( np.sum( self.sdf_values * self.value_BEL, axis = 0)) #sum product, one value for each column
        self.final_BEL = self.final_BEL.mean()
        print(f"The Value of the BEL using the cliquet method is: {self.final_BEL}")
        
        return self.final_BEL
    
    

    #define print methods
    def print_rate(self):
        self.rate_simulation()
        print(f'Hull-White Rate: \n{self.r_values}')
    def print_SDF(self):
        self.sdf_simulation()
        print(f'SDF:  \n{self.sdf_values}')
    def print_return_rate(self):
        self.ESG()
        print(self.Ra)
        plt.figure(figsize=(25,15))
        plt.title(r'$R_{a}$ Monte Carlo Simulation', fontsize=30)
        plt.ylabel('Rate',fontsize= 20)
        plt.xlabel('Year',fontsize = 20)
        plt.grid(True)
        plt.plot(self.Ra)
        plt.show()
        
                
    
    
if __name__=='__main__':
    
    """Only the most important methods are called in the main, to explore other functions look at the class methods using print(dir(BEL)) """
    
    #create an instance of the class BEL
    simulation1 = BEL()
    
    
    #plot the rate obtained from HULL-WHITE method
    simulation1.plot_rate_simulation()
    
    
    #Compute the Value of the BEL
    simulation1.BEL_computation_bop()
    simulation1.BEL_computation_cliquet()
    
    simulation1.print_return_rate()
    
    
     
    
    

    
    
    
                        

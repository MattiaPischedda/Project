import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.optimize import newton, brentq
from scipy.interpolate import CubicSpline
import random
plt.style.use('seaborn-v0_8-dark')


def yield_interpolation(yield_curve, step = "annual"):
    
    """Interpolates yield curve data over specified time step intervals.
    Designed for ECB spot rate 3,6,9 months and annual term structure.

    Args:
    yield_curve (list or np.array): array of yield points starting from month 3.
    step (str): Interval step, must be 'annual', 'semiannual', 'monthly', or 'weekly'.

    Returns:
    np.array: Interpolated yield values.
    """     
    
    step_map = {'annual': 1, 'semiannual': 1/2, 'monthly': 1/12}
    step_length = {'annual': 1, 'semiannual': 2, 'monthly': 12}
    
    if step not in step_map:
        raise ValueError("dt must be 'annual', 'semiannual', 'monthly' ")
    
    if step == "annual":
        
        year_final = [rate / 100 for rate in yield_curve[3:]]
        return [(i + 1, year_final[i]) for i in range(len(year_final))]
    
    #divide between the the rates before the first year and after 
    first_year = yield_curve[:4]
    annual_curve = yield_curve[3:]
    
    
    time_step = step_length[step]   # dt 0.5, 0.08..
    step = step_map[step]           # number of steps per year
    
    
    years = np.array(range(1, len(annual_curve)+1))
    cs1 = CubicSpline(years, annual_curve)

    # Generate monthly points
    yields = np.linspace(1, len(annual_curve), time_step * (len(annual_curve)-1) + 1)  # Monthly from year 1 to 30
    converted_yields = cs1(yields)
    
    
    if step == 0.5:
        
        converted_yields = list(converted_yields) #convert into a list to insert the yield of month 6
        converted_yields.insert(0,first_year[1]) 
        converted_yields = np.array(converted_yields)/100 #reconvert to array
        
    
    elif step == (1/12):
    
        months = np.array([3, 6, 9, 12]) / 12  # Existing data points in years
        # Create a cubic spline interpolation (allowing extrapolation)
        cs2 = CubicSpline(months, first_year, extrapolate=True)

        # Define new range including the first and second months
        extended_months = np.linspace(1/12, 1, 12)  # From 1 month to 12 months

        # Calculate interpolated yields for the extended range
        extended_yields = cs2(extended_months)
        extended_yields = extended_yields[:-1] #getting rid of the last element (already present on the annual interpolation)
        
        converted_yields = np.append(extended_yields,converted_yields)/100
        
        
    return [(i*step, converted_yields[i-1]) for i in range(1, time_step*len(annual_curve)+1)]


class trinomial:
    
    def __init__(self,a,sigma, N, n, yield_curve):
        
        
        self.a = a                                                     #mean reversion
        self.sigma = sigma                                             #volatility
        self.N = N                                                     #maturity
        self.n = n                                                     #step                         
        self.dt = N/n                                                  #dt
        self.yield_curve = yield_curve                                 #yield curve 
        

        self.deltaR = self.sigma*np.sqrt(3*self.dt)                                # delta R (space between interest rates)
        self.j_max = math.ceil(0.184/(self.a*self.dt))                             #upper bound
        self.j_min = - self.j_max                                                  #lower bound
        self.rows = min(self.j_max,self.n)*3-(min(self.j_max,self.n)-1)            #number of rows
        self.columns = self.n+1                                                    #number of columns
        
        

        
        
        self.R_star = np.zeros((min(self.j_max,self.n)*3-(min(self.j_max,self.n)-1), self.n+1))             #matrix for rate framework
        self.probability = [[0 for _ in range(self.columns)] for _ in range(self.rows)]      #matrix for probabilities
        self.R = np.zeros((min(self.j_max,self.n)*3-(min(self.j_max,self.n)-1), self.n+1))        #matrix for interest rates
         
    #the r_star_frame method is used to create the basic framework to model the real term structure
    #It also assign the positive/negative node i,j to the R matrix
    def r_star_frame(self):
        
        for i in range(1, self.n+1):
    
    
            upper_bound = min(3 + (i-1)*2  , self.j_max*3-(self.j_max-1))  #increase by 2 until it reach the j_max
            
            
            for j in range(0, upper_bound):
                
                current_j = -(j - min(i,self.j_max))   #compute the node i,j number
                self.R[j][i] = current_j               #fill the R matrix with the respective value
                 
                 
                #iterate until you reach the timestep bound   
                if i <= self.j_max:                       
                    
                    #lower node                    
                    if j == upper_bound-1:
                    
                        self.R_star[j][i] = self.R_star[j-2][i-1] - self.deltaR 
                    
                    #upper node    
                    elif j == 0:
                        
                        self.R_star[j][i] = self.R_star[j][i-1] + self.deltaR 
                    
                    else: 
                        self.R_star[j][i] = self.R_star[j-1][i-1]
                
                
                #in this point the upper/lower bound is reached
                else:
                    
                    #upper bound
                    if current_j >= self.j_max:   
                        
                        self.R_star[j][i] = self.R_star[j][i-1]

                    #lower bound
                    elif current_j <= self.j_min:
                            
                        self.R_star[j][i] = self.R_star[j][i-1]
                        
                    else:
                        
                        self.R_star[j][i] = self.R_star[j][i-1]
        
                        
        return self.R_star, self.R
    
    #the probability method is used to compute the probability associated to each node
    def probability_computation(self):
        
        
        for i in range(0, self.n+1):
    
            upper_bound = min(3 + (i-1)*2  , self.j_max*3-(self.j_max-1))
            
            for j in range(0, upper_bound):
            
                current_j = -(j - min(i,self.j_max)) 
                
                #check if we are below the upper/lower limit    
                if i < self.j_max:            
                    
                    p_up = (1/6) + 0.5*(self.a**2 * current_j**2 * self.dt**2 - self.a*current_j*self.dt)    #up probability
                    p_mid = (2/3) - (self.a**2 * current_j**2 * self.dt**2 )                                 #mid probability
                    p_down = (1/6) + 0.5*(self.a**2 * current_j**2 * self.dt**2 + self.a*current_j*self.dt)  #down probability
                    norm_p = [p_up, p_mid, p_down]
                    self.probability[j][i] =  norm_p
                
                    
                else:
                    
                    #upper bound probabilities
                    if current_j >= self.j_max:
                        
                        p_up_upper = (7/6) + (1/2) * (self.a**2 * current_j**2 * self.dt**2 - 3*self.a*current_j*self.dt)    #up (straight)
                        p_mid_upper = (-1/3) - (self.a**2 * current_j**2 * self.dt**2 ) + (2 * self.a * current_j * self.dt) #mid (down)
                        p_down = (1/6) + 0.5*(self.a**2 * current_j**2 * self.dt**2 - self.a*current_j*self.dt)                   #down (down down)
                        upper_p = [p_up_upper, p_mid_upper, p_down]
                        self.probability[j][i] =  upper_p
                        
                    
                    #lower bound probabilities    
                    elif current_j <= self.j_min:
                        
                        p_up = (1/6) + 0.5*(self.a**2 * current_j**2 * self.dt**2 + self.a*current_j*self.dt)                     #up (up up)
                        p_mid_lower = (-1/3) - (self.a**2 * current_j**2 * self.dt**2 ) - (2 * self.a * current_j * self.dt) #mid (up)
                        p_down_lower = (7/6) + (1/2) * (self.a**2 * current_j**2 * self.dt**2 + 3*self.a*current_j*self.dt)  #down (straight)
                        lower_p = [p_up, p_mid_lower, p_down_lower]
                        
                        self.probability[j][i] =  lower_p                     
                        
                    else:
                        
                        p_up = (1/6) + 0.5*(self.a**2 * current_j**2 * self.dt**2 - self.a*current_j*self.dt)           #up probability
                        p_mid = (2/3) - (self.a**2 * current_j**2 * self.dt**2 )                                        #mid probability
                        p_down = (1/6) + 0.5*(self.a**2 * current_j**2 * self.dt**2 + self.a*current_j*self.dt)         #down probability
                        norm_p = [p_up, p_mid, p_down]
                        self.probability[j][i] =  norm_p

        return self.probability
    


class hull_white(trinomial):   
    
    def __init__(self,a,sigma, N, n, yield_curve):
        
        """
        Compute the trinomial tree for the Hull-White model.

        Args:
        a: mean reversion parameter.
        sigma: volatility.
        N: maturity.
        n: step
        yield_curve (list or np.array): List of tuple.
        """   
        super().__init__(a,sigma, N, n, yield_curve)
        
        
        
    
    #the rate_tree method match the r_star framework to the real term structure inputed through the yield_curve list
    def rate_tree(self):
        
        #Recall R_star, R, and the probability matrix
        self.R_star, self.R = self.r_star_frame()
        self.probability = self.probability_computation()
        
        #initialize Q and the vectors
        #Q is the present value of a security that pays off $1 if node (i, j) is reached
        self.Q = np.zeros((min(self.j_max,self.n)*3-(min(self.j_max,self.n)-1), self.n+1))
        self.zcb = np.zeros(self.n+2)                                                           #ZCB value
        self.alpha = np.zeros(self.n+2)                                                         #alpha value


        self.alpha[0] = self.yield_curve[0][1]  #initialize the alpha vector
        self.R[0][0] = self.yield_curve[0][1]   #initialize the R matrix with the first Rate
        #print(R)

        #compute the ZCB with the provided yield curve
        for i in range(1, self.n+1):
            
            self.zcb[i] = np.exp(-self.yield_curve[i][0]*self.yield_curve[i][1])
            
        #print(f"yield curve: {yield_curve}")
        #print(f"ZCB: {zcb}\n")
            
        
        
        for i in range(1,self.n+1):
            
            upper_bound = min(3 + (i-1)*2  , self.j_max*3-(self.j_max-1))
           
            Q_sum = 0
            
            for j in range(0, upper_bound):
                
                  
                
        #first step -----------------------------------------------------------------------------------------------------------------------------------------       
                if i == 1: 
                
                            
                    Qtemp = self.probability[0][0][j]*np.exp(-self.R[0][0]*self.dt)
                    self.Q[j][i] = Qtemp
                    Q_sum += Qtemp*np.exp(-self.R_star[j][i])
                    
        #second step----------------------------------------------------------------------------------------------------------------------------------------------       
                elif 1 < i <= self.j_max:
                    
                    #extreme upward node - 1 possible path                       
                    if j == 0:
                        
                        Qtemp = self.probability[j][i-1][0]*np.exp(-self.R[j][i-1]*self.dt)*self.Q[0][i-1]
                        #print(f"p {i,j}: {probability[j][i-1][0]}, R: {R[j][i-1]}, Q: {Q[0][i-1]}, R*: {R_star[j][i]}")
                        #print(f"\n")
                        self.Q[j][i] = Qtemp     
                        Q_sum += Qtemp*np.exp(-self.R_star[j][i])
                
                        
                    #upward node - 2 possible paths
                    elif j == 1:
                        
                        Q_node = 0
                        
                        for k in range(2):
                            
                            
                            Qtemp = self.probability[k][i-1][1-k]*np.exp(-self.R[k][i-1]*self.dt)*self.Q[k][i-1]
                            #print(f"p {i,j}: {probability[k][i-1][1-k]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")

                            Q_node += Qtemp
                            
                        #print(f"\n")    
                        self.Q[j][i] = Q_node
                        Q_sum += Q_node*np.exp(-self.R_star[j][i])  
                            
                
                    
                    elif j == upper_bound - 2:  #if size = 5, j = 3, if size = 7, j = 5
                        
                        Q_node = 0
                        
                        for k in range(2):
                            
                            Qtemp = self.probability[j-k-1][i-1][1+k]*np.exp(-self.R[j-k-1][i-1]*self.dt)*self.Q[j-k-1][i-1]
                            #print(f"p {i,j}: {probability[j-k-1][i-1][1+k]}, R: {R[j-k-1][i-1]}, Q: {Q[j-k-1][i-1]}, R*: {R_star[j][i]}")
                            Q_node += Qtemp
                            
                        #print(f"\n")   
                        self.Q[j][i] = Q_node
                        Q_sum += Q_node*np.exp(-self.R_star[j][i])  
                    
                
                    elif j == upper_bound - 1:
                        
                        Qtemp = self.probability[j-2][i-1][2]*np.exp(-self.R[j-2][i-1]*self.dt)*self.Q[j-2][i-1]
                        #print(f"p {i,j}: {probability[j-2][i-1][2]}, R: {R[j-2][i-1]}, Q: {Q[j-2][i-1]}, R*: {R_star[j][i]}")
                        #print(f"\n")
                        self.Q[j][i] = Qtemp
                        Q_sum += Qtemp*np.exp(-self.R_star[j][i])  
                        
                    else:
                        
                        Q_node = 0
                        
                        for k in range(3):
                        
                            Qtemp = self.probability[j+k-2][i-1][2-k]*np.exp(-self.R[j+k-2][i-1]*self.dt)*self.Q[j+k-2][i-1]
                            #print(f"p {i,j}: {probability[j+k-2][i-1][2-k]}, R: {R[j+k-2][i-1]}, Q: {Q[j+k-2][i-1]}, R*: {R_star[j][i]}")
                            Q_node += Qtemp
                            
                        #print(f"\n")   
                        self.Q[j][i] = Q_node
                        Q_sum += Q_node*np.exp(-self.R_star[j][i])  
        
            
                
        #Limit reached-------------------------------------------------------------------------------------------------------------------------------------------
                
                else:
                
        #check if it is the third step, after that all the same-----------------------------------------------------------------------------------------------------   
                    if i == 3 or self.R.shape[0] == 5:
                        
                        
                        if j == 0:
                            
                            Q_node = 0
                        
                            for k in range(2):
                                    
                                Qtemp = self.probability[k][i-1][0]*np.exp(-self.R[k][i-1]*self.dt)*self.Q[k][i-1]
                                #print(f"p {i,j}: {probability[k][i-1][0]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")

                                Q_node += Qtemp
                                
                            #print(f"\n")    
                            self.Q[j][i] = Q_node
                            Q_sum += Q_node*np.exp(-self.R_star[j][i]) 
                            
                        #second node    
                        elif j == 1:
                            
                            Q_node = 0
                        
                            for k in range(3):
                                
                                if k <= 1:
                                    Qtemp = self.probability[k][i-1][1]*np.exp(-self.R[k][i-1]*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][1]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[k][i-1][0]*np.exp(-self.R[k][i-1]*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][0]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum += Q_node*np.exp(-self.R_star[j][i])
                            
                        #the central node at t=3 can be reached by 5 position    
                        elif j == 2:
                            
                            Q_node = 0
                        
                            for k in range(5):
                                
                                
                                if k <= 1:
                                    Qtemp = self.probability[k][i-1][2]*np.exp(-self.R[k][i-1]*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][2]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                elif k == 2:
                                    Qtemp = self.probability[2][i-1][1]*np.exp(-self.R[2][i-1]*self.dt)*self.Q[2][i-1]
                                    #print(f"p {i,j}: {probability[2][i-1][1]}, R: {R[2][i-1]}, Q: {Q[2][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[k][i-1][0]*np.exp(-self.R[k][i-1]*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][0]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum += Q_node*np.exp(-self.R_star[j][i])
                        
                        #fourth node    
                        elif j == 3:
                            
                            Q_node = 0
                        
                            for k in range(3):
                                
                                if k == 0:
                                    Qtemp = self.probability[j+k-1][i-1][2]*np.exp(-self.R[j+k-1][i-1]*self.dt)*self.Q[j+k-1][i-1]
                                    #print(f"p {i,j}: {probability[j+k-1][i-1][2]}, R: {R[j+k-1][i-1]}, Q: {Q[j+k-1][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                else:
                                    Qtemp = self.probability[j+k-1][i-1][1]*np.exp(-self.R[j+k-1][i-1]*self.dt)*self.Q[j+k-1][i-1]
                                    #print(f"p {i,j}: {probability[j+k-1][i-1][1]}, R: {R[j+k-1][i-1]}, Q: {Q[j+k-1][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum += Q_node*np.exp(-self.R_star[j][i])
                                
                        #fifth node
                        elif j == 4:
                        
                            Q_node = 0
                        
                            for k in range(2):
                                
                                Qtemp = self.probability[4-k][i-1][2]*np.exp(-self.R[4-k][i-1]*self.dt)*self.Q[4-k][i-1]
                                #print(f"p {i,j}: {probability[4-k][i-1][2]}, R: {R[4-k][i-1]}, Q: {Q[4-k][i-1]}, R*: {R_star[j][i]}")
                                Q_node += Qtemp
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum += Q_node*np.exp(-self.R_star[j][i]) 

                                        
        #from here all the same--------------------------------------------------------------------------------------------------------------------------------------
                            

        #------------------------------------------------------------------------------------------------------------------------------------------------------------            
                    else:
                        
                        if j == 0:
                            
                            Q_node = 0
                        
                            for k in range(2):
                                    
                                Qtemp = self.probability[k][i-1][0]*np.exp(-self.R[k][i-1]*self.dt)*self.Q[k][i-1]
                                #print(f"p {i,j}: {probability[k][i-1][0]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")

                                Q_node += Qtemp
                                
                            #print(f"\n")    
                            self.Q[j][i] = Q_node
                            Q_sum += Q_node*np.exp(-self.R_star[j][i]) 
                            
                        #second node    
                        elif j == 1:
                            
                            Q_node = 0
                        
                            for k in range(3):
                                
                                if k <= 1:
                                    Qtemp = self.probability[k][i-1][1]*np.exp(-self.R[k][i-1]*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][1]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[k][i-1][0]*np.exp(-self.R[k][i-1]*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][0]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum += Q_node*np.exp(-self.R_star[j][i]*self.dt)
                        
                        
                        #the node at j = 2 can be reached by four different nodes 
                        elif j == 2:
                            
                            Q_node = 0
                        
                            for k in range(4):
                                
                                
                                if k <= 1:
                                    Qtemp = self.probability[k][i-1][2]*np.exp(-self.R[k][i-1]*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][2]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[k][i-1][3-k]*np.exp(-self.R[k][i-1]*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][3-k]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum += Q_node*np.exp(-self.R_star[j][i])
                            
                        
                        #the node at position -2 in the matrix, can be reached by four different nodes 
                        elif j == (upper_bound-3):
                            
                            Q_node = 0
                        
                            for k in range(4):
                                
                                if  k >= 2:
                                    Qtemp = self.probability[-k-1][i-1][-1+k]*np.exp(-self.R[-k-1][i-1]*self.dt)*self.Q[-k-1][i-1]
                                    #print(f"p {i,j}: {probability[-k-1][i-1][-1+k]}, R: {R[-k-1][i-1]}, Q: {Q[-k-1][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[-k-1][i-1][0]*np.exp(-self.R[-k-1][i-1]*self.dt)*self.Q[-k-1][i-1]
                                    #print(f"p {i,j}: {probability[-k-1][i-1][0]}, R: {R[-k-1][i-1]}, Q: {Q[-k-1][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                        
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum += Q_node*np.exp(-self.R_star[j][i])    
                        
                        
                        #fourth node    
                        elif j == (upper_bound-2):
                            
                            Q_node = 0
                        
                            for k in range(3):
                                
                                if k == 0:
                                    Qtemp = self.probability[-3][i-1][2]*np.exp(-self.R[-3][i-1]*self.dt)*self.Q[-3][i-1]
                                    #print(f"p {i,j}: {probability[-3][i-1][2]}, R: {R[-3][i-1]}, Q: {Q[-3][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                else:
                                    Qtemp = self.probability[-3+k][i-1][1]*np.exp(-self.R[-3+k][i-1]*self.dt)*self.Q[-3+k][i-1]
                                    #print(f"p {i,j}: {probability[-3+k][i-1][1]}, R: {R[-3+k][i-1]}, Q: {Q[-3+k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum += Q_node*np.exp(-self.R_star[j][i])
                                
                                
                        #lower limit node, it can be reached by 2 other nodes
                        elif j == (upper_bound-1):
                           
                            Q_node = 0
                        
                            for k in range(2):
                                
                                Qtemp = self.probability[-2+k][i-1][2]*np.exp(-self.R[-2+k][i-1]*self.dt)*self.Q[-2+k][i-1]
                                #print(f"p {i,j}: {probability[-2+k][i-1][2]}, R: {R[-2+k][i-1]}, Q: {Q[-2+k][i-1]}, R*: {R_star[j][i]}")
                                Q_node += Qtemp
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum += Q_node*np.exp(-self.R_star[j][i])
                        
                        else:
                            
                            Q_node = 0   #i.e. if size = 7, j = 3
                        
                            for k in range(3):
                            
                                Qtemp = self.probability[j+k-1][i-1][2-k]*np.exp(-self.R[j+k-1][i-1]*self.dt)*self.Q[j+k-1][i-1]
                                #print(f"p {i,j}: {probability[j+k-1][i-1][2-k]}, R: {R[j+k-1][i-1]}, Q: {Q[j+k-1][i-1]}, R*: {R_star[j][i]}")
                                Q_node += Qtemp
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum += Q_node*np.exp(-self.R_star[j][i])  #check the discounting R_star
        
                            
                        
                    
#Match the term structure-----------------------------------------------------------------------------------------------------------------------------------------           


                        
                        
            self.alpha[i] = (np.log(Q_sum)-np.log(self.zcb[i]))/self.dt
            #print(f"a: {alpha[i+1]}, Q_sum: {Q_sum,}, zcb: {zcb[i]}")
            

            for j in range(0, upper_bound):
                
                
                #print(f"R: {R[j][i], a: {alpha[i]}}")
                self.R[j][i] = self.alpha[i] + self.R_star[j][i]
            
        
                  

        return self.R
         
    
    
    #compute bond price
    def bond_price(self, face_value, coupon):
        
        """This method is used to compute the tree of the bond
            Args.
            face_value: face value of the bond.
            coupon: coupon value in decimals.

        Returns:
            _type_: numpy array
        """
        
        self.face_value = face_value                                            #face value of the bond
        self.coupon = coupon                                                    #coupon
        self.last_cashflow = self.face_value + (self.face_value*self.coupon)    #cashflow for the last period
    
        #recall the rate matrix and the probabilities, generate bond matrix
        self.R = self.rate_tree()   
        self.probability = self.probability_computation()
        self.bond = np.zeros_like(self.R)                                       #bond matrix
        
        
        #fill the last column
        for i in range(self.R.shape[0]):
            
            self.bond[i][-1] = self.last_cashflow/(1+self.R[i][-1])
            

        for i in range(self.R.shape[1]-2, -1, -1):
            
            
            
            if i >= self.j_max:
                
                
                
                for j in range(self.R.shape[0]):
                        
                    
                    
                    if j == 0:
                        
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},bond: {bond[j+k][i+1]}")
                            price_sum += self.probability[j][i][k]*self.bond[j+k][i+1]
                            
                        price_sum += (coupon*face_value)
                        price_sum = price_sum/(1+self.R[j][i])
                        
                        self.bond[j][i] = price_sum
                        #print(f"\n")     
                            
                    
                    elif j == self.R.shape[0]-1:
                        
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},bond: {bond[j+k-2][i+1]}")
                            price_sum += self.probability[j][i][k]*self.bond[j+k-2][i+1]
                            
                        price_sum += (coupon*face_value)
                        price_sum = price_sum/(1+self.R[j][i])
                        
                        self.bond[j][i] = price_sum
                        #print(f"\n") 
                        
                    else:
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},bond: {bond[j+k-1][i+1]}")
                            price_sum += self.probability[j][i][k]*self.bond[j+k-1][i+1]
                            
                        price_sum += (coupon*face_value)
                        price_sum = price_sum/(1+self.R[j][i])
                        
                        self.bond[j][i] = price_sum
                        #print(f"\n") 
                        
                        
                        
                    
            else:
                
                
                for j in range(self.R.shape[0]):
                    
                    if self.R[j][i] == 0:
                        continue
                        
                    else:
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},bond: {bond[j+k][i+1]}")
                            price_sum += self.probability[j][i][k]*self.bond[j+k][i+1]
                            
                        price_sum += (coupon*face_value)
                        price_sum = price_sum/(1+self.R[j][i])
                        
                        self.bond[j][i] = price_sum
                        
                        #print(f"\n")  
                        
        return self.bond          
            
    
    #compute bond price with embedded option
    def embedded_option_bond(self, face_value, coupon,opt_type, value, conversion_period = 1):
        
        """This method is used to compute the tree of the bond
            Args.
            face_value: face value of the bond.
            coupon: coupon value in decimals.
            opt_type: string C or P for callable or putable bond.
            value:.
            conversion_period: default 1, otherwise select period (i.e. 1,2,3,4)
        Returns:
            _type_: numpy array
            print: value of the call or put option
        """
        
        self.face_value = face_value                                            #face value of the bond
        self.coupon = coupon                                                    #coupon
        self.last_cashflow = self.face_value + (self.face_value*self.coupon)    #cashflow for the last period
        self.opt_type = opt_type                                                # callable or putable type
        self.value = value                                                      #callling price
        self.option = 0
        self.conversion_period = conversion_period
    
        #recall the rate matrix and the probabilities, generate bond matrix
        self.R = self.rate_tree()   
        self.probability = self.probability_computation()
        self.bond = np.zeros_like(self.R)                                       #bond matrix
        
        #value of the bond at node 0:
        normal_bond = self.bond_price(face_value,coupon)
        normal_bond = normal_bond[0][0]
        
        
        
        try:
            if self.opt_type == "C":
        #fill the last column
                for i in range(self.R.shape[0]):
                    
                    self.bond[i][-1] = min(self.last_cashflow/(1+self.R[i][-1]), self.value)
                    

                for i in range(self.R.shape[1]-2, -1, -1):
                
             
                    if i >= self.j_max:
                        
                            
                        for j in range(self.R.shape[0]):
                                
                                    
                            if j == 0:
                                
                                
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},bond: {bond[j+k][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                
                                if i >= self.conversion_period:
                                
                                    self.bond[j][i] = min(price_sum,self.value)
                                    
                                else: 
                                    self.bond[j][i] = price_sum  
                                    
                            
                            elif j == self.R.shape[0]-1:
                                
                                
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},bond: {bond[j+k-2][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k-2][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                
                                    self.bond[j][i] = min(price_sum,self.value)
                                    
                                else: 
                                    self.bond[j][i] = price_sum   
                                
                            else:
                                
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},bond: {bond[j+k-1][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k-1][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                
                                    self.bond[j][i] = min(price_sum,self.value)
                                    
                                else: 
                                    self.bond[j][i] = price_sum   
                                         
                    else:
                        
                        
                        for j in range(self.R.shape[0]):
                            
                            if self.R[j][i] == 0:
                                continue
                                
                            else:
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},bond: {bond[j+k][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                
                                    self.bond[j][i] = min(price_sum,self.value)
                                    
                                else: 
                                    self.bond[j][i] = price_sum    
                
                self.option = normal_bond - self.bond[0][0]
                                
                
                
            elif self.opt_type == "P":
                  
            #fill the last column
                for i in range(self.R.shape[0]):
                        
                    self.bond[i][-1] = max(self.last_cashflow/(1+self.R[i][-1]), self.value)
                        
                    
                for i in range(self.R.shape[1]-2, -1, -1): 
                    
                    if i >= self.j_max:
                        
                        
                        
                        for j in range(self.R.shape[0]):
                                
                            
                            
                            if j == 0:
                                
                                
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},bond: {bond[j+k][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                    
                                    self.bond[j][i] = max(price_sum,self.value)
                                else:
                                    
                                    self.bond[j][i] = price_sum
                                    
                            
                            elif j == self.R.shape[0]-1:
                                
                                
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},bond: {bond[j+k-2][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k-2][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                    
                                    self.bond[j][i] = max(price_sum,self.value)
                                else:
                                    
                                    self.bond[j][i] = price_sum
                                
                            else:
                                
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},bond: {bond[j+k-1][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k-1][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                    
                                    self.bond[j][i] = max(price_sum,self.value)
                                else:
                                    
                                    self.bond[j][i] = price_sum
                                            
                    else:
                        
                        
                        for j in range(self.R.shape[0]):
                            
                            if self.R[j][i] == 0:
                                continue
                                
                            else:
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},bond: {bond[j+k][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                    
                                    self.bond[j][i] = max(price_sum,self.value)
                                else:
                                    
                                    self.bond[j][i] = price_sum
                
                self.option = self.bond[0][0] - normal_bond 
        
            else:  
                print(f"Option type must be C for the Callable or P for the Putable")
        
        except ValueError as e:
            print(f"ValueError: {e}")

        except Exception as e:
            
            print(f"An unexpected error occurred: {e}")
             
                  
        print(f"The Option value is {round(self.option,4)}")                    
        return self.bond 
    
    
    #compute interest rate swap
    def swap_tree(self, principal, pay):
        
        """This method is used to compute the value of a Interest Rate Swap.
            Args.
            principal: amount of the principal.
            pay: payment frequency 

        Returns:
            _type_: numpy Array
        """
        
        self.R = self.rate_tree()
        self.probability = self.probability_computation()
        
        self.discount = []
        self.principal = principal
        self.pay = pay
        
        #compute the discount factors
        for i in range(self.n+1):
        
            disc = np.exp(-(self.yield_curve[i][1])*self.yield_curve[i][0])
            self.discount.append(disc)
            
        
        #compute the swap rate
        self.swap_rate = self.pay * ((1-self.discount[-1])/(np.sum(self.discount)))
        
        self.cf = np.zeros_like(self.R) #MATRIX FOR CF
        self.swap = np.zeros_like(self.R) #matrix for swap   
        
        
        self.cf[0][0] = self.principal * self.dt*(self.pay*(np.exp(self.R[0][0]* self.dt)-1)-self.swap_rate)
        
        
        #fill the matrix
        for i in range(1, self.R.shape[1]):
            
            
            for j in range(0, self.R.shape[0]):
                
                if self.R[j][i] == 0:
                    continue
                else:
                    
                    self.cf[j][i] = self.principal*self.dt*(self.pay*(np.exp(self.R[j][i]*self.dt)-1)-self.swap_rate)


        #-----------------------------------------------------------------------------------------------------------------
        #COMPUTE THE SWAP VALUE
        
        #fill the last column
        for i in range(self.R.shape[0]):
            
            self.swap[i][-1] = self.cf[i][-1]/(1+self.R[i][-1]*self.dt)
            

        for i in range(self.R.shape[1]-2, -1, -1):
            
            
            if i >= self.j_max:
                
                
                
                for j in range(self.R.shape[0]):
                        
                    
                    
                    if j == 0:
                        
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},swap: {swap[j+k][i+1]}")
                            price_sum += self.probability[j][i][k]*self.swap[j+k][i+1]
                            
                        price_sum += self.cf[j][i]
                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        self.swap[j][i] = price_sum
                        #print(f"\n")     
                            
                    
                    elif j == self.R.shape[0]-1:
                        
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {swap[j+k-2][i+1]}")
                            price_sum += self.probability[j][i][k]*self.swap[j+k-2][i+1]
                            
                        price_sum += self.cf[j][i]
                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        self.swap[j][i] = price_sum
                        #print(f"\n") 
                        
                    else:
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {swap[j+k-1][i+1]}")
                            price_sum += self.probability[j][i][k]*self.swap[j+k-1][i+1]
                            
                        price_sum += self.cf[j][i]
                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        self.swap[j][i] = price_sum
                        #print(f"\n") 
                        
                        
                        
                    
            else:
                
                
                for j in range(self.R.shape[0]):
                    
                    if self.R[j][i] == 0:
                        continue
                        
                    else:
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {swap[j+k][i+1]}")
                            price_sum += self.probability[j][i][k]*self.swap[j+k][i+1]
                            
                        price_sum += self.cf[j][i]
                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        self.swap[j][i] = price_sum
                        #print(f"\n") 
                        


        return self.swap
    
    
    #compute swaption
    def swaption(self,principal,pay, exercise=None):
        
        """This method is used to compute the value of a Interest Rate Swap.
            Args.
            principal: amount of the principal.
            pay: payment frequency.
            exercise: python list --> used to specifit American,European, Bermudian style.
            American is the default, insert the last time step for the European, specify the time step for the Bermudian.
            
        Returns:
            _type_: numpy Array
        """
        
        self.R = self.rate_tree()
        self.probability = self.probability_computation()
        self.S = self.swap_tree(principal,pay)
        self.exercise = exercise
        self.C = np.zeros_like(self.S)
        
        
        #-----------------------------------------------------------------------------------------------------------------
        #COMPUTE THE SWAPTION VALUE
        
        #fill the last column
        for i in range(self.R.shape[0]):
            
            self.C[i][-1] = max(0, self.S[i][-1])
            

        for i in range(self.R.shape[1]-2, -1, -1):
            
            if i == 0:
                
                for j in range(self.R.shape[0]):
                    
                    if self.R[j][i] == 0:
                        continue
                        
                    else:
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"node: {j,i}prob: {self.probability[j][i][k]},swap: {self.swap[j+k][i+1]}, S: {self.C[j+k][i+1]}")
                            price_sum += self.probability[j][i][k]*self.C[j+k][i+1]

                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        self.C[j][i] = price_sum
                        
            
        
            
            elif i >= self.j_max:
                
                   
                for j in range(self.R.shape[0]):
                        
                         
                    if j == 0:
                        
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"node: {j,i}, prob: {self.probability[j][i][k]},swap: {self.S[j+k][i+1]}, S: {self.C[j+k][i+1]}")
                            price_sum += self.probability[j][i][k]*self.C[j+k][i+1]
                            
                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        if self.exercise == None or i in self.exercise:
                            
                            self.C[j][i] = max(price_sum, 0)
                        else:
                            self.C[j][i] = price_sum
                            
                        #print(f"\n")     
                            
                    
                    elif j == self.R.shape[0]-1:
                        
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {swap[j+k-2][i+1]}")
                            price_sum += self.probability[j][i][k]*self.C[j+k-2][i+1]
                            
                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        if self.exercise == None or i in self.exercise:
                            self.C[j][i] = max(price_sum, 0)
                        else:
                            self.C[j][i] = price_sum
                        #print(f"\n") 
                        
                    else:
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {swap[j+k-1][i+1]}")
                            price_sum += self.probability[j][i][k]*self.C[j+k-1][i+1]

                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        if self.exercise == None or i in self.exercise:
                            self.C[j][i] = max(price_sum, 0)
                        else:
                            self.C[j][i] = price_sum
                        
                                      
                    
            else:
                
                
                for j in range(self.R.shape[0]):
                    
                    if self.R[j][i] == 0:
                        continue
                        
                    else:
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"node: {j,i}, prob: {self.probability[j][i][k]},swap: {self.S[j+k][i+1]}, C: {self.C[j+k][i+1]} ")
                            price_sum += self.probability[j][i][k]*self.C[j+k][i+1]

                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        if self.exercise == None or i in self.exercise:
                            self.C[j][i] = max(price_sum, 0)
                        else:
                            self.C[j][i] = price_sum
                        #print(f"\n") 
                        

        return self.C
    
        
    #print the rate tree
    def print_tree(self):
        
        rate = self.rate_tree()*100
        
        rows, cols = rate.shape
        middle_row = rows // 2
        centered_matrix = np.zeros_like(rate)
        
        for col in range(cols):
            # Extract non-zero elements in column
            non_zeros = rate[:, col][rate[:, col] != 0]
            start_row = middle_row - len(non_zeros) // 2
            
            # Place elements starting from the calculated start row
            for i, value in enumerate(non_zeros):
                centered_matrix[start_row + i, col] = value
        
        # Apply the centering function to the new array R

        df = pd.DataFrame(centered_matrix)
        df.replace(0, '-',inplace=True)
        
        return df
    
    
    #method to print the bond matrix
    def print_bond(self, face_value, coupon):
        
        """This method is used to compute the tree of the bond
            Args.
            face_value: face value of the bond.
            coupon: coupon value in decimals.

        Returns:
            _type_: pandas DataFrame
        """
        
        bond = self.bond_price(face_value, coupon)
        
        rows, cols = bond.shape
        middle_row = rows // 2
        centered_matrix = np.zeros_like(bond)
        
        for col in range(cols):
            # Extract non-zero elements in column
            non_zeros = bond[:, col][bond[:, col] != 0]
            start_row = middle_row - len(non_zeros) // 2
            
            # Place elements starting from the calculated start row
            for i, value in enumerate(non_zeros):
                centered_matrix[start_row + i, col] = value
        
        # Apply the centering function to the new array R

        df = pd.DataFrame(centered_matrix)
        df.replace(0, '-',inplace=True)
        
        return df
    
    
    #method to print the bond matrix
    def print_embedded_option_bond(self, face_value, coupon, opt_type, value, conversion_period = 1):
        
        """This method is used to print the tree of the bond
            Args.
            face_value: face value of the bond.
            coupon: coupon value in decimals.
            opt_type: string C or P for callable or putable.
            value:.
            conversion_period: default 1, otherwise select period (i.e. 1,2,3,4)

        Returns:
            _type_: pandas DataFrame
        """
        
        bond = self.embedded_option_bond(face_value, coupon, opt_type, value, conversion_period)
        
        rows, cols = bond.shape
        middle_row = rows // 2
        centered_matrix = np.zeros_like(bond)
        
        for col in range(cols):
            # Extract non-zero elements in column
            non_zeros = bond[:, col][bond[:, col] != 0]
            start_row = middle_row - len(non_zeros) // 2
            
            # Place elements starting from the calculated start row
            for i, value in enumerate(non_zeros):
                centered_matrix[start_row + i, col] = value
        
        # Apply the centering function to the new array R

        df = pd.DataFrame(centered_matrix)
        df.replace(0, '-',inplace=True)
        
        return df
    
    # get alpha value
    def get_alpha(self):
        
        self.rate_tree()
        
        return self.alpha
        
        
    #print interest rate swap tree   
    def print_swap(self, principal, pay):
        
        """This method is used to print the value of a Interest Rate Swap.
            Args.
            principal: amount of the principal.
            pay: payment frequency 

        Returns:
            _type_: Pandas DataFrame
        """
            
        S = np.round(self.swap_tree(principal, pay),2) 
        
        rows, cols = S.shape
        middle_row = rows // 2
        centered_matrix = np.zeros_like(S)
        
        for col in range(cols):
            # Extract non-zero elements in column
            non_zeros = S[:, col][S[:, col] != 0]
            start_row = middle_row - len(non_zeros) // 2
            
            # Place elements starting from the calculated start row
            for i, value in enumerate(non_zeros):
                centered_matrix[start_row + i, col] = value
        
        # Apply the centering function to the new array R

        df = pd.DataFrame(centered_matrix)
        
        # Replace 0 with '-' except for the value in the middle row of the first column
        for i in range(rows):
            for j in range(cols):
                if centered_matrix[i, j] == 0:
                    if not (i == middle_row and j == 0):
                        df.iloc[i, j] = '-'
        
        return df
   

        


class black_Karasinski(trinomial):   
    
    def __init__(self,a,sigma, N, n, yield_curve):
        
        
        """
        Compute the trinomial tree for the Black-Karasinski model.

        Args:
        a: mean reversion parameter.
        sigma: volatility.
        N: maturity.
        n: steps.
        yield_curve (list or np.array): List of tuple.
        """
        super().__init__(a,sigma, N, n, yield_curve)
    
    
    #the rate_tree method match the r_star framework to the real term structure inputed through the yield_curve list
    def rate_tree(self, optimization_type: str = "newton"):
        
        """Optimization type must be of type string 'newton' or 'brentq' """
        
        if optimization_type not in ['newton', 'brentq']:
            raise ValueError("optimization_type must be 'newton' or 'brentq' ")
        
        self.type = optimization_type
        
        #Recall R_star, R, and the probability matrix
        self.R_star, self.R = self.r_star_frame()
        self.probability = self.probability_computation()
        
        self.Q = np.zeros((min(self.j_max,self.n)*3-(min(self.j_max,self.n)-1), self.n+1))
        self.X = np.zeros((min(self.j_max,self.n)*3-(min(self.j_max,self.n)-1), self.n+1))
        self.zcb = np.zeros(self.n+2)
        self.alpha = np.zeros(self.n+2)


        self.alpha[0] = self.yield_curve[0][1]
        self.R[0][0] = self.yield_curve[0][1]
        self.X[0][0] = np.log(self.R[0][0])
        



        for i in range(1, self.n+1):
            
            self.zcb[i] = np.exp(-self.yield_curve[i][0]*self.yield_curve[i][1])
            
        #print(f"yield curve: {yield_curve}")
        #print(f"ZCB: {zcb}\n")
            
        
        
        for i in range(1,self.n+1):
            
            upper_bound = min(3 + (i-1)*2  , self.j_max*3-(self.j_max-1))
          
            Q_sum = []
            
            
            for j in range(0, upper_bound):
                
                
        #first step -----------------------------------------------------------------------------------------------------------------------------------------       
                if i == 1: 
                
                    
                    Qtemp = self.probability[0][0][j]*np.exp(-np.exp(self.X[0][0])*self.dt)
                    #print(f"prob: {round(probability[0][0][j],5)}, X: {X[0][0]}")
                    self.Q[j][i] = Qtemp
                    Q_sum.append(Qtemp)
                    
        #second step----------------------------------------------------------------------------------------------------------------------------------------------       
                elif 1 < i <= self.j_max:
                    
                    #extreme upward node - 1 possible path                       
                    if j == 0:
                        
                    
                        Qtemp = self.probability[j][i-1][0]*np.exp(-np.exp(self.X[j][i-1])*self.dt)*self.Q[0][i-1]
                        #print(f"p {i,j}: {probability[j][i-1][0]}, X: {X[j][i-1]}, Q: {Q[0][i-1]}, R*: {R_star[j][i]}")
                        #print(f"\n")
                        
                        
                        #print(f"Q: {Qtemp}")
                        self.Q[j][i] = Qtemp     
                        Q_sum.append(Qtemp)
                
                        
                    #upward node - 2 possible paths
                    elif j == 1:
                        
                
                        Q_node = 0
                        
                        for k in range(2):
                            
                            
                            Qtemp = self.probability[k][i-1][1-k]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                            #print(f"p {i,j}: {probability[k][i-1][1-k]}, X: {X[k][i-1]}, Q: {Q[k][i-1]}")

                            Q_node += Qtemp
                        
                        #print(f"Q: {Q_node}")   
                        #print(f"\n")    
                        self.Q[j][i] = Q_node
                        Q_sum.append(Q_node)
                            
                
                    
                    elif j == upper_bound - 2:  #if size = 5, j = 3, if size = 7, j = 5
                        
                        
                        Q_node = 0
                        
                        for k in range(2):
                            
                            Qtemp = self.probability[j-k-1][i-1][1+k]*np.exp(-np.exp(self.X[j-k-1][i-1])*self.dt)*self.Q[j-k-1][i-1]
                            #print(f"p {i,j}: {probability[j-k-1][i-1][1+k]}, X: {X[j-k-1][i-1]}, Q: {Q[j-k-1][i-1]}")
                            Q_node += Qtemp
                        
                        #print(f"Q: {Q_node}")   
                        #print(f"\n")   
                        self.Q[j][i] = Q_node
                        Q_sum.append(Q_node)  
                    
                
                    elif j == upper_bound - 1:
                        
                        
                        Qtemp = self.probability[j-2][i-1][2]*np.exp(-np.exp(self.X[j-2][i-1])*self.dt)*self.Q[j-2][i-1]
                        #print(f"p {i,j}: {probability[j-2][i-1][2]}, X: {X[j-2][i-1]}, Q: {Q[j-2][i-1]}")
                        #print(f"\n")
                        #print(f"Q: {Qtemp}")
                        self.Q[j][i] = Qtemp
                        Q_sum.append(Qtemp) 
                        
                    else:
                        
                        Q_node = 0
                        
                        for k in range(3):
                        
                            Qtemp = self.probability[j+k-2][i-1][2-k]*np.exp(-np.exp(self.X[j+k-2][i-1])*self.dt)*self.Q[j+k-2][i-1]
                            #print(f"p {i,j}: {probability[j+k-2][i-1][2-k]}, X: {X[j+k-2][i-1]}, Q: {Q[j+k-2][i-1]}, R*: {R_star[j][i]}")
                            Q_node += Qtemp
                            
                        #print(f"Q: {Q_node}")   
                        #print(f"\n")   
                        self.Q[j][i] = Q_node
                        Q_sum.append(Q_node) 
        
            
                
        #Limit reached-------------------------------------------------------------------------------------------------------------------------------------------

                else:
                
        #check if it is the third step, after that all the same-----------------------------------------------------------------------------------------------------   
                    if i == 3 or self.R.shape[0] == 5:
                        
                        
                        if j == 0:
                            
                            Q_node = 0
                        
                            for k in range(2):
                                    
                                Qtemp = self.probability[k][i-1][0]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                #print(f"p {i,j}: {probability[k][i-1][0]}, X: {X[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")

                                Q_node += Qtemp
                                
                            #print(f"\n")    
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)
                            
                        #second node    
                        elif j == 1:
                            
                            Q_node = 0
                        
                            for k in range(3):
                                
                                if k <= 1:
                                    Qtemp = self.probability[k][i-1][1]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][1]}, X: {X[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[k][i-1][0]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][0]}, X: {X[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)
                            
                        #the central node at t=3 can be reached by 5 position    
                        elif j == 2:
                            
                            Q_node = 0
                        
                            for k in range(5):
                                
                                
                                if k <= 1:
                                    Qtemp = self.probability[k][i-1][2]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][2]}, X: {X[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                elif k == 2:
                                    Qtemp = self.probability[2][i-1][1]*np.exp(-np.exp(self.X[2][i-1])*self.dt)*self.Q[2][i-1]
                                    #print(f"p {i,j}: {probability[2][i-1][1]}, X: {X[2][i-1]}, Q: {Q[2][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[k][i-1][0]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][0]}, X: {X[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)
                        
                        #fourth node    
                        elif j == 3:
                            
                            Q_node = 0
                        
                            for k in range(3):
                                
                                if k == 0:
                                    Qtemp = self.probability[j+k-1][i-1][2]*np.exp(-np.exp(self.X[j+k-1][i-1])*self.dt)*self.Q[j+k-1][i-1]
                                    #print(f"p {i,j}: {probability[j+k-1][i-1][2]}, X: {X[j+k-1][i-1]}, Q: {Q[j+k-1][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                else:
                                    Qtemp = self.probability[j+k-1][i-1][1]*np.exp(-np.exp(self.X[j+k-1][i-1])*self.dt)*self.Q[j+k-1][i-1]
                                    #print(f"p {i,j}: {probability[j+k-1][i-1][1]}, X: {X[j+k-1][i-1]}, Q: {Q[j+k-1][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)
                                
                        #fifth node
                        elif j == 4:
                        
                            Q_node = 0
                        
                            for k in range(2):
                                
                                Qtemp = self.probability[4-k][i-1][2]*np.exp(-np.exp(self.X[4-k][i-1])*self.dt)*self.Q[4-k][i-1]
                                #print(f"p {i,j}: {probability[4-k][i-1][2]}, X: {X[4-k][i-1]}, Q: {Q[4-k][i-1]}, R*: {R_star[j][i]}")
                                Q_node += Qtemp
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)

        #from here all the same--------------------------------------------------------------------------------------------------------------------------------------
                            

        #------------------------------------------------------------------------------------------------------------------------------------------------------------            
                    else:
            #ok            
                        if j == 0:
                            
                            Q_node = 0
                        
                            for k in range(2):
                                    
                                Qtemp = self.probability[k][i-1][0]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                #print(f"p {i,j}: {probability[k][i-1][0]}, X: {X[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")

                                Q_node += Qtemp
                                
                            #print(f"\n")    
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)
                            
                        #second node    
                        elif j == 1:
            #ok                
                            Q_node = 0
                        
                            for k in range(3):
                                
                                if k <= 1:
                                    Qtemp = self.probability[k][i-1][1]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][1]}, X: {X[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[k][i-1][0]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][0]}, X: {X[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)
                        
                        
                        #the node at j = 2 can be reached by four different nodes 
                        elif j == 2:
            #ok                
                            Q_node = 0
                        
                            for k in range(4):
                                
                                
                                if k <= 1:
                                    Qtemp = self.probability[k][i-1][2]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][2]}, X: {X[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[k][i-1][3-k]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][3-k]}, X: {X[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)
                            
                        
                        #the node at position -2 in the matrix, can be reached by four different nodes 
                        elif j == (upper_bound-3):
            #ok                
                            Q_node = 0

                        
                            for k in range(4):
                                
                                if  k >= 2:
                                    Qtemp = self.probability[-k-1][i-1][-1+k]*np.exp(-np.exp(self.X[-k-1][i-1])*self.dt)*self.Q[-k-1][i-1]
                                    #print(f"p {i,j}: {probability[-k-1][i-1][-1+k]}, X: {X[-k-1][i-1]}, Q: {Q[-k-1][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[-k-1][i-1][0]*np.exp(-np.exp(self.X[-k-1][i-1])*self.dt)*self.Q[-k-1][i-1]
                                    #print(f"p {i,j}: {probability[-k-1][i-1][0]}, X: {X[-k-1][i-1]}, Q: {Q[-k-1][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                        
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)    
                        
                        
                        #fourth node    
                        elif j == (upper_bound-2):
            #ok                
                            Q_node = 0
                        
                            for k in range(3):
                                
                                if k == 0:
                                    Qtemp = self.probability[-3][i-1][2]*np.exp(-np.exp(self.X[-3][i-1])*self.dt)*self.Q[-3][i-1]
                                    #print(f"p {i,j}: {probability[-3][i-1][2]}, X: {X[-3][i-1]}, Q: {Q[-3][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                else:
                                    Qtemp = self.probability[-3+k][i-1][1]*np.exp(-np.exp(self.X[-3+k][i-1])*self.dt)*self.Q[-3+k][i-1]
                                    #print(f"p {i,j}: {probability[-3+k][i-1][1]}, X: {X[-3+k][i-1]}, Q: {Q[-3+k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)
                                
                        #lower limit node, it can be reached by 2 other nodes
                        elif j == (upper_bound-1):
            #ok               
                            Q_node = 0
                        
                            for k in range(2):
                                
                                Qtemp = self.probability[-2+k][i-1][2]*np.exp(-np.exp(self.X[-2+k][i-1])*self.dt)*self.Q[-2+k][i-1]
                                #print(f"p {i,j}: {probability[-2+k][i-1][2]}, X: {X[-2+k][i-1]}, Q: {Q[-2+k][i-1]}, R*: {R_star[j][i]}")
                                Q_node += Qtemp
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)
                        
                        else:
                            
                            Q_node = 0   
            #ok            
                            for k in range(3):
                            
                                Qtemp = self.probability[j+k-1][i-1][2-k]*np.exp(-np.exp(self.X[j+k-1][i-1])*self.dt)*self.Q[j+k-1][i-1]
                                #print(f"p {i,j}: {probability[j+k-1][i-1][2-k]}, X: {X[j+k-1][i-1]}, Q: {Q[j+k-1][i-1]}, R*: {R_star[j][i]}")
                                Q_node += Qtemp
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)
                            
                    
#Match the term structure-----------------------------------------------------------------------------------------------------------------------------------------           
            

    
            
            def func(a):
                total = 0
                for k in range(0,upper_bound):
                    s = self.R_star[k][i]  # Get the volatility shift for the current time step
                    bond = self.zcb[i]  # Get the ZCB rate for the current time step
                    
                    
                    # Compute each term based on the current time step's parameters
                    term = Q_sum[k] * np.exp(-np.exp(a + s) * self.dt)
                    total += term
                return total - bond 
            
                
            
                
            if self.type == "newton":
                
                for j in range(0, upper_bound):
                    
                    #print(f"Q: {Q_sum[j]}, R*: {R_star[j][i]}")
                    self.X[j][i] = newton(func,-3) + self.R_star[j][i]
                    self.R[j][i] = np.exp(self.X[j][i])
                    
                    if self.R_star[j][i] == 0:
                        self.alpha[i] = np.exp(self.X[j][i])
                        
            elif self.type == "brentq":
                
                for j in range(0, upper_bound):
                    
                    #print(f"Q: {Q_sum[j]}, R*: {R_star[j][i]}")
                    self.X[j][i] = brentq(func,-10,10) + self.R_star[j][i]
                    self.R[j][i] = np.exp(self.X[j][i])
                
                    if self.R_star[j][i] == 0:
                        self.alpha[i] = np.exp(self.X[j][i])
                
        
        return self.R            


    #compute bond price
    def bond_price(self, face_value, coupon):
        
        """This method is used to compute the tree of the bond
            Args.
            face_value: face value of the bond.
            coupon: coupon value in decimals.

        Returns:
            _type_: numpy array
        """
        
        self.face_value = face_value                                            #face value of the bond
        self.coupon = coupon                                                    #coupon
        self.last_cashflow = self.face_value + (self.face_value*self.coupon)    #cashflow for the last period
    
        #recall the rate matrix and the probabilities, generate bond matrix
        self.R = self.rate_tree()   
        self.probability = self.probability_computation()
        self.bond = np.zeros_like(self.R)                                       #bond matrix
        
        
        #fill the last column
        for i in range(self.R.shape[0]):
            
            self.bond[i][-1] = self.last_cashflow/(1+self.R[i][-1])
            

        for i in range(self.R.shape[1]-2, -1, -1):
            
            
            
            if i >= self.j_max:
                
                
                
                for j in range(self.R.shape[0]):
                        
                    
                    
                    if j == 0:
                        
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {bond[j+k][i+1]}")
                            price_sum += self.probability[j][i][k]*self.bond[j+k][i+1]
                            
                        price_sum += (coupon*face_value)
                        price_sum = price_sum/(1+self.R[j][i])
                        
                        self.bond[j][i] = price_sum
                        #print(f"\n")     
                            
                    
                    elif j == self.R.shape[0]-1:
                        
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {bond[j+k-2][i+1]}")
                            price_sum += self.probability[j][i][k]*self.bond[j+k-2][i+1]
                            
                        price_sum += (coupon*face_value)
                        price_sum = price_sum/(1+self.R[j][i])
                        
                        self.bond[j][i] = price_sum
                        #print(f"\n") 
                        
                    else:
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {bond[j+k-1][i+1]}")
                            price_sum += self.probability[j][i][k]*self.bond[j+k-1][i+1]
                            
                        price_sum += (coupon*face_value)
                        price_sum = price_sum/(1+self.R[j][i])
                        
                        self.bond[j][i] = price_sum
                        #print(f"\n") 
                        
                        
                        
                    
            else:
                
                
                for j in range(self.R.shape[0]):
                    
                    if self.R[j][i] == 0:
                        continue
                        
                    else:
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {bond[j+k][i+1]}")
                            price_sum += self.probability[j][i][k]*self.bond[j+k][i+1]
                            
                        price_sum += (coupon*face_value)
                        price_sum = price_sum/(1+self.R[j][i])
                        
                        self.bond[j][i] = price_sum
                        
                        #print(f"\n")  
                        
        return self.bond         



    #compute bond price
    def embedded_option_bond(self, face_value, coupon,opt_type, value,conversion_period = 1):
        
        """This method is used to compute the tree of the bond
            Args.
            face_value: face value of the bond.
            coupon: coupon value in decimals.
            opt_type: string C or P for callable or putable bond
            value: 
            conversion_period: default 1, otherwise 1,2,3,4..
        Returns:
            _type_: numpy array
            print: value of the call or put option
        """
        
        self.face_value = face_value                                            #face value of the bond
        self.coupon = coupon                                                    #coupon
        self.last_cashflow = self.face_value + (self.face_value*self.coupon)    #cashflow for the last period
        self.opt_type = opt_type                                                # callable or putable type
        self.value = value                                                      #callling price
        self.option = 0
        self.conversion_period = conversion_period
    
        #recall the rate matrix and the probabilities, generate bond matrix
        self.R = self.rate_tree()   
        self.probability = self.probability_computation()
        self.bond = np.zeros_like(self.R)                                       #bond matrix
        
        #value of the bond at node 0:
        normal_bond = self.bond_price(face_value,coupon)
        normal_bond = normal_bond[0][0]
        
        
        
        
        try:
            if self.opt_type == "C":
        #fill the last column
                for i in range(self.R.shape[0]):
                    
                    self.bond[i][-1] = min(self.last_cashflow/(1+self.R[i][-1]), self.value)
                    

                for i in range(self.R.shape[1]-2, -1, -1):
                
             
                    if i >= self.j_max:
                        
                            
                        for j in range(self.R.shape[0]):
                                
                                    
                            if j == 0:
                                
                                
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},R: {bond[j+k][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                #check if the bond can be converted
                                if i >= self.conversion_period:
                                
                                    self.bond[j][i] = min(price_sum,self.value)
                                
                                else:
                                    self.bond[j][i] = price_sum
                                    
                            
                            elif j == self.R.shape[0]-1:
                                
                                
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},R: {bond[j+k-2][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k-2][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                
                                    self.bond[j][i] = min(price_sum,self.value)
                                
                                else:
                                    self.bond[j][i] = price_sum
                                
                            else:
                                
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},R: {bond[j+k-1][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k-1][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                
                                    self.bond[j][i] = min(price_sum,self.value)
                                
                                else:
                                    self.bond[j][i] = price_sum
                                         
                    else:
                        
                        
                        for j in range(self.R.shape[0]):
                            
                            if self.R[j][i] == 0:
                                continue
                                
                            else:
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},R: {bond[j+k][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                
                                    self.bond[j][i] = min(price_sum,self.value)
                                
                                else:
                                    self.bond[j][i] = price_sum  
                                
                self.option = normal_bond - self.bond[0][0]     
                
            elif self.opt_type == "P":
                  
            #fill the last column
                for i in range(self.R.shape[0]):
                        
                    self.bond[i][-1] = max(self.last_cashflow/(1+self.R[i][-1]), self.value)
                        
                    
                for i in range(self.R.shape[1]-2, -1, -1): 
                    
                    if i >= self.j_max:
                        
                        
                        
                        for j in range(self.R.shape[0]):
                                
                            
                            
                            if j == 0:
                                
                                
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},R: {bond[j+k][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                    
                                    self.bond[j][i] = max(price_sum,self.value)
                                    
                                else:  
                                    self.bond[j][i] = price_sum
                                    
                            
                            elif j == self.R.shape[0]-1:
                                
                                
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},R: {bond[j+k-2][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k-2][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                    
                                    self.bond[j][i] = max(price_sum,self.value)
                                    
                                else:  
                                    self.bond[j][i] = price_sum
                                
                            else:
                                
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},R: {bond[j+k-1][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k-1][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                    
                                    self.bond[j][i] = max(price_sum,self.value)
                                    
                                else: 
                                     
                                    self.bond[j][i] = price_sum 
                                            
                    else:
                        
                        
                        for j in range(self.R.shape[0]):
                            
                            if self.R[j][i] == 0:
                                continue
                                
                            else:
                                price_sum = 0
                                
                                for k in range(3):
                                    
                                    #print(f"prob: {probability[j][i][k]},R: {bond[j+k][i+1]}")
                                    price_sum += self.probability[j][i][k]*self.bond[j+k][i+1]
                                    
                                price_sum += (coupon*face_value)
                                price_sum = price_sum/(1+self.R[j][i])
                                
                                if i >= self.conversion_period:
                                    
                                    self.bond[j][i] = max(price_sum,self.value)
                                    
                                else: 
                                     
                                    self.bond[j][i] = price_sum  
                
                self.option = self.bond[0][0] - normal_bond

            else:  
                print(f"Option type must be C for the Callable or P for the Putable")
        
        except ValueError as e:
            print(f"ValueError: {e}")

        except Exception as e:
            
            print(f"An unexpected error occurred: {e}")
             
        
        
        print(f"The Option value is {round(self.option,4)}")                  
        return self.bond


    #compute interest rate swap
    def swap_tree(self, principal, pay):
        
        """This method is used to compute the value of a Interest Rate Swap.
            Args.
            principal: amount of the principal.
            pay: payment frequency 

        Returns:
            _type_: numpy Array
        """
        
        self.R = self.rate_tree()
        self.probability = self.probability_computation()
        
        self.discount = []
        self.principal = principal
        self.pay = pay
        
        #compute the discount factors
        for i in range(self.n+1):
        
            disc = np.exp(-(self.yield_curve[i][1])*self.yield_curve[i][0])
            self.discount.append(disc)
            
        
        #compute the swap rate
        self.swap_rate = self.pay * ((1-self.discount[-1])/(np.sum(self.discount)))
        
        self.cf = np.zeros_like(self.R) #MATRIX FOR CF
        self.swap = np.zeros_like(self.R) #matrix for swap   
        
        
        self.cf[0][0] = self.principal * self.dt*(self.pay*(np.exp(self.R[0][0]* self.dt)-1)-self.swap_rate)
        
        
        #fill the matrix
        for i in range(1, self.R.shape[1]):
            
            
            for j in range(0, self.R.shape[0]):
                
                if self.R[j][i] == 0:
                    continue
                else:
                    
                    self.cf[j][i] = self.principal*self.dt*(self.pay*(np.exp(self.R[j][i]*self.dt)-1)-self.swap_rate)


        #-----------------------------------------------------------------------------------------------------------------
        #COMPUTE THE SWAP VALUE
        
        #fill the last column
        for i in range(self.R.shape[0]):
            
            self.swap[i][-1] = self.cf[i][-1]/(1+self.R[i][-1]*self.dt)
            

        for i in range(self.R.shape[1]-2, -1, -1):
            
            
            if i >= self.j_max:
                
                
                
                for j in range(self.R.shape[0]):
                        
                    
                    
                    if j == 0:
                        
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},swap: {swap[j+k][i+1]}")
                            price_sum += self.probability[j][i][k]*self.swap[j+k][i+1]
                            
                        price_sum += self.cf[j][i]
                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        self.swap[j][i] = price_sum
                        #print(f"\n")     
                            
                    
                    elif j == self.R.shape[0]-1:
                        
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {swap[j+k-2][i+1]}")
                            price_sum += self.probability[j][i][k]*self.swap[j+k-2][i+1]
                            
                        price_sum += self.cf[j][i]
                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        self.swap[j][i] = price_sum
                        #print(f"\n") 
                        
                    else:
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {swap[j+k-1][i+1]}")
                            price_sum += self.probability[j][i][k]*self.swap[j+k-1][i+1]
                            
                        price_sum += self.cf[j][i]
                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        self.swap[j][i] = price_sum
                        #print(f"\n") 
                        
                        
                        
                    
            else:
                
                
                for j in range(self.R.shape[0]):
                    
                    if self.R[j][i] == 0:
                        continue
                        
                    else:
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {swap[j+k][i+1]}")
                            price_sum += self.probability[j][i][k]*self.swap[j+k][i+1]
                            
                        price_sum += self.cf[j][i]
                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        self.swap[j][i] = price_sum
                        #print(f"\n") 
                        

                
        #print(np.round(self.cf,5))
        #print('\n')
        #print(np.round(self.swap,3))
        return self.swap
    
    
    #compute swaption
    def swaption(self,principal,pay, exercise=None):
        
        """This method is used to compute the value of a Interest Rate Swap.
            Args.
            principal: amount of the principal.
            pay: payment frequency.
            exercise: python list --> used to specifit American,European, Bermudian style.
            American is the default, insert the last time step for the European, specify the time step for the Bermudian.
            
        Returns:
            _type_: numpy Array
        """
        
        self.R = self.rate_tree()
        self.probability = self.probability_computation()
        self.S = self.swap_tree(principal,pay)
        self.exercise = exercise
        self.C = np.zeros_like(self.S)
        
        
        #-----------------------------------------------------------------------------------------------------------------
        #COMPUTE THE SWAPTION VALUE
        
        #fill the last column
        for i in range(self.R.shape[0]):
            
            self.C[i][-1] = max(0, self.S[i][-1])
            

        for i in range(self.R.shape[1]-2, -1, -1):
            
            if i == 0:
                
                for j in range(self.R.shape[0]):
                    
                    if self.R[j][i] == 0:
                        continue
                        
                    else:
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"node: {j,i}prob: {self.probability[j][i][k]},swap: {self.swap[j+k][i+1]}, S: {self.C[j+k][i+1]}")
                            price_sum += self.probability[j][i][k]*self.C[j+k][i+1]

                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        self.C[j][i] = price_sum
                        
            
        
            
            elif i >= self.j_max:
                
                   
                for j in range(self.R.shape[0]):
                        
                         
                    if j == 0:
                        
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"node: {j,i}, prob: {self.probability[j][i][k]},swap: {self.S[j+k][i+1]}, S: {self.C[j+k][i+1]}")
                            price_sum += self.probability[j][i][k]*self.C[j+k][i+1]
                            
                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        if self.exercise == None or i in self.exercise:
                            
                            self.C[j][i] = max(price_sum, 0)
                        else:
                            self.C[j][i] = price_sum
                            
                        #print(f"\n")     
                            
                    
                    elif j == self.R.shape[0]-1:
                        
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {swap[j+k-2][i+1]}")
                            price_sum += self.probability[j][i][k]*self.C[j+k-2][i+1]
                            
                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        if self.exercise == None or i in self.exercise:
                            self.C[j][i] = max(price_sum, 0)
                        else:
                            self.C[j][i] = price_sum
                        #print(f"\n") 
                        
                    else:
                        
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"prob: {probability[j][i][k]},R: {swap[j+k-1][i+1]}")
                            price_sum += self.probability[j][i][k]*self.C[j+k-1][i+1]

                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        if self.exercise == None or i in self.exercise:
                            self.C[j][i] = max(price_sum, 0)
                        else:
                            self.C[j][i] = price_sum
                        
                                      
                    
            else:
                
                
                for j in range(self.R.shape[0]):
                    
                    if self.R[j][i] == 0:
                        continue
                        
                    else:
                        price_sum = 0
                        
                        for k in range(3):
                            
                            #print(f"node: {j,i}, prob: {self.probability[j][i][k]},swap: {self.S[j+k][i+1]}, C: {self.C[j+k][i+1]} ")
                            price_sum += self.probability[j][i][k]*self.C[j+k][i+1]

                        price_sum = price_sum*np.exp(-self.R[j][i]*self.dt)
                        
                        if self.exercise == None or i in self.exercise:
                            self.C[j][i] = max(price_sum, 0)
                        else:
                            self.C[j][i] = price_sum
                        #print(f"\n") 
                        

        return self.C
    
    
    #print rate tree
    def print_tree(self, method: str = "newton"):
        
        """Optimization type must be of type string 'newton' or 'brentq' """
        
        if method not in ['newton', 'brentq']:
            raise ValueError("optimization_type must be 'newton' or 'brentq' ")
        
        self.method = method
        
        
        if self.method == "newton":
            rate = self.rate_tree()*100
        else:
            rate = self.rate_tree("brentq")*100
        
        
        rows, cols = rate.shape
        middle_row = rows // 2
        centered_matrix = np.zeros_like(rate)
        
        for col in range(cols):
            # Extract non-zero elements in column
            non_zeros = rate[:, col][rate[:, col] != 0]
            start_row = middle_row - len(non_zeros) // 2
            
            # Place elements starting from the calculated start row
            for i, value in enumerate(non_zeros):
                centered_matrix[start_row + i, col] = value
        
        # Apply the centering function to the new array R

        df = pd.DataFrame(centered_matrix)
        df.replace(0, '-',inplace=True)
        
        return df
    
    
    #print the bond matrix
    def print_bond(self, face_value, coupon):
        
        """This method is used to print the tree of the bond
            Args.
            face_value: face value of the bond.
            coupon: coupon value in decimals.

        Returns:
            _type_: pandas DataFrame
        """
        
        bond = self.bond_price(face_value, coupon)
        
        rows, cols = bond.shape
        middle_row = rows // 2
        centered_matrix = np.zeros_like(bond)
        
        for col in range(cols):
            # Extract non-zero elements in column
            non_zeros = bond[:, col][bond[:, col] != 0]
            start_row = middle_row - len(non_zeros) // 2
            
            # Place elements starting from the calculated start row
            for i, value in enumerate(non_zeros):
                centered_matrix[start_row + i, col] = value
        
        # Apply the centering function to the new array R

        df = pd.DataFrame(centered_matrix)
        df.replace(0, '-',inplace=True)
        
        return df
    
    
    #method to print the bond matrix
    def print_embedded_option_bond(self, face_value, coupon, opt_type, value, conversion_period=1):
        
        """This method is used to print the tree of the bond
            Args.
            face_value: face value of the bond.
            coupon: coupon value in decimals.
            opt_type: string C or P for callable or putable.
            value:.
            conversion_value: default 1, otherwise indicates the period (i.e. 1,2,3,4)

        Returns:
            _type_: pandas DataFrame
        """
        
        bond = self.embedded_option_bond(face_value, coupon, opt_type, value,conversion_period)
        
        rows, cols = bond.shape
        middle_row = rows // 2
        centered_matrix = np.zeros_like(bond)
        
        for col in range(cols):
            # Extract non-zero elements in column
            non_zeros = bond[:, col][bond[:, col] != 0]
            start_row = middle_row - len(non_zeros) // 2
            
            # Place elements starting from the calculated start row
            for i, value in enumerate(non_zeros):
                centered_matrix[start_row + i, col] = value
        
        # Apply the centering function to the new array R

        df = pd.DataFrame(centered_matrix)
        df.replace(0, '-',inplace=True)
        
        return df
    
    #get alpha
    def get_alpha(self):
        
        self.rate_tree()
        
        return self.alpha
     
    #print interest rate swap tree   
    def print_swap(self, principal, pay):
        
        """This method is used to print the value of a Interest Rate Swap.
            Args.
            principal: amount of the principal.
            pay: payment frequency 

        Returns:
            _type_: Pandas DataFrame
        """
            
        S = np.round(self.swap_tree(principal, pay),2) 
        
        rows, cols = S.shape
        middle_row = rows // 2
        centered_matrix = np.zeros_like(S)
        
        for col in range(cols):
            # Extract non-zero elements in column
            non_zeros = S[:, col][S[:, col] != 0]
            start_row = middle_row - len(non_zeros) // 2
            
            # Place elements starting from the calculated start row
            for i, value in enumerate(non_zeros):
                centered_matrix[start_row + i, col] = value
        
        # Apply the centering function to the new array R

        df = pd.DataFrame(centered_matrix)
        
        # Replace 0 with '-' except for the value in the middle row of the first column
        for i in range(rows):
            for j in range(cols):
                if centered_matrix[i, j] == 0:
                    if not (i == middle_row and j == 0):
                        df.iloc[i, j] = '-'
        
        return df
   

    
if __name__=="__main__":
    
    #yield_curve = [(0.5,0.03430),(1, 0.03824),(1.5, 0.04183),(2, 0.04512), (2.5, 0.04812)]
    yield_curve = [(0.5*i, 0.04+0.005*i) for i in range(1, 10)]
    
    discount = []
    for i in range(len(yield_curve)):
        
        disc = np.exp(-(yield_curve[i][1])*yield_curve[i][0])
        discount.append(disc)
    #print(discount)
    
    opt = black_Karasinski(0.15, 0.15, 2,4, yield_curve)
    print(opt.rate_tree())
    #print(yield_curve)
    print(np.round(opt.swap_tree(100, 2),2))
    print(np.round(opt.swaption(100, 2),2))
    
    

    
    
    
    
    
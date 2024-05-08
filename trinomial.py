import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.optimize import newton, brentq
from scipy.interpolate import CubicSpline
import random
plt.style.use('seaborn-v0_8-dark')
#final

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



class hull_white:   
    
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
    
    
            upper_bound = min(3 + (i-1)*2  , self.j_max*3-(self.j_max-1))  
            
            
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
            #print(upper_bound)
            Q_sum = 0
            
            for j in range(0, upper_bound):
                
                #print(f"i: {i}, j: {j}")    
                
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
                        Q_sum += Q_node*np.exp(-self.R_star[j][i])  #check the discounting R_star
                            
                
                    
                    elif j == upper_bound - 2:  #if size = 5, j = 3, if size = 7, j = 5
                        
                        Q_node = 0
                        
                        for k in range(2):
                            
                            Qtemp = self.probability[j-k-1][i-1][1+k]*np.exp(-self.R[j-k-1][i-1]*self.dt)*self.Q[j-k-1][i-1]
                            #print(f"p {i,j}: {probability[j-k-1][i-1][1+k]}, R: {R[j-k-1][i-1]}, Q: {Q[j-k-1][i-1]}, R*: {R_star[j][i]}")
                            Q_node += Qtemp
                            
                        #print(f"\n")   
                        self.Q[j][i] = Q_node
                        Q_sum += Q_node*np.exp(-self.R_star[j][i])  #check the discounting R_star
                    
                
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
                        Q_sum += Q_node*np.exp(-self.R_star[j][i])  #check the discounting R_star
        
            
                
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
        #print(f"{np.round(self.R,4)*100}\n")
        #print(np.round((self.Q),4))  
    
   
    #print rate tree
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
    
    
    # get alpha value
    def get_alpha(self):
        
        self.rate_tree()
        
        return self.alpha
    
     

class black_Karasinski:   
    
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
        
        self.a = a                                                     #mean reversion
        self.sigma = sigma                                             #volatility
        self.N = N                                                     #maturity
        self.n = n                                                     #step
        self.dt = N/n                                                  #time step
        self.yield_curve = yield_curve


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
    
    
            upper_bound = min(3 + (i-1)*2  , self.j_max*3-(self.j_max-1))  
            
            
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
            #print(upper_bound)
            Q_sum = []
            
            
            for j in range(0, upper_bound):
                
                #print(f"i: {i}, j: {j}")    
                
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
                            #print(f"p {i,j}: {probability[j-k-1][i-1][1+k]}, R: {R[j-k-1][i-1]}, Q: {Q[j-k-1][i-1]}, R*: {R_star[j][i]}")
                            Q_node += Qtemp
                        
                        #print(f"Q: {Q_node}")   
                        #print(f"\n")   
                        self.Q[j][i] = Q_node
                        Q_sum.append(Q_node)  #check the discounting R_star
                    
                
                    elif j == upper_bound - 1:
                        
                        
                        Qtemp = self.probability[j-2][i-1][2]*np.exp(-np.exp(self.X[j-2][i-1])*self.dt)*self.Q[j-2][i-1]
                        #print(f"p {i,j}: {probability[j-2][i-1][2]}, R: {R[j-2][i-1]}, Q: {Q[j-2][i-1]}, R*: {R_star[j][i]}")
                        #print(f"\n")
                        #print(f"Q: {Qtemp}")
                        self.Q[j][i] = Qtemp
                        Q_sum.append(Qtemp) 
                        
                    else:
                        
                        Q_node = 0
                        
                        for k in range(3):
                        
                            Qtemp = self.probability[j+k-2][i-1][2-k]*np.exp(-np.exp(self.X[j+k-2][i-1])*self.dt)*self.Q[j+k-2][i-1]
                            #print(f"p {i,j}: {probability[j+k-2][i-1][2-k]}, R: {R[j+k-2][i-1]}, Q: {Q[j+k-2][i-1]}, R*: {R_star[j][i]}")
                            Q_node += Qtemp
                            
                        #print(f"Q: {Q_node}")   
                        #print(f"\n")   
                        self.Q[j][i] = Q_node
                        Q_sum.append(Q_node) #check the discounting R_star
        
            
                
        #Limit reached-------------------------------------------------------------------------------------------------------------------------------------------

                else:
                
        #check if it is the third step, after that all the same-----------------------------------------------------------------------------------------------------   
                    if i == 3 or self.R.shape[0] == 5:
                        
                        
                        if j == 0:
                            
                            Q_node = 0
                        
                            for k in range(2):
                                    
                                Qtemp = self.probability[k][i-1][0]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                #print(f"p {i,j}: {probability[k][i-1][0]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")

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
                                    #print(f"p {i,j}: {probability[k][i-1][1]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[k][i-1][0]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][0]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
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
                                    #print(f"p {i,j}: {probability[k][i-1][2]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                elif k == 2:
                                    Qtemp = self.probability[2][i-1][1]*np.exp(-np.exp(self.X[2][i-1])*self.dt)*self.Q[2][i-1]
                                    #print(f"p {i,j}: {probability[2][i-1][1]}, R: {R[2][i-1]}, Q: {Q[2][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[k][i-1][0]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][0]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
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
                                    #print(f"p {i,j}: {probability[j+k-1][i-1][2]}, R: {R[j+k-1][i-1]}, Q: {Q[j+k-1][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                else:
                                    Qtemp = self.probability[j+k-1][i-1][1]*np.exp(-np.exp(self.X[j+k-1][i-1])*self.dt)*self.Q[j+k-1][i-1]
                                    #print(f"p {i,j}: {probability[j+k-1][i-1][1]}, R: {R[j+k-1][i-1]}, Q: {Q[j+k-1][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)
                                
                        #fifth node
                        elif j == 4:
                        
                            Q_node = 0
                        
                            for k in range(2):
                                
                                Qtemp = self.probability[4-k][i-1][2]*np.exp(-np.exp(self.X[4-k][i-1])*self.dt)*self.Q[4-k][i-1]
                                #print(f"p {i,j}: {probability[4-k][i-1][2]}, R: {R[4-k][i-1]}, Q: {Q[4-k][i-1]}, R*: {R_star[j][i]}")
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
                                #print(f"p {i,j}: {probability[k][i-1][0]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")

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
                                    #print(f"p {i,j}: {probability[k][i-1][1]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[k][i-1][0]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][0]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
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
                                    #print(f"p {i,j}: {probability[k][i-1][2]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[k][i-1][3-k]*np.exp(-np.exp(self.X[k][i-1])*self.dt)*self.Q[k][i-1]
                                    #print(f"p {i,j}: {probability[k][i-1][3-k]}, R: {R[k][i-1]}, Q: {Q[k][i-1]}, R*: {R_star[j][i]}")
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
                                    #print(f"p {i,j}: {probability[-k-1][i-1][-1+k]}, R: {R[-k-1][i-1]}, Q: {Q[-k-1][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                    
                                else:
                                    Qtemp = self.probability[-k-1][i-1][0]*np.exp(-np.exp(self.X[-k-1][i-1])*self.dt)*self.Q[-k-1][i-1]
                                    #print(f"p {i,j}: {probability[-k-1][i-1][0]}, R: {R[-k-1][i-1]}, Q: {Q[-k-1][i-1]}, R*: {R_star[j][i]}")
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
                                    #print(f"p {i,j}: {probability[-3][i-1][2]}, R: {R[-3][i-1]}, Q: {Q[-3][i-1]}, R*: {R_star[j][i]}")
                                    Q_node += Qtemp
                                else:
                                    Qtemp = self.probability[-3+k][i-1][1]*np.exp(-np.exp(self.X[-3+k][i-1])*self.dt)*self.Q[-3+k][i-1]
                                    #print(f"p {i,j}: {probability[-3+k][i-1][1]}, R: {R[-3+k][i-1]}, Q: {Q[-3+k][i-1]}, R*: {R_star[j][i]}")
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
                                #print(f"p {i,j}: {probability[-2+k][i-1][2]}, R: {R[-2+k][i-1]}, Q: {Q[-2+k][i-1]}, R*: {R_star[j][i]}")
                                Q_node += Qtemp
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)
                        
                        else:
                            
                            Q_node = 0   
            #ok            
                            for k in range(3):
                            
                                Qtemp = self.probability[j+k-1][i-1][2-k]*np.exp(-np.exp(self.X[j+k-1][i-1])*self.dt)*self.Q[j+k-1][i-1]
                                #print(f"p {i,j}: {probability[j+k-1][i-1][2-k]}, R: {R[j+k-1][i-1]}, Q: {Q[j+k-1][i-1]}, R*: {R_star[j][i]}")
                                Q_node += Qtemp
                                
                            #print(f"\n")   
                            self.Q[j][i] = Q_node
                            Q_sum.append(Q_node)
                            
                        
                    
        #Match the term structure-----------------------------------------------------------------------------------------------------------------------------------------           

                                        

                    
        #Match the term structure-----------------------------------------------------------------------------------------------------------------------------------------           


                        
                    
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
    
    def get_alpha(self):
        
        self.rate_tree()
        
        return self.alpha
        
        
        
          
    
if __name__=="__main__":
    
    yield_curve = [(0.5,0.03430),(1, 0.03824),(1.5, 0.04183),(2, 0.04512), (2.5, 0.04812), (3, 0.05086),(3.5,0.055),(4,0.06),(4.5,0.07)]
    
    opt = hull_white(0.08, 0.01, 4,8, yield_curve)
    
    print(np.round(opt.rate_tree()*100,3))
    
    
    
    
    
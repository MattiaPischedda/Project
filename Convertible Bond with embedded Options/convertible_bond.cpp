#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <chrono>

using namespace std;

//define option type
enum class OptionType { CALL, PUT };


class convertible {
private:
    int N;
    int T;
    double R0;
    double dt;
    double ur = 1.1;
    double dr = 0.95;
    double q = 0.5;

public:

    // Constructor
    convertible(int T, int N, double R0) : N(N), T(T), R0(R0) {
        dt = static_cast<double>(T) / N;
    }


    //callable tree
    vector<vector<double>> callable(double faceValue, double coupon, double cp) {
        // Create the vectors
        vector<vector<double>> R(N + 1, vector<double>(N + 1, 0.0));
        vector<vector<double>> B(N + 1, vector<double>(N + 1, 0.0));
        vector<vector<double>> C(N + 1, vector<double>(N + 1, 0.0));

        // Initialize the rate matrix
        R[0][0] = R0;

        // Compute the rate tree
        for (int i = 1; i <= N; i++) {
            for (int j = 0; j <= i; j++) {
                R[j][i] = R0 * pow(ur, i - j) * pow(dr, j);
            }
        }

        //compute the final cashflow
        double finalCF = faceValue + (faceValue * coupon);


        //compute the last step
        for (int i = 0; i < N+1; i++){

            C[i][N] = min(cp, finalCF / (1+ R[i][N]));
            B[i][N] = finalCF / (1 + R[i][N]);
        }
        

        //compute the tree for the bond and for the callable bond

        for (int i = N-1; i > -1; i--){

            for (int j = 0; j < i+1; j++){
                
                C[j][i] = min(cp, ((C[j][i+1]*q + (1-q)*C[j+1][i+1] + (coupon*faceValue)) / (1+R[j][i])));
                B[j][i] = (B[j][i+1]*q + (1-q)*B[j+1][i+1] + (coupon*faceValue))/(1+R[j][i]);

            }
        }
        
         
        return C;
    }


    //callable tree
    vector<vector<double>> putable(double faceValue, double coupon, double pp) {
        // Create the vectors
        vector<vector<double>> R(N + 1, vector<double>(N + 1, 0.0));
        vector<vector<double>> B(N + 1, vector<double>(N + 1, 0.0));
        vector<vector<double>> C(N + 1, vector<double>(N + 1, 0.0));

        // Initialize the rate matrix
        R[0][0] = R0;

        // Compute the rate tree
        for (int i = 1; i <= N; i++) {
            for (int j = 0; j <= i; j++) {
                R[j][i] = R0 * pow(ur, i - j) * pow(dr, j);
            }
        }

        //compute the final cashflow
        double finalCF = faceValue + (faceValue * coupon);


        //compute the last step
        for (int i = 0; i < N+1; i++){

            C[i][N] = max(pp, finalCF / (1+ R[i][N]));
            B[i][N] = finalCF / (1 + R[i][N]);
        }
        

        //compute the tree for the bond and for the callable bond

        for (int i = N-1; i > -1; i--){

            for (int j = 0; j < i+1; j++){
                
                C[j][i] = max(pp, ((C[j][i+1]*q + (1-q)*C[j+1][i+1] + (coupon*faceValue)) / (1+R[j][i])));
                B[j][i] = (B[j][i+1]*q + (1-q)*B[j+1][i+1] + (coupon*faceValue))/(1+R[j][i]);


            }
        }
        
         
        return C;
    }


    //convertible bond price with embedded option
    vector<vector<double>> convertibleBond(double sigma, double S0, double faceValue, double coupon, double conversionRatio, double optPrice, OptionType optionType, int lockPeriod){


        //up down factors
        double us = exp(sigma * sqrt(dt));
        double ds = 1/us;

        //increase the size of the matrix
        N++;

        // Create the vectors
        vector<vector<double>> R(N + 1, vector<double>(N + 1, 0.0));
        vector<vector<double>> S(N + 1, vector<double>(N + 1, 0.0));
        vector<vector<double>> B(N + 1, vector<double>(N + 1, 0.0));


        //initialize the Stock and Rate matrix
        S[0][0] = S0;
        R[0][0] = R0;

        //tree for R and S
        for (int i = 1; i < N+1; i++){
            for (int j = 0; j < i+1; j++){

                S[j][i] = S0* pow(us,(i-j)) * pow(ds, (j));
                R[j][i] = 0.16 - 0.001*S[j][i];
            }
        }
        

        // Handle invalid option type
        if (optionType == OptionType::CALL) {

            //initialize last step 
            for (int i = 0; i < N+1; i++){
           
                B[i][N] = max(conversionRatio*S[i][N], faceValue) + (coupon * faceValue);
            }   
            
            for (int i = N-1; i > -1; i--){
            
                for (int j = 0; j < i+1; j++){
                
                    
                    if (i == N-1){  //Penultimate step
                        
                        if  (i >= lockPeriod){
                            
                            B[j][i] = max(S[j][i] * conversionRatio, min(optPrice, (B[j][i+1] * q + (1-q) * B[j+1][i+1]) / (1 + R[j][i])));
                        }    
                        else{
                            
                            B[j][i] = max(S[j][i]* conversionRatio , (B[j][i+1] * q + (1-q) * B[j+1][i+1]) / (1 + R[j][i]));
                        } 
                    }

                    else if (i == 0){

                        B[j][i] =(B[j][i+1] * q + (1-q) * B[j+1][i+1] + (coupon * faceValue)) / (1 + R[j][i]);
                    }

                    else {

                        if (i >= lockPeriod){

                            B[j][i] = max(S[j][i] * conversionRatio, min(optPrice, (B[j][i+1] * q + (1-q) * B[j+1][i+1] + (coupon * faceValue)) / (1 + R[j][i])));

                        }
                        else{

                            B[j][i] = max(S[j][i]*conversionRatio , (B[j][i+1] * q + (1-q) * B[j+1][i+1] + (coupon * faceValue)) / (1 + R[j][i]));

                        }
                    }
                }    
            }
        }

        else if (optionType == OptionType::PUT) {

            //initialize last step 
            for (int i = 0; i < N+1; i++){
           
                B[i][N] = max(conversionRatio*S[i][N], faceValue) + (coupon * faceValue);
            }   
            
            for (int i = N-1; i > -1; i--){
            
                for (int j = 0; j < i+1; j++){
                
                    
                    if (i == N-1){  //Penultimate step
                        
                        if  (i >= lockPeriod){
                            
                            B[j][i] = max(S[j][i] * conversionRatio, max(optPrice, (B[j][i+1] * q + (1-q) * B[j+1][i+1]) / (1 + R[j][i])));
                        }    
                        else{
                            
                            B[j][i] = max(S[j][i]* conversionRatio , (B[j][i+1] * q + (1-q) * B[j+1][i+1]) / (1 + R[j][i]));
                        } 
                    }

                    else if (i == 0){

                        B[j][i] =(B[j][i+1] * q + (1-q) * B[j+1][i+1] + (coupon * faceValue)) / (1 + R[j][i]);
                    }

                    else {

                        if (i >= lockPeriod){

                            B[j][i] = max(S[j][i] * conversionRatio, max(optPrice, (B[j][i+1] * q + (1-q) * B[j+1][i+1] + (coupon * faceValue)) / (1 + R[j][i])));

                        }
                        else{

                            B[j][i] = max(S[j][i]*conversionRatio , (B[j][i+1] * q + (1-q) * B[j+1][i+1] + (coupon * faceValue)) / (1 + R[j][i]));

                        }
                    }
                }    
            }   
        } 

        else {

            throw invalid_argument("Invalid option type. Must be CALL or PUT.");
        }

        cout << "The bond price is: " << B[0][0] << endl;
        return B;

    };

};


int main() {

    convertible opt(1, 10, 0.068);

    vector<vector<double>> result = opt.convertibleBond(0.3,92,1000,0.1, 10,1100,OptionType::CALL,1);

    
    // Determine the maximum width needed for the largest element
    int maxWidth = 0;
    for (const auto& row : result) {
        for (const auto& element : row) {
            int width = std::to_string(element).length();
            maxWidth = std::max(maxWidth, width);
        }
    }

    // Print the matrix with proper formatting
    for (const auto& row : result) {
        for (const auto& element : row) {
            std::cout << std::setw(maxWidth + 1) << element; // +1 for space between columns
        }
        std::cout << std::endl;
    }
    

    /*
        auto start = std::chrono::high_resolution_clock::now();

        opt.convertibleBond(0.3,92,1000,0.1, 10,1100,OptionType::CALL,1);

        auto end = std::chrono::high_resolution_clock::now();

        // Calculate duration
        std::chrono::duration<double> duration = end - start;

        std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
        */


    return 0;
}




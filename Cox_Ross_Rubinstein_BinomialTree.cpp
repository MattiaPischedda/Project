#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

class Option{
protected:

    double r;
    double T;
    int N;
    double dt;
    double S0;
    double K;
    double sigma;
    char optionType;
    

public:

    //constructor
    Option(double r, double T, int N, double S0, double K, double sigma, char optionType):
                r(r), T(T), N(N), S0(S0), K(K), sigma(sigma), optionType(optionType) {
                    dt = T / N;
                }

   
    virtual double price() {}

};


//Binomial Cox Ross Rubinstein model for European Options
class CRR_European: public Option {
private:

    double u;
    double d;
    double q;
    double disc;

public:

    // Constructor with initialization list
    CRR_European(double r, double T, int N, double S0, double K, double sigma, char optionType)
        : Option(r, T, N, S0, K, sigma, optionType) {
        dt = T / N;
        u = exp(sigma * sqrt(dt));
        d = 1 / u;
        q = (exp(r * dt) - d) / (u - d);
        disc = exp(-r * dt);
    }

    double price() override {

        //initialize the underlying vector
        vector<double> S(N+1, 0);
        S[0] = S0 * pow(u, N);                   //compute the upper node
        for (int i = 1; i < S.size()-1; ++i){
            S[i] = S[i-1]*(d/u);                //iterate to compute the underlying at maturity
        }

        //initialize Optioin vector
        vector<double> Option(N+1, 0);

        try{
            if (optionType == 'C'){
                for (int i = 0; i < Option.size(); ++i){

                    Option[i] = std::max(0.0, S[i] - K);
                }
            }
            
            else if (optionType == 'P'){

                for (int i = 0; i < Option.size(); ++i){

                    Option[i] = std::max(0.0, K - S[i]);
                }
            }
            else{
                cout << "optionType must be C for the call or P for the Put" << endl;
            }
        }
        catch (const std::exception& e){
            std::cerr << e.what() << std::endl;
        }

        for (int i = N; i > 0; i--){
            for (int j = 0; j < i; ++j){

                Option[j] = disc * (Option[j]*q + (1-q)*Option[j+1]);
            }
        }

        return Option[0];
    }
    

};


class American:public Option {
private:

    double u;
    double d;
    double q;
    double disc;

public:

    //Constructor
    American(double r, double T, int N, double S0, double K, double sigma, char optionType)
        : Option(r, T, N, S0, K, sigma, optionType){
        dt = T / N;
        u = exp(sigma * sqrt(dt));
        d = 1 / u;
        q = (exp(r * dt) - d) / (u - d);
        disc = exp(-r * dt);  
    }
    

    double price() override {

        vector<double> S(N+1, 0);
        for (int i = 0; i < S.size(); ++i){
            S[i] = S0 * pow(u, N-i) * pow(d, i);
        }

        vector<double> Opt(N+1, 0);

        try{
            if (optionType == 'C'){

                for (int i = 0; i < Opt.size(); ++i){
                    Opt[i] = std::max(0.0, S[i] - K);
                }
                

                for (int i = N-1; i >= 0; i--){
                    for (int j = 0; j < (i+1); ++j){

                        double S = S0 * pow(u, i-j) * pow(d, j);
                        Opt[j] = disc*(Opt[j]*q + (1-q)*Opt[j+1]);
                        Opt[j] = std::max(Opt[j], S -K);
                    }
                }


            }
            else if (optionType == 'P'){

                for (int i = 0; i < Opt.size(); ++i){
                    Opt[i] = std::max(0.0, K - S[i]);
                }
                
                for (int i = N-1; i >= 0; i--){
                    for (int j = 0; j < (i+1); ++j){

                        double S = S0 * pow(u, i-j) * pow(d, j);
                        Opt[j] = disc*(Opt[j]*q + (1-q)*Opt[j+1]);
                        Opt[j] = std::max(Opt[j], K - S);
                    }
                }        
            }
            else{
                cout << "optionType must be C for the Call or P for the Put" << endl;
            }
        }
        catch (const std::exception& e){
            std::cerr << e.what() << std::endl;
        }

        return Opt[0];
    }

};





int main(){

    CRR_European option1(0.03, 1.0, 5, 100.0, 98.0, 0.33, 'C');
    double result1 = option1.price();

    cout << "The price of the Option is: " << std::setprecision(8)<< result1 << endl;

    American option2(0.06, 1.0, 3, 100, 100, 0.33, 'P');
    double result2 = option2.price();

    cout << "The price of the Options is: " << std::setprecision(8) << result2 << endl;
    

    return 0;
}
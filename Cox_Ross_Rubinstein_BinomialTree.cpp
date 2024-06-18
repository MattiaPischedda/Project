//Implementation of American and European Option using Cox Ross Rubinstein.


#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept> 

using namespace std;

// Enum class to represent option types
enum class OptionType { CALL, PUT };

class Option {
protected:
    double r;
    double T;
    int N;
    double dt;
    double S0;
    double K;
    double sigma;
    OptionType optionType; // Use enum class instead of char

public:
    // Constructor
    Option(double r, double T, int N, double S0, double K, double sigma, OptionType optionType)
        : r(r), T(T), N(N), S0(S0), K(K), sigma(sigma), optionType(optionType) {
        dt = T / N;
    }

    // Pure virtual function for price calculation
    virtual double price() = 0; // Forces subclasses to implement this function
};

// Binomial Cox Ross Rubinstein model for European Options
class CRR_European : public Option {
private:
    double u;
    double d;
    double q;
    double disc;

public:
    // Constructor
    CRR_European(double r, double T, int N, double S0, double K, double sigma, OptionType optionType)
        : Option(r, T, N, S0, K, sigma, optionType) {
        dt = T / N;
        u = exp(sigma * sqrt(dt));
        d = 1 / u;
        q = (exp(r * dt) - d) / (u - d);
        disc = exp(-r * dt);
    }

    double price() override {
        vector<double> S(N + 1, 0);
        S[0] = S0 * pow(u, N);

        for (int i = 1; i < S.size() - 1; ++i) {
            S[i] = S[i - 1] * (d / u);
        }

        vector<double> Option(N + 1, 0);

        // Handle invalid option type
        if (optionType == OptionType::CALL) {
            for (int i = 0; i < Option.size(); ++i) {
                Option[i] = max(0.0, S[i] - K);
            }
        } else if (optionType == OptionType::PUT) {
            for (int i = 0; i < Option.size(); ++i) {
                Option[i] = max(0.0, K - S[i]);
            }
        } else {
            throw invalid_argument("Invalid option type. Must be CALL or PUT.");
        }

        for (int i = N; i > 0; i--) {
            for (int j = 0; j < i; ++j) {
                Option[j] = disc * (Option[j] * q + (1 - q) * Option[j + 1]);
            }
        }

        return Option[0];
    }
};

class American : public Option {
private:
    double u;
    double d;
    double q;
    double disc;

public:
    // Constructor
    American(double r, double T, int N, double S0, double K, double sigma, OptionType optionType)
        : Option(r, T, N, S0, K, sigma, optionType) {
        dt = T / N;
        u = exp(sigma * sqrt(dt));
        d = 1 / u;
        q = (exp(r * dt) - d) / (u - d);
        disc = exp(-r * dt);
    }

    double price() override {
        vector<double> S(N + 1, 0);

        for (int i = 0; i < S.size(); ++i) {
            S[i] = S0 * pow(u, N - i) * pow(d, i);
        }

        vector<double> Opt(N + 1, 0);

        if (optionType == OptionType::CALL) {
            for (int i = 0; i < Opt.size(); ++i) {
                Opt[i] = max(0.0, S[i] - K);
            }

            for (int i = N - 1; i >= 0; i--) {
                for (int j = 0; j < (i + 1); ++j) {
                    double S = S0 * pow(u, i - j) * pow(d, j);
                    Opt[j] = disc * (Opt[j] * q + (1 - q) * Opt[j + 1]);
                    Opt[j] = max(Opt[j], S - K);
                }
            }
        } else if (optionType == OptionType::PUT) {
            for (int i = 0; i < Opt.size(); ++i) {
                Opt[i] = max(0.0, K - S[i]);
            }

            for (int i = N - 1; i >= 0; i--) {
                for (int j = 0; j < (i + 1); ++j) {
                    double S = S0 * pow(u, i - j) * pow(d, j);
                    Opt[j] = disc * (Opt[j] * q + (1 - q) * Opt[j + 1]);
                    Opt[j] = max(Opt[j], K - S);
                }
            }
        } else {
            throw invalid_argument("Invalid option type. Must be CALL or PUT.");
        }

        return Opt[0];
    }
};

int main() {
    // Test cases
    CRR_European option1(0.03, 1.0, 5, 100.0, 98.0, 0.33, OptionType::CALL);
    double result1 = option1.price();
    cout << "The price of the Option is: " << setprecision(8) << result1 << endl;

    American option2(0.06, 1.0, 3, 100, 100, 0.33, OptionType::PUT);
    double result2 = option2.price();
    cout << "The price of the Options is: " << setprecision(8) << result2 << endl;

    return 0;
}

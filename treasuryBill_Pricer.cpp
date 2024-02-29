/*This is a simple treasury bill pricer: The Exercise is taken from Pietro Veronesi's Fixed
Income Securities, chapter 2.
- given the time to maturity and discount, compute:
    - Price
    - Yield
    - Continuously compunded yield
    - Semi Annually compunded yield if T = 1
*/
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;


class bondPricer{

private:

    int row;
    int columns;
    vector<vector<double>> table;
    vector<double> n,T,Discount;

public:

    //constructor
    bondPricer(const int row, const int columns, vector<double> n, vector<double> T, vector<double> Discount):
                row(row), columns(columns), n(n), T(T), Discount(Discount){     
                table.resize(row, std::vector<double>(columns, 0.0));

                for (int i = 0; i < table.size(); ++i) {
                    table[i][0] = n[i];
                    table[i][1] = T[i];
                    table[i][2] = Discount[i];
        }
    }


    vector<vector<double>> compute_price(){

        for (int i = 0; i < row; ++i){

            table[i][3] = 100 * (1 - table[i][1] * table[i][2]);  // Price
            table[i][4] = ((100 - table[i][3]) / table[i][3]) * (365 / table[i][0]); //Bond equivalent yield
            table[i][5] = -(365 / table[i][0]) * log(table[i][3] / 100); // continuosly compunded yield

            if (table[i][1] == 1) {
                table[i][6] = 2 * ((1 / sqrt(table[i][3] / 100)) - 1); //semi-annualy compounded yield
            }
        }

        return table;
    }    

};

int main(){

    int row = 10;
    int columns = 7;
    vector<double> n = {28,28,90,90,90,180,180,180,360,360}; //maturity
    vector<double> T; 
    for (int i = 0; i < n.size(); ++i){
        T.push_back(n[i]/360);             // T = (n / 360)
    }
    vector<double> discount = {0.0348,0.0013,0.0493,0.0476,0.0048,0.0472, 0.0475,0.0089,0.0173,0.019}; //disocunt factor

    bondPricer bond1(row, columns,n,T,discount);
    vector<vector<double>> result = bond1.compute_price();

    //set the precision
    cout << fixed << setprecision(3);

    for (auto row:result){
        for (auto value:row){
            cout << value << "   ";
            
        }
        cout << endl;
    }

    return 0;
}
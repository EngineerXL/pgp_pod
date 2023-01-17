#include <fstream>
#include <iostream>

using namespace std;

const double EPS = 1e-6;

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Too few parameters to run checker!\n";
        return 1;
    }
    cout.precision(12);
    cout << fixed;
    ifstream inp1(argv[1]);
    ifstream inp2(argv[2]);
    double a, b;
    while (inp1 >> a) {
        inp2 >> b;
        double d = abs(a - b);
        if (d > EPS) {
            cout << "ERROR: " << a << ", " << b << '\n';
            cout << "error = " << d << '\n';
            return 0;
        }
    }
}

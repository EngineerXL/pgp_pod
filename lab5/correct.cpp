#include <bits/stdc++.h>

using namespace std;

using pii = pair<int, int>;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    int n;
    cin >> n;
    vector<float> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
    }
    sort(a.begin(), a.end());
    cout.precision(3);
    cout << fixed;
    for (int i = 0; i < n; ++i) {
        if (i) {
            cout << ' ';
        }
        cout << a[i];
    }
    cout << endl;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const double M_PI = acos(-1);



// Функция mu
double mu(double x, double y) {

    if (pow((x + 0.1), 2) / 0.12 + pow((y - 0.2), 2) / 0.04 <= 1)
        return 1.2;
    if (pow((x - 0.32), 2) + pow((y - 0.7), 2) <= 0.01)
        return 0.7;
    if ((x - 0.1) * (x - 0.1) + pow((y - 0.8), 2) <= 0.01)
        return 0.5;

    if (y >= -0.3 && y <= 0 && x >= 0.3 && x <= 0.6)
        return 0.4;

    if (pow((x + 0.2), 2) + (y + 0.5) * (y + 0.5) <= 0.08)
        return 1;

    return 2;
}

// Функция для вычисления интеграла методом трапеций
double trapeze(double rho, double phi) {

    double len = sqrt(1 - rho * rho);
    double h = 2.0 * len / 1000.0;
    double cs = cos(phi);
    double sn = sin(phi);
    double integral = (mu(rho * cs - len * sn, rho * sn + len * cs)
        + mu(rho * cs + len * sn, rho * sn - len * cs)) / 2.0;

    for (int tau = 1; tau < 1000; ++tau) {
        double x = rho * cs - sn * (-len + tau * h);
        double y = rho * sn + cs * (-len + tau * h);
        integral += mu(x, y);
    }
    return integral * h;
}
// Функция для вычисления конечных разностей
vector<vector<double>> diff(const vector<vector<double>>& m, int M1, int M2) {
    vector<vector<double>> c(2 * M1 + 3, vector<double>(M2, 0));
    for (int i = 0; i < M2; ++i) {
        c[0][i] = 0;
    }
    for (int j = -M1 + 1; j < M1; ++j) {
        for (int i = 0; i < M2; ++i) {
            c[j + M1][i] = abs(m[j + M1 + 1][i] - m[j + M1][i]);
        }
    }
    for (int i = 0; i < M2; ++i) {
        c[2 * M1 + 1][i] = 0;
    }
    return c;
}

// Функция для обратного проецирования
double back_projection(const vector<vector<double>>& m, double x, double y, int M1) {
    int N = static_cast<int>(m[0].size() - 1);  // Устранение предупреждения C4267
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double phi = M_PI * i / N;
        double rho = x * cos(phi) + y * sin(phi);
        int s = static_cast<int>(floor(rho * M1));
        double t = abs(rho * M1 - s);
        sum += (1 - t) * m[s + M1][i] + t * m[s + M1 + 1][i];
    }
    return 2 * M1 * sum / (M_PI * N);
}

int main() {
    int M1 = 1000000;
    int M2 = 3000000;
    int N = 400;
    vector<vector<double>> matrix(2 * M1 + 2, vector<double>(M2, 0));


    for (int i = -M1; i <= M1; ++i) {
        double rho = static_cast<double>(i) / M1;
        double val = sqrt(1 - rho * rho);
        for (int j = 0; j < M2; ++j) {
            double phi = M_PI * j / M2;
            matrix[i + M1][j] = val;
        }
    }
    for (int i = -M1; i <= M1; ++i) {
        double rho = static_cast<double>(i) / M1;
        for (int j = 0; j < M2; ++j) {
            double phi = M_PI * j / M2;
            matrix[i + M1][j] = trapeze(rho, phi);
        }
    }

    vector<vector<double>> d = diff(matrix, M1, M2);
    vector<vector<double>> fx(2 * N + 1, vector<double>(2 * N + 1, 0));
    for (int i = -N; i <= N; ++i) {
        double y = -static_cast<double>(i) / N;
        for (int j = -N; j <= N; ++j) {
            double x = static_cast<double>(j) / N;
            if (sqrt(x * x + y * y) > 1) {
                fx[i + N][j + N] = 0;
            }
            else {
                fx[i + N][j + N] = back_projection(d, x, y, M1);
            }
        }
    }

    // Отображение изображения с помощью OpenCV
    Mat image(2 * N + 1, 2 * N + 1, CV_8UC3, Scalar(255, 255, 255));
    double maxVal = *max_element(fx[0].begin(), fx[0].end());
    maxVal *= 0.95;
    for (int i = 0; i < 2 * N + 1; ++i) {
        for (int j = 0; j < 2 * N + 1; ++j) {
            int c = static_cast<int>(255 * fx[i][j] / maxVal);
            int a = 255 - c;
            image.at<Vec3b>(i, j) = Vec3b(a, a, a);
        }
    }
    imshow("Image", image);
    imwrite("1000-3000.png", image);
    waitKey(0);

    return 0;
}

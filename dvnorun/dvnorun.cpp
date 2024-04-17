#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const double M_PI = 3.14159265358979323846; // Определение константы M_PI

// Функция mu
double mu(double x, double y);  // Объявление функции mu

// Функция для вычисления интеграла методом трапеций
double trapeze(double t, double cs, double sn, double l, double r) {
    double h = (r - l) / 50.0;
    double integral = 0.0;
    for (int tau = 0; tau < 50; ++tau) {
        double dh = (l + tau * h - (l + (tau + 1) * h)) / 2.0;
        double x = t * cs - (l + tau * h) * sn;
        double y = t * sn + (l + tau * h) * cs;
        double x1 = t * cs - (l + (tau + 1) * h) * sn;
        double y1 = t * sn + (l + (tau + 1) * h) * cs;
        integral += dh * (mu(x, y) + mu(x1, y1));
    }
    return integral;
}

// Функция mu
double mu(double x, double y) {
    return pow(pow(x, 2) + pow(y, 2), 2) - 7 * pow(pow(x, 2) - pow(y, 2), 2);
}

// Функция для вычисления конечных разностей
vector<vector<double>> diff(const vector<vector<double>>& m) {
    int M = static_cast<int>(m.size() - 2);  // Устранение предупреждения C4267
    int N = static_cast<int>(m[0].size());  // Устранение предупреждения C4267
    vector<vector<double>> c(M + 2, vector<double>(N, 0));
    for (int i = 0; i < N; ++i) {
        c[0][i] = 0;
    }
    for (int j = -N + 1; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            c[j + N][i] = abs(m[j + N + 1][i] - m[j + N][i]);
        }
    }
    for (int i = 0; i < N; ++i) {
        c[M][i] = 0;
    }
    return c;
}

// Функция для обратного проецирования
double back_projection(const vector<vector<double>>& m, double x, double y) {
    int N = static_cast<int>(m[0].size());  // Устранение предупреждения C4267
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double phi = M_PI * i / N;
        double rho = x * cos(phi) + y * sin(phi);
        int s = static_cast<int>(floor(rho * N));
        double t = abs(rho * N - s);
        sum += (1 - t) * m[s + N][i] + t * m[s + N + 1][i];
    }
    return 2 * N * sum / (M_PI * N);
}

int main() {
    int M = 200;
    int N = M / 2;
    vector<vector<double>> matrix(M + 2, vector<double>(N, 0));

    // Заполнение матрицы matrix
    for (int i = -N; i <= N; ++i) {
        double t = static_cast<double>(i) / N;
        double val = 2 * sqrt(1 - t * t);
        for (int j = 0; j < N; ++j) {
            matrix[i + N][j] = val;
        }
    }
    for (int i = -N; i <= N; ++i) {
        double t = static_cast<double>(i) / N;
        double g = sqrt(1 - t * t);
        double a = -g, b = g;
        for (int j = 0; j < N; ++j) {
            double phi = M_PI * j / N;
            matrix[i + N][j] = trapeze(t, cos(phi), sin(phi), a, b);
        }
    }

    vector<vector<double>> d = diff(matrix);
    vector<vector<double>> fx(M + 1, vector<double>(M + 1, 0));
    for (int i = -N; i <= N; ++i) {
        double y = -static_cast<double>(i) / N;
        for (int j = -N; j <= N; ++j) {
            double x = static_cast<double>(j) / N;
            if (sqrt(x * x + y * y) > 1) {
                fx[i + N][j + N] = 0;
            }
            else {
                fx[i + N][j + N] = back_projection(d, x, y);
            }
        }
    }

    // Отображение изображения с помощью OpenCV
    Mat image(M + 1, M + 1, CV_8UC3, Scalar(255, 255, 255));
    double maxVal = *max_element(fx[0].begin(), fx[0].end());
    for (int i = 0; i < M + 1; ++i) {
        for (int j = 0; j < M + 1; ++j) {
            int c = static_cast<int>(255 * fx[i][j] / maxVal);
            image.at<Vec3b>(i, j) = Vec3b(c, c, c);
        }
    }
    imshow("Image", image);
    waitKey(0);

    return 0;
}
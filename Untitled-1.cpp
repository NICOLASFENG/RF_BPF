#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>

// 计算 Γin (S11) 的原始函数
double calculate_Func1(double C1, double C2, double C3, double C4, double C5,
                       double C6, double C7, double C8, double C9, double w,
                       double L1, double L2, double L3, double L4, double L5)
{
    double R2, R3, r1, r2, r3, R11, R22, R33, r22, r33, R111, R222, R333, r111, r222, r333, zin_real, zin_complex;
    
    // 原始逻辑...
    r3 = w * L5 + ((w * C4 * L5) / C7) - 1.0 / (w * C7);
    r2 = w * L5 + ((w * C7 * L5) / C4) - 1.0 / (w * C4);
    r1 = 1.0 / (w * w * w * C4 * C7 * L5) - 1.0 / (w * C4) - 1.0 / (w * C7);
    R2 = (r2 * w * L3) / (r2 + w * L3);
    R3 = (r3 * w * L4) / (r3 + w * L4);
    R11 = (R2 * R3) / (R2 + R3 + r1);
    R22 = (r1 * R3) / (R2 + R3 + r1);
    R33 = (r1 * R2) / (R2 + R3 + r1);
    r22 = R22 - 1.0 / (w * C6);
    r33 = R33 - 1.0 / (w * C3);
    R111 = (r33 * r22) / (r33 + r22 - 1.0 / (w * C5));
    R222 = (-1.0) * (r33 * (1.0 / (w * C5))) / (r22 - 1.0 / (w * C5) + r33);
    R333 = (-1.0) * (r22 * (1.0 / (w * C5))) / (r22 - 1.0 / (w * C5) + r33);
    r111 = R111 + R11;
    r222 = R222 - 1.0 / (w * C2) + ((L1 / ((1.0 / w) - w * L1 * C1)));
    r333 = R333 - 1.0 / (w * C8) + ((L2 / ((1.0 / w) - w * L2 * C9)));
    
    zin_real = (50 * r111 * (r111 + r333) - r111 * r333 * 50) / (2500 + (r111 + r333) * (r111 + r333));
    zin_complex = r222 + ((r111 * 2500 + r333 * r111 * (r111 + r333)) / (2500 + (r111 + r333) * (r111 + r333)));
    
    return sqrt((((zin_real - 50.0) * (zin_real - 50.0) + zin_complex * zin_complex)
                 / ((zin_real + 50.0) * (zin_real + 50.0) + zin_complex * zin_complex)));
}

int main() {
    // 1. 定义参数列表以便于循环输入或展示
    std::vector<std::string> names = {
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", 
        "w (角频率)", "L1", "L2", "L3", "L4", "L5"
    };
    
    // 2. 初始化 15 个输入值 (你的函数定义里其实有 15 个参数，包含 w)
    // 这里使用一些典型的射频电路量级作为示例值（单位通常为 pF 和 nH，w 为 rad/s）
    double p[15] = {
        0.2e-12, 1.49e-12, 6e-12, 1.12e-12, 1.13e-12, 5e-12, 1.43e-12, 0.67e-12, 0.81e-12, // C1-C9
        2 * M_PI * 3.3e9,                                               // w (1GHz)
        0.87e-9, 1.5e-9, 0.87e-9, 0.77e-9, 0.1e-9                                  // L1-L5
    };

    std::cout << "--- 射频电路 S11 反射系数计算 ---" << std::endl;
    std::cout << "当前输入参数设定：" << std::endl;
    for(int i = 0; i < 15; ++i) {
        std::cout << std::left << std::setw(12) << names[i] << ": " << std::scientific << p[i] << std::endl;
    }

    // 3. 调用函数
    // 注意：你的函数定义中确实有 15 个形参 (9个C + 1个w + 5个L)
    try {
        double s11 = calculate_Func1(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], 
                                     p[9], p[10], p[11], p[12], p[13], p[14]);

        // 4. 输出结果与判定
        std::cout << "\n--------------------------------" << std::endl;
        std::cout << "计算结果 S11 (Magnitude): " << std::fixed << std::setprecision(6) << s11 << std::endl;

        if (s11 < 0.14) {
            std::cout << "状态判定: 优秀 (完美匹配, <-20dB)" << std::endl;
        } else if (s11 < 0.30061) {
            std::cout << "状态判定: 良好 (良好匹配, <-10dB)" << std::endl;
        } else {
            std::cout << "状态判定: 匹配较差 (请调整参数)" << std::endl;
        }
    } catch (...) {
        std::cerr << "错误：计算过程中可能存在除零异常，请检查输入参数。" << std::endl;
    }

    return 0;
}
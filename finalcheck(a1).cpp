#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double PF = 1e-12;
constexpr double NH = 1e-9;
constexpr double GHZ = 1e9;
constexpr double Z0 = 50.0;
constexpr double INF = 1e100;

using Complex = std::complex<double>;

enum GeneIndex {
    C2 = 0,
    C3 = 1,
    C4 = 2,
    C5 = 3,
    C6 = 4,
    C7 = 5,
    C8 = 6,
    C9 = 7,
    GENE_COUNT = 8
};

constexpr std::array<const char*, GENE_COUNT> kGeneNames = {
    "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"
};

struct FixedValues {
    double c1 = 0.20 * PF;
    double l1 = 0.87 * NH;
    double l2 = 1.50 * NH;
    double l3 = 0.87 * NH;
    double l4 = 0.77 * NH;
    double l5 = 0.10 * NH;
};

struct BandSpec {
    std::string name;
    double pass_lo_hz = 0.0;
    double pass_hi_hz = 0.0;
    double center_hz = 0.0;

    double lower_stop_lo_hz = 0.05 * GHZ;
    double lower_stop_hi_hz = 0.0;
    double upper_stop_lo_hz = 0.0;
    double upper_stop_hi_hz = 15.0 * GHZ;

    double harmonic2_lo_hz = 0.0;
    double harmonic2_hi_hz = 0.0;
    double harmonic3_lo_hz = 0.0;
    double harmonic3_hi_hz = 0.0;

    double pass_return_loss_min_db = 16.0;
    double pass_insertion_loss_max_db = 0.30;
    double stop_rejection_min_db = 20.0;
    double harmonic_rejection_min_db = 25.0;
};

struct FrequencyPlan {
    std::vector<double> pass;
    std::vector<double> lower_stop;
    std::vector<double> upper_stop;
    std::vector<double> harmonic2;
    std::vector<double> harmonic3;
};

struct SearchBounds {
    std::array<double, GENE_COUNT> lower{};
    std::array<double, GENE_COUNT> upper{};
    std::array<double, GENE_COUNT> step{};
};

struct TransmissionZeros {
    double tz1_hz = 0.0;
    double tz2_hz = 0.0;
    double tz3_hz = 0.0;
    double tz4_hz = 0.0;
};

struct SParameters {
    Complex s11{};
    Complex s21{};
    bool valid = false;
};

struct Metrics {
    bool valid = true;
    bool pass_return_loss_ok = false;
    bool pass_insertion_loss_ok = false;
    bool lower_stop_ok = false;
    bool upper_stop_ok = false;
    bool harmonic2_ok = false;
    bool harmonic3_ok = false;
    bool transmission_zero_order_ok = false;
    bool constraints_ok = false;

    double fitness = INF;
    double min_pass_return_loss_db = INF;
    double max_pass_insertion_loss_db = -INF;
    double avg_pass_return_loss_db = 0.0;
    double avg_pass_insertion_loss_db = 0.0;

    double center_return_loss_db = 0.0;
    double center_insertion_loss_db = 0.0;
    double low_edge_return_loss_db = 0.0;
    double high_edge_return_loss_db = 0.0;

    double lower_stop_min_rejection_db = INF;
    double upper_stop_min_rejection_db = INF;
    double harmonic2_min_rejection_db = INF;
    double harmonic3_min_rejection_db = INF;

    double deepest_stop_rejection_db = -INF;
    double deepest_stop_frequency_hz = 0.0;

    TransmissionZeros zeros;
};

struct Candidate {
    std::array<double, GENE_COUNT> gene{};
    double fitness = INF;
};

struct GAConfig {
    int population_size = 360;
    int max_generations = 900;
    int max_restarts = 4;
    int elite_count = 18;
    int tournament_size = 4;
    int dense_check_interval = 20;
    int progress_interval = 100;
    int min_generations_after_feasible = 160;
    int stagnation_generations = 120;
    double crossover_rate = 0.92;
    double mutation_rate = 0.24;
    double reset_mutation_rate = 0.025;
    double resonance_mutation_rate = 0.18;
    double mutation_scale = 0.10;
    std::uint64_t seed = 20260427ULL;
};

struct SearchResult {
    std::string band_name;
    Candidate best;
    Metrics metrics;
    bool feasible = false;
    bool stopped_by_solution = false;
    int restart = 0;
    int generation = 0;
    double elapsed_seconds = 0.0;
};

std::mutex g_cout_mutex;

double clamp_value(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

double sqr(double x) {
    return x * x;
}

bool finite_positive(double x) {
    return std::isfinite(x) && x > 0.0;
}

double db_from_magnitude(double mag) {
    return -20.0 * std::log10(std::max(mag, 1e-15));
}

double cap_for_tz(double f_hz, double inductance_h) {
    const double w = 2.0 * PI * f_hz;
    return 1.0 / (w * w * inductance_h);
}

double tz_from_lc(double capacitance_f, double inductance_h) {
    if (!finite_positive(capacitance_f) || !finite_positive(inductance_h)) {
        return 0.0;
    }
    return 1.0 / (2.0 * PI * std::sqrt(capacitance_f * inductance_h));
}

std::vector<double> linspace(double lo_hz, double hi_hz, int count) {
    std::vector<double> values;
    if (hi_hz <= lo_hz || count <= 0) {
        return values;
    }
    values.reserve(static_cast<std::size_t>(count));
    if (count == 1) {
        values.push_back(0.5 * (lo_hz + hi_hz));
        return values;
    }
    const double step = (hi_hz - lo_hz) / static_cast<double>(count - 1);
    for (int i = 0; i < count; ++i) {
        values.push_back(lo_hz + static_cast<double>(i) * step);
    }
    return values;
}

BandSpec make_band(std::string name, double pass_lo_hz, double pass_hi_hz) {
    const double transition_hz = 0.25 * GHZ;

    BandSpec band;
    band.name = std::move(name);
    band.pass_lo_hz = pass_lo_hz;
    band.pass_hi_hz = pass_hi_hz;
    band.center_hz = 0.5 * (pass_lo_hz + pass_hi_hz);

    band.lower_stop_hi_hz = pass_lo_hz - transition_hz;
    band.lower_stop_hi_hz = std::max(band.lower_stop_hi_hz, band.lower_stop_lo_hz);

    band.upper_stop_lo_hz = pass_hi_hz + transition_hz;
    band.upper_stop_hi_hz = 15.0 * GHZ;
    if (band.upper_stop_hi_hz <= band.upper_stop_lo_hz) {
        band.upper_stop_hi_hz = 15.0 * GHZ;
    }

    const double harmonic_window = 0.035;
    band.harmonic2_lo_hz = 2.0 * band.center_hz * (1.0 - harmonic_window);
    band.harmonic2_hi_hz = 2.0 * band.center_hz * (1.0 + harmonic_window);
    band.harmonic3_lo_hz = 3.0 * band.center_hz * (1.0 - harmonic_window);
    band.harmonic3_hi_hz = std::min(15.0 * GHZ,
                                    3.0 * band.center_hz * (1.0 + harmonic_window));
    return band;
}

FrequencyPlan make_frequency_plan(const BandSpec& band, bool dense) {
    FrequencyPlan plan;
    plan.pass = linspace(band.pass_lo_hz, band.pass_hi_hz, dense ? 91 : 35);
    plan.lower_stop = linspace(band.lower_stop_lo_hz, band.lower_stop_hi_hz,
                               dense ? 91 : 35);
    plan.upper_stop = linspace(band.upper_stop_lo_hz, band.upper_stop_hi_hz,
                               dense ? 151 : 55);
    plan.harmonic2 = linspace(band.harmonic2_lo_hz, band.harmonic2_hi_hz,
                              dense ? 25 : 9);
    plan.harmonic3 = linspace(band.harmonic3_lo_hz, band.harmonic3_hi_hz,
                              dense ? 25 : 9);
    return plan;
}

SearchBounds make_bounds(const BandSpec& band, const FixedValues& fixed) {
    SearchBounds bounds;
    bounds.lower = {
        0.25 * PF, 0.35 * PF, 0.10 * PF, 0.30 * PF,
        0.35 * PF, 0.10 * PF, 0.12 * PF, 0.12 * PF
    };
    bounds.upper = {
        3.20 * PF, 7.50 * PF, 3.20 * PF, 2.50 * PF,
        7.50 * PF, 3.20 * PF, 1.80 * PF, 1.40 * PF
    };
    bounds.step = {
        0.01 * PF, 0.01 * PF, 0.01 * PF, 0.01 * PF,
        0.01 * PF, 0.01 * PF, 0.01 * PF, 0.01 * PF
    };

    // TZ3 = 1 / (2*pi*sqrt(C9*L2)).  Narrow C9 around the expected
    // upper-side zero to reduce wasted search while still allowing margin.
    const double c9_lo = cap_for_tz(2.05 * band.center_hz, fixed.l2);
    const double c9_hi = cap_for_tz(1.08 * band.center_hz, fixed.l2);
    bounds.lower[C9] = clamp_value(c9_lo, 0.12 * PF, 1.40 * PF);
    bounds.upper[C9] = clamp_value(c9_hi, bounds.lower[C9] + 0.05 * PF, 1.40 * PF);
    return bounds;
}

double quantize(double value, double step) {
    if (step <= 0.0) {
        return value;
    }
    return std::round(value / step) * step;
}

void repair_candidate(std::array<double, GENE_COUNT>& gene,
                      const SearchBounds& bounds) {
    for (int i = 0; i < GENE_COUNT; ++i) {
        gene[i] = clamp_value(gene[i], bounds.lower[i], bounds.upper[i]);
        gene[i] = quantize(gene[i], bounds.step[i]);
        gene[i] = clamp_value(gene[i], bounds.lower[i], bounds.upper[i]);
    }
}

TransmissionZeros compute_transmission_zeros(
    const std::array<double, GENE_COUNT>& gene,
    const FixedValues& fixed) {
    TransmissionZeros z;
    z.tz1_hz = tz_from_lc(gene[C3] + gene[C4], fixed.l3);
    z.tz2_hz = tz_from_lc(gene[C6] + gene[C7], fixed.l4);
    z.tz3_hz = tz_from_lc(gene[C9], fixed.l2);
    z.tz4_hz = tz_from_lc(fixed.c1, fixed.l1);
    return z;
}

void stamp_admittance(std::array<std::array<Complex, 9>, 9>& y,
                      int a,
                      int b,
                      Complex value) {
    if (b < 0) {
        y[a][a] += value;
        return;
    }
    y[a][a] += value;
    y[b][b] += value;
    y[a][b] -= value;
    y[b][a] -= value;
}

bool solve_7_by_2(
    std::array<std::array<Complex, 7>, 7> a,
    std::array<std::array<Complex, 2>, 7> b,
    std::array<std::array<Complex, 2>, 7>& x) {
    std::array<std::array<Complex, 9>, 7> aug{};
    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 7; ++c) {
            aug[r][c] = a[r][c];
        }
        aug[r][7] = b[r][0];
        aug[r][8] = b[r][1];
    }

    for (int col = 0; col < 7; ++col) {
        int pivot = col;
        double pivot_abs = std::abs(aug[pivot][col]);
        for (int r = col + 1; r < 7; ++r) {
            const double candidate_abs = std::abs(aug[r][col]);
            if (candidate_abs > pivot_abs) {
                pivot_abs = candidate_abs;
                pivot = r;
            }
        }
        if (pivot_abs < 1e-24 || !std::isfinite(pivot_abs)) {
            return false;
        }
        if (pivot != col) {
            std::swap(aug[pivot], aug[col]);
        }

        const Complex inv_pivot = Complex(1.0, 0.0) / aug[col][col];
        for (int c = col; c < 9; ++c) {
            aug[col][c] *= inv_pivot;
        }

        for (int r = 0; r < 7; ++r) {
            if (r == col) {
                continue;
            }
            const Complex factor = aug[r][col];
            if (std::abs(factor) == 0.0) {
                continue;
            }
            for (int c = col; c < 9; ++c) {
                aug[r][c] -= factor * aug[col][c];
            }
        }
    }

    for (int r = 0; r < 7; ++r) {
        x[r][0] = aug[r][7];
        x[r][1] = aug[r][8];
    }
    return true;
}

SParameters calculate_s_parameters(const std::array<double, GENE_COUNT>& gene,
                                   const FixedValues& fixed,
                                   double f_hz) {
    if (!finite_positive(f_hz)) {
        return {};
    }

    const double w = 2.0 * PI * f_hz;
    const Complex j(0.0, 1.0);
    std::array<std::array<Complex, 9>, 9> y{};

    auto cap = [&](int a, int b, double c) {
        stamp_admittance(y, a, b, j * w * c);
    };
    auto ind = [&](int a, int b, double l) {
        stamp_admittance(y, a, b, Complex(1.0, 0.0) / (j * w * l));
    };

    // Full Fig. 2 topology.
    // 0 = port 1, 8 = port 2, 1..7 are internal nodes.
    // L1//C1 and L2//C9 are series resonator branches between two nodes.
    cap(0, 1, fixed.c1);
    ind(0, 1, fixed.l1);
    cap(1, 2, gene[C2]);
    cap(2, 3, gene[C3]);
    ind(3, -1, fixed.l3);
    cap(3, 6, gene[C4]);
    ind(6, -1, fixed.l5);
    cap(6, 5, gene[C7]);
    cap(4, 5, gene[C6]);
    ind(5, -1, fixed.l4);
    cap(2, 4, gene[C5]);
    cap(4, 7, gene[C8]);
    cap(7, 8, gene[C9]);
    ind(7, 8, fixed.l2);

    constexpr std::array<int, 2> ports = {0, 8};
    constexpr std::array<int, 7> internal = {1, 2, 3, 4, 5, 6, 7};

    std::array<std::array<Complex, 2>, 2> ypp{};
    std::array<std::array<Complex, 7>, 2> ypi{};
    std::array<std::array<Complex, 2>, 7> yip{};
    std::array<std::array<Complex, 7>, 7> yii{};

    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            ypp[r][c] = y[ports[r]][ports[c]];
        }
        for (int c = 0; c < 7; ++c) {
            ypi[r][c] = y[ports[r]][internal[c]];
        }
    }
    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 2; ++c) {
            yip[r][c] = y[internal[r]][ports[c]];
        }
        for (int c = 0; c < 7; ++c) {
            yii[r][c] = y[internal[r]][internal[c]];
        }
    }

    std::array<std::array<Complex, 2>, 7> solved{};
    if (!solve_7_by_2(yii, yip, solved)) {
        return {};
    }

    std::array<std::array<Complex, 2>, 2> yeff = ypp;
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            Complex sum = 0.0;
            for (int k = 0; k < 7; ++k) {
                sum += ypi[r][k] * solved[k][c];
            }
            yeff[r][c] -= sum;
        }
    }

    std::array<std::array<Complex, 2>, 2> a{};
    std::array<std::array<Complex, 2>, 2> b{};
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            const Complex identity = (r == c) ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
            a[r][c] = identity + Z0 * yeff[r][c];
            b[r][c] = identity - Z0 * yeff[r][c];
        }
    }

    const Complex det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if (std::abs(det) < 1e-24 || !std::isfinite(std::abs(det))) {
        return {};
    }

    const std::array<std::array<Complex, 2>, 2> inv_a = {{
        {{ a[1][1] / det, -a[0][1] / det }},
        {{ -a[1][0] / det, a[0][0] / det }}
    }};

    std::array<std::array<Complex, 2>, 2> s{};
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            s[r][c] = b[r][0] * inv_a[0][c] + b[r][1] * inv_a[1][c];
        }
    }

    if (!std::isfinite(std::abs(s[0][0])) || !std::isfinite(std::abs(s[1][0]))) {
        return {};
    }
    return {s[0][0], s[1][0], true};
}

double lower_target_zero_ratio() {
    return 2.20 / 3.75;
}

double upper_target_zero_ratio() {
    return 5.59 / 3.75;
}

double outside_window(double x, double lo, double hi) {
    if (x < lo) {
        return lo - x;
    }
    if (x > hi) {
        return x - hi;
    }
    return 0.0;
}

double add_soft_target(double x, double target, double width) {
    return sqr((x - target) / width);
}

Metrics evaluate_candidate(const std::array<double, GENE_COUNT>& gene,
                           const FixedValues& fixed,
                           const BandSpec& band,
                           const FrequencyPlan& plan) {
    Metrics m;
    m.zeros = compute_transmission_zeros(gene, fixed);
    double penalty = 0.0;
    double objective = 0.0;

    auto invalid = [&]() {
        m.valid = false;
        m.fitness = INF;
        return m;
    };

    double pass_rl_sum = 0.0;
    double pass_il_sum = 0.0;
    int pass_count = 0;
    for (double f : plan.pass) {
        const auto sp = calculate_s_parameters(gene, fixed, f);
        if (!sp.valid) {
            return invalid();
        }
        const double rl_db = db_from_magnitude(std::abs(sp.s11));
        const double il_db = db_from_magnitude(std::abs(sp.s21));
        if (!std::isfinite(rl_db) || !std::isfinite(il_db)) {
            return invalid();
        }
        m.min_pass_return_loss_db = std::min(m.min_pass_return_loss_db, rl_db);
        m.max_pass_insertion_loss_db = std::max(m.max_pass_insertion_loss_db, il_db);
        pass_rl_sum += rl_db;
        pass_il_sum += il_db;
        ++pass_count;

        const double rl_deficit = std::max(0.0, band.pass_return_loss_min_db - rl_db);
        const double il_excess = std::max(0.0, il_db - band.pass_insertion_loss_max_db);
        penalty += 4500.0 * sqr(rl_deficit);
        penalty += 2800.0 * sqr(il_excess);
    }

    if (pass_count == 0) {
        return invalid();
    }
    m.avg_pass_return_loss_db = pass_rl_sum / static_cast<double>(pass_count);
    m.avg_pass_insertion_loss_db = pass_il_sum / static_cast<double>(pass_count);

    auto edge_or_center_metric = [&](double f, double& rl_db, double& il_db) {
        const auto sp = calculate_s_parameters(gene, fixed, f);
        if (!sp.valid) {
            rl_db = -INF;
            il_db = INF;
            return;
        }
        rl_db = db_from_magnitude(std::abs(sp.s11));
        il_db = db_from_magnitude(std::abs(sp.s21));
    };
    edge_or_center_metric(band.center_hz, m.center_return_loss_db,
                          m.center_insertion_loss_db);
    double unused_il = 0.0;
    edge_or_center_metric(band.pass_lo_hz, m.low_edge_return_loss_db, unused_il);
    edge_or_center_metric(band.pass_hi_hz, m.high_edge_return_loss_db, unused_il);

    auto scan_rejection = [&](const std::vector<double>& frequencies,
                              double target_db,
                              double weight,
                              double& min_rejection_db) {
        min_rejection_db = frequencies.empty() ? INF : INF;
        for (double f : frequencies) {
            const auto sp = calculate_s_parameters(gene, fixed, f);
            if (!sp.valid) {
                m.valid = false;
                continue;
            }
            const double rejection_db = db_from_magnitude(std::abs(sp.s21));
            min_rejection_db = std::min(min_rejection_db, rejection_db);
            if (rejection_db > m.deepest_stop_rejection_db) {
                m.deepest_stop_rejection_db = rejection_db;
                m.deepest_stop_frequency_hz = f;
            }
            const double deficit = std::max(0.0, target_db - rejection_db);
            penalty += weight * sqr(deficit);
        }
    };

    scan_rejection(plan.lower_stop, band.stop_rejection_min_db, 850.0,
                   m.lower_stop_min_rejection_db);
    scan_rejection(plan.upper_stop, band.stop_rejection_min_db, 850.0,
                   m.upper_stop_min_rejection_db);
    scan_rejection(plan.harmonic2, band.harmonic_rejection_min_db, 1400.0,
                   m.harmonic2_min_rejection_db);
    scan_rejection(plan.harmonic3, band.harmonic_rejection_min_db, 1400.0,
                   m.harmonic3_min_rejection_db);
    if (!m.valid) {
        return invalid();
    }

    const double r1 = m.zeros.tz1_hz / band.center_hz;
    const double r2 = m.zeros.tz2_hz / band.center_hz;
    const double r3 = m.zeros.tz3_hz / band.center_hz;
    const double r4 = m.zeros.tz4_hz / band.center_hz;

    // Paper formulas:
    // TZ1 = 1/(2*pi*sqrt((C3+C4)*L3))
    // TZ2 = 1/(2*pi*sqrt((C6+C7)*L4))
    // TZ3 = 1/(2*pi*sqrt(C9*L2))
    // TZ4 = 1/(2*pi*sqrt(C1*L1))
    // The windows are soft search guides, not replacements for S-parameter
    // verification.
    penalty += 600.0 * sqr(outside_window(r1, 0.45, 0.82));
    penalty += 600.0 * sqr(outside_window(r2, 0.45, 0.82));
    penalty += 450.0 * sqr(outside_window(r3, 1.05, 2.10));
    objective += 3.0 * add_soft_target(r1, lower_target_zero_ratio(), 0.18);
    objective += 3.0 * add_soft_target(r2, lower_target_zero_ratio(), 0.18);
    objective += 2.0 * add_soft_target(r3, upper_target_zero_ratio(), 0.25);

    m.pass_return_loss_ok =
        m.min_pass_return_loss_db >= band.pass_return_loss_min_db;
    m.pass_insertion_loss_ok =
        m.max_pass_insertion_loss_db <= band.pass_insertion_loss_max_db;
    m.lower_stop_ok =
        m.lower_stop_min_rejection_db >= band.stop_rejection_min_db;
    m.upper_stop_ok =
        m.upper_stop_min_rejection_db >= band.stop_rejection_min_db;
    m.harmonic2_ok =
        m.harmonic2_min_rejection_db >= band.harmonic_rejection_min_db;
    m.harmonic3_ok =
        m.harmonic3_min_rejection_db >= band.harmonic_rejection_min_db;
    m.transmission_zero_order_ok =
        (m.zeros.tz1_hz < band.pass_lo_hz) &&
        (m.zeros.tz2_hz < band.pass_lo_hz) &&
        (m.zeros.tz3_hz > band.pass_hi_hz) &&
        (m.zeros.tz4_hz > band.pass_hi_hz) &&
        (r4 > 1.40);

    m.constraints_ok =
        m.pass_return_loss_ok &&
        m.pass_insertion_loss_ok &&
        m.lower_stop_ok &&
        m.upper_stop_ok &&
        m.harmonic2_ok &&
        m.harmonic3_ok &&
        m.transmission_zero_order_ok;

    objective += 0.25 * std::max(0.0, 22.0 - m.min_pass_return_loss_db);
    objective += 8.0 * m.max_pass_insertion_loss_db;
    objective += 1.5 * m.avg_pass_insertion_loss_db;
    objective += 0.05 * std::max(0.0, 35.0 - m.lower_stop_min_rejection_db);
    objective += 0.05 * std::max(0.0, 35.0 - m.upper_stop_min_rejection_db);
    objective += 0.04 * std::max(0.0, 40.0 - m.harmonic2_min_rejection_db);
    objective += 0.04 * std::max(0.0, 40.0 - m.harmonic3_min_rejection_db);

    m.fitness = penalty + objective;
    return m;
}

std::array<double, GENE_COUNT> make_relation_seed(const BandSpec& band,
                                                  const FixedValues& fixed,
                                                  double c8_pf,
                                                  double lower_ratio,
                                                  double upper_ratio) {
    std::array<double, GENE_COUNT> seed{};
    const double c34_sum = cap_for_tz(lower_ratio * band.center_hz, fixed.l3);
    const double c67_sum = cap_for_tz((lower_ratio + 0.02) * band.center_hz, fixed.l4);
    const double c9_value = cap_for_tz(upper_ratio * band.center_hz, fixed.l2);

    const double c34_split = 5.0 / (5.0 + 1.12);
    const double c67_split = 5.0 / (5.0 + 1.43);

    seed[C2] = 0.92 * PF;
    seed[C3] = c34_sum * c34_split;
    seed[C4] = c34_sum * (1.0 - c34_split);
    seed[C5] = 1.14 * PF;
    seed[C6] = c67_sum * c67_split;
    seed[C7] = c67_sum * (1.0 - c67_split);
    seed[C8] = c8_pf * PF;
    seed[C9] = c9_value;
    return seed;
}

std::vector<std::array<double, GENE_COUNT>> make_seed_bank(
    const BandSpec& band,
    const FixedValues& fixed,
    const SearchBounds& bounds) {
    const double scale_linear = 3.75 * GHZ / band.center_hz;
    const double scale_lc = sqr(scale_linear);

    std::vector<std::array<double, GENE_COUNT>> seeds;

    std::array<double, GENE_COUNT> paper = {
        0.92 * PF, 5.00 * PF, 1.12 * PF, 1.14 * PF,
        5.00 * PF, 1.43 * PF, 0.67 * PF, 0.54 * PF
    };
    seeds.push_back(paper);

    std::array<double, GENE_COUNT> scaled_lc = {
        0.92 * PF,
        5.00 * scale_lc * PF,
        1.12 * scale_lc * PF,
        1.14 * PF,
        5.00 * scale_lc * PF,
        1.43 * scale_lc * PF,
        0.67 * scale_linear * PF,
        0.54 * scale_lc * PF
    };
    seeds.push_back(scaled_lc);

    std::array<double, GENE_COUNT> scaled_all_lc = {
        0.92 * PF,
        5.00 * scale_lc * PF,
        1.12 * scale_lc * PF,
        1.14 * scale_lc * PF,
        5.00 * scale_lc * PF,
        1.43 * scale_lc * PF,
        0.67 * scale_lc * PF,
        0.54 * scale_lc * PF
    };
    seeds.push_back(scaled_all_lc);

    seeds.push_back(make_relation_seed(band, fixed,
                                       0.67 * scale_linear,
                                       lower_target_zero_ratio(),
                                       upper_target_zero_ratio()));
    seeds.push_back(make_relation_seed(band, fixed,
                                       0.55 * scale_linear,
                                       0.64,
                                       1.62));
    seeds.push_back(make_relation_seed(band, fixed,
                                       0.85 * scale_linear,
                                       0.54,
                                       1.35));

    for (auto& seed : seeds) {
        repair_candidate(seed, bounds);
    }
    return seeds;
}

Candidate make_random_candidate(std::mt19937_64& rng,
                                const SearchBounds& bounds) {
    Candidate c;
    for (int i = 0; i < GENE_COUNT; ++i) {
        std::uniform_real_distribution<double> dist(bounds.lower[i], bounds.upper[i]);
        c.gene[i] = dist(rng);
    }
    repair_candidate(c.gene, bounds);
    return c;
}

Candidate make_seeded_candidate(const std::array<double, GENE_COUNT>& seed,
                                double jitter,
                                std::mt19937_64& rng,
                                const SearchBounds& bounds) {
    Candidate c;
    c.gene = seed;
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i < GENE_COUNT; ++i) {
        const double span = bounds.upper[i] - bounds.lower[i];
        c.gene[i] += jitter * span * nd(rng);
    }
    repair_candidate(c.gene, bounds);
    return c;
}

void apply_resonance_mutation(std::array<double, GENE_COUNT>& gene,
                              const BandSpec& band,
                              const FixedValues& fixed,
                              std::mt19937_64& rng) {
    std::uniform_real_distribution<double> urand(0.0, 1.0);
    std::uniform_real_distribution<double> lower_ratio(0.50, 0.76);
    std::uniform_real_distribution<double> upper_ratio(1.15, 1.90);

    if (urand(rng) < 0.50) {
        const double total = cap_for_tz(lower_ratio(rng) * band.center_hz, fixed.l3);
        std::normal_distribution<double> split_dist(0.82, 0.06);
        const double split = clamp_value(split_dist(rng), 0.62, 0.92);
        gene[C3] = total * split;
        gene[C4] = total * (1.0 - split);
    }
    if (urand(rng) < 0.50) {
        const double total = cap_for_tz(lower_ratio(rng) * band.center_hz, fixed.l4);
        std::normal_distribution<double> split_dist(0.78, 0.07);
        const double split = clamp_value(split_dist(rng), 0.58, 0.91);
        gene[C6] = total * split;
        gene[C7] = total * (1.0 - split);
    }
    if (urand(rng) < 0.65) {
        gene[C9] = cap_for_tz(upper_ratio(rng) * band.center_hz, fixed.l2);
    }
}

const Candidate& tournament_select(const std::vector<Candidate>& population,
                                   int tournament_size,
                                   std::mt19937_64& rng) {
    std::uniform_int_distribution<int> pick(
        0, static_cast<int>(population.size()) - 1);
    int best = pick(rng);
    for (int i = 1; i < tournament_size; ++i) {
        const int idx = pick(rng);
        if (population[idx].fitness < population[best].fitness) {
            best = idx;
        }
    }
    return population[best];
}

Candidate crossover_and_mutate(const Candidate& a,
                               const Candidate& b,
                               const BandSpec& band,
                               const FixedValues& fixed,
                               const SearchBounds& bounds,
                               const GAConfig& cfg,
                               int generation,
                               std::mt19937_64& rng) {
    Candidate child;
    std::uniform_real_distribution<double> urand(0.0, 1.0);
    std::normal_distribution<double> nrand(0.0, 1.0);

    const double progress = static_cast<double>(generation) /
                            static_cast<double>(std::max(1, cfg.max_generations));
    const double anneal = std::max(0.18, 1.0 - progress);

    for (int i = 0; i < GENE_COUNT; ++i) {
        double value = a.gene[i];
        if (urand(rng) < cfg.crossover_rate) {
            const double alpha = 1.30 * urand(rng) - 0.15;
            value = alpha * a.gene[i] + (1.0 - alpha) * b.gene[i];
        }
        if (urand(rng) < cfg.mutation_rate) {
            const double span = bounds.upper[i] - bounds.lower[i];
            value += nrand(rng) * span * cfg.mutation_scale * anneal;
        }
        if (urand(rng) < cfg.reset_mutation_rate) {
            std::uniform_real_distribution<double> reset(bounds.lower[i], bounds.upper[i]);
            value = reset(rng);
        }
        child.gene[i] = value;
    }

    if (urand(rng) < cfg.resonance_mutation_rate) {
        apply_resonance_mutation(child.gene, band, fixed, rng);
    }
    repair_candidate(child.gene, bounds);
    return child;
}

void evaluate_population(std::vector<Candidate>& population,
                         const FixedValues& fixed,
                         const BandSpec& band,
                         const FrequencyPlan& plan) {
    for (auto& candidate : population) {
        candidate.fitness = evaluate_candidate(candidate.gene, fixed, band, plan).fitness;
    }
}

std::string yes_no(bool value) {
    return value ? "YES" : "NO";
}

void print_progress(const BandSpec& band,
                    int restart,
                    int generation,
                    const Metrics& m) {
    std::lock_guard<std::mutex> lock(g_cout_mutex);
    std::cout << "[" << band.name << "] restart " << restart
              << " gen " << std::setw(4) << generation
              << " | score=" << std::scientific << std::setprecision(3) << m.fitness
              << std::fixed << std::setprecision(2)
              << " | pass RL min=" << m.min_pass_return_loss_db << " dB"
              << " | pass IL max=" << m.max_pass_insertion_loss_db << " dB"
              << " | low stop min=" << m.lower_stop_min_rejection_db << " dB"
              << " | upper stop min=" << m.upper_stop_min_rejection_db << " dB"
              << " | H2/H3 min=" << m.harmonic2_min_rejection_db
              << "/" << m.harmonic3_min_rejection_db << " dB"
              << " | ok=" << yes_no(m.constraints_ok) << "\n";
}

SearchResult run_ga_for_band(const BandSpec& band,
                             const FixedValues& fixed,
                             const GAConfig& cfg) {
    const auto started = std::chrono::steady_clock::now();
    const SearchBounds bounds = make_bounds(band, fixed);
    const FrequencyPlan coarse_plan = make_frequency_plan(band, false);
    const FrequencyPlan dense_plan = make_frequency_plan(band, true);
    const auto seeds = make_seed_bank(band, fixed, bounds);

    const std::uint64_t band_hash =
        static_cast<std::uint64_t>(std::hash<std::string>{}(band.name));
    std::mt19937_64 rng(cfg.seed ^ (band_hash + 0x9e3779b97f4a7c15ULL));

    SearchResult best_result;
    best_result.band_name = band.name;

    for (int restart = 1; restart <= cfg.max_restarts; ++restart) {
        std::vector<Candidate> population;
        population.reserve(static_cast<std::size_t>(cfg.population_size));

        for (const auto& seed : seeds) {
            population.push_back(make_seeded_candidate(seed, 0.00, rng, bounds));
            population.push_back(make_seeded_candidate(seed, 0.035, rng, bounds));
            population.push_back(make_seeded_candidate(seed, 0.090, rng, bounds));
        }
        while (static_cast<int>(population.size()) < cfg.population_size) {
            Candidate c = make_random_candidate(rng, bounds);
            if (population.size() % 3 == 0) {
                apply_resonance_mutation(c.gene, band, fixed, rng);
                repair_candidate(c.gene, bounds);
            }
            population.push_back(c);
        }

        evaluate_population(population, fixed, band, coarse_plan);
        std::sort(population.begin(), population.end(),
                  [](const Candidate& lhs, const Candidate& rhs) {
                      return lhs.fitness < rhs.fitness;
                  });

        int first_feasible_generation = -1;
        int last_best_generation = 0;

        for (int gen = 0; gen <= cfg.max_generations; ++gen) {
            if (gen % cfg.dense_check_interval == 0 || gen == cfg.max_generations) {
                const Metrics dense_metrics = evaluate_candidate(
                    population.front().gene, fixed, band, dense_plan);
                if (dense_metrics.valid &&
                    (dense_metrics.fitness < best_result.metrics.fitness ||
                     (dense_metrics.constraints_ok && !best_result.feasible))) {
                    best_result.best = population.front();
                    best_result.best.fitness = dense_metrics.fitness;
                    best_result.metrics = dense_metrics;
                    best_result.feasible = dense_metrics.constraints_ok;
                    best_result.restart = restart;
                    best_result.generation = gen;
                    last_best_generation = gen;
                }
                if (dense_metrics.constraints_ok && first_feasible_generation < 0) {
                    first_feasible_generation = gen;
                }

                const bool progress_tick = (gen % cfg.progress_interval == 0);
                const bool first_feasible_tick =
                    (dense_metrics.constraints_ok && first_feasible_generation == gen);
                if (progress_tick || first_feasible_tick) {
                    print_progress(band, restart, gen, dense_metrics);
                }

                const bool enough_polishing =
                    first_feasible_generation >= 0 &&
                    gen - first_feasible_generation >= cfg.min_generations_after_feasible;
                const bool stagnant =
                    gen - last_best_generation >= cfg.stagnation_generations;
                if (dense_metrics.constraints_ok && enough_polishing && stagnant) {
                    best_result.feasible = true;
                    best_result.stopped_by_solution = true;
                    best_result.restart = restart;
                    best_result.generation = gen;
                    const auto ended = std::chrono::steady_clock::now();
                    best_result.elapsed_seconds =
                        std::chrono::duration<double>(ended - started).count();
                    return best_result;
                }
            }

            if (gen == cfg.max_generations) {
                break;
            }

            std::vector<Candidate> next;
            next.reserve(static_cast<std::size_t>(cfg.population_size));
            const int elite_count = std::min(cfg.elite_count,
                                             static_cast<int>(population.size()));
            for (int i = 0; i < elite_count; ++i) {
                next.push_back(population[i]);
            }
            while (static_cast<int>(next.size()) < cfg.population_size) {
                const Candidate& p1 = tournament_select(population,
                                                        cfg.tournament_size, rng);
                const Candidate& p2 = tournament_select(population,
                                                        cfg.tournament_size, rng);
                next.push_back(crossover_and_mutate(p1, p2, band, fixed, bounds,
                                                    cfg, gen, rng));
            }

            population = std::move(next);
            evaluate_population(population, fixed, band, coarse_plan);
            std::sort(population.begin(), population.end(),
                      [](const Candidate& lhs, const Candidate& rhs) {
                          return lhs.fitness < rhs.fitness;
                      });
        }
    }

    const auto ended = std::chrono::steady_clock::now();
    best_result.elapsed_seconds =
        std::chrono::duration<double>(ended - started).count();
    best_result.feasible = best_result.metrics.constraints_ok;
    return best_result;
}

void print_component_table(const SearchResult& result, const FixedValues& fixed) {
    std::cout << "Optimized C2-C9 (pF):\n";
    for (int i = 0; i < GENE_COUNT; ++i) {
        std::cout << "  " << kGeneNames[i] << " = "
                  << std::fixed << std::setprecision(4)
                  << result.best.gene[i] / PF << " pF\n";
    }

    std::cout << "Fixed values:\n"
              << "  C1 = " << fixed.c1 / PF << " pF\n"
              << "  L1 = " << fixed.l1 / NH << " nH\n"
              << "  L2 = " << fixed.l2 / NH << " nH\n"
              << "  L3 = " << fixed.l3 / NH << " nH\n"
              << "  L4 = " << fixed.l4 / NH << " nH\n"
              << "  L5 = " << fixed.l5 / NH << " nH\n";
}

void print_condition_line(const std::string& name, bool ok, const std::string& detail) {
    std::cout << "  [" << (ok ? "OK " : "BAD") << "] "
              << name << " : " << detail << "\n";
}

std::string db_detail(double actual, const std::string& op, double target) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << actual << " dB " << op << " " << target << " dB";
    return oss.str();
}

void print_result_block(const SearchResult& result,
                        const FixedValues& fixed,
                        const BandSpec& band,
                        const std::string& title) {
    const Metrics& m = result.metrics;
    std::cout << "\n============================================================\n";
    std::cout << title << "\n";
    std::cout << "Band: " << result.band_name << " ("
              << std::fixed << std::setprecision(3)
              << band.pass_lo_hz / GHZ << "-"
              << band.pass_hi_hz / GHZ << " GHz)\n";
    std::cout << "Status: " << (result.feasible ? "FEASIBLE" : "BEST SO FAR")
              << " | stopped by solution: " << yes_no(result.stopped_by_solution)
              << " | restart/gen: " << result.restart << "/"
              << result.generation << " | elapsed: "
              << std::setprecision(2) << result.elapsed_seconds << " s\n";
    std::cout << "Fitness/score: " << std::scientific << std::setprecision(6)
              << m.fitness << std::fixed << "\n\n";

    print_component_table(result, fixed);

    std::cout << "\nCondition checks:\n";
    print_condition_line(
        "Passband return loss",
        m.pass_return_loss_ok,
        db_detail(m.min_pass_return_loss_db, ">=", band.pass_return_loss_min_db));
    print_condition_line(
        "Passband insertion loss",
        m.pass_insertion_loss_ok,
        db_detail(m.max_pass_insertion_loss_db, "<=", band.pass_insertion_loss_max_db));
    print_condition_line(
        "Lower stopband rejection",
        m.lower_stop_ok,
        db_detail(m.lower_stop_min_rejection_db, ">=", band.stop_rejection_min_db));
    print_condition_line(
        "Upper stopband rejection",
        m.upper_stop_ok,
        db_detail(m.upper_stop_min_rejection_db, ">=", band.stop_rejection_min_db));
    print_condition_line(
        "Second harmonic rejection",
        m.harmonic2_ok,
        db_detail(m.harmonic2_min_rejection_db, ">=", band.harmonic_rejection_min_db));
    print_condition_line(
        "Third harmonic rejection",
        m.harmonic3_ok,
        db_detail(m.harmonic3_min_rejection_db, ">=", band.harmonic_rejection_min_db));
    print_condition_line(
        "Transmission-zero ordering",
        m.transmission_zero_order_ok,
        "TZ1/TZ2 below passband, TZ3/TZ4 above passband");

    std::cout << "\nKey metrics:\n";
    std::cout << "  Pass RL min/avg       = " << std::setprecision(3)
              << m.min_pass_return_loss_db << " / "
              << m.avg_pass_return_loss_db << " dB\n";
    std::cout << "  Pass IL max/avg       = "
              << m.max_pass_insertion_loss_db << " / "
              << m.avg_pass_insertion_loss_db << " dB\n";
    std::cout << "  Center RL/IL          = "
              << m.center_return_loss_db << " / "
              << m.center_insertion_loss_db << " dB\n";
    std::cout << "  Edge RL low/high      = "
              << m.low_edge_return_loss_db << " / "
              << m.high_edge_return_loss_db << " dB\n";
    std::cout << "  Lower stop min rej    = "
              << m.lower_stop_min_rejection_db << " dB\n";
    std::cout << "  Upper stop min rej    = "
              << m.upper_stop_min_rejection_db << " dB\n";
    std::cout << "  H2/H3 min rejection   = "
              << m.harmonic2_min_rejection_db << " / "
              << m.harmonic3_min_rejection_db << " dB\n";
    std::cout << "  Deepest sampled notch = "
              << m.deepest_stop_rejection_db << " dB at "
              << m.deepest_stop_frequency_hz / GHZ << " GHz\n";

    std::cout << "\nFormula-based transmission zeros:\n";
    std::cout << "  TZ1(C3+C4,L3) = " << m.zeros.tz1_hz / GHZ << " GHz\n";
    std::cout << "  TZ2(C6+C7,L4) = " << m.zeros.tz2_hz / GHZ << " GHz\n";
    std::cout << "  TZ3(C9,L2)    = " << m.zeros.tz3_hz / GHZ << " GHz\n";
    std::cout << "  TZ4(C1,L1)    = " << m.zeros.tz4_hz / GHZ << " GHz\n";
}

class ResultPublisher {
public:
    ResultPublisher(const FixedValues& fixed, const std::vector<BandSpec>& bands)
        : fixed_(fixed), bands_(bands) {}

    void publish(const SearchResult& result) {
        std::lock_guard<std::mutex> lock(g_cout_mutex);
        const BandSpec& band = band_for_name(result.band_name);
        ++published_count_;
        std::ostringstream title;
        title << "Thread result " << published_count_ << " ("
              << result.band_name << ")";
        print_result_block(result, fixed_, band, title.str());
    }

private:
    const BandSpec& band_for_name(const std::string& name) const {
        for (const auto& band : bands_) {
            if (band.name == name) {
                return band;
            }
        }
        return bands_.front();
    }

    const FixedValues& fixed_;
    const std::vector<BandSpec>& bands_;
    int published_count_ = 0;
};

void print_startup(const FixedValues& fixed,
                   const std::vector<BandSpec>& bands,
                   const GAConfig& cfg) {
    std::lock_guard<std::mutex> lock(g_cout_mutex);
    std::cout << "GA search for Fig. 2 TGV-IPD BPF, optimizing C2-C9\n";
    std::cout << "Bands queued:";
    for (const auto& band : bands) {
        std::cout << " " << band.name;
    }
    std::cout << "\n";
    std::cout << "Passband return-loss requirement:\n";
    for (const auto& band : bands) {
        std::cout << "  " << band.name << " >= "
                  << band.pass_return_loss_min_db << " dB\n";
    }
    std::cout << "Passband insertion-loss requirement: <= 0.3 dB\n";
    std::cout << "Running one thread per band.\n\n";
    std::cout << "Fixed values: C1=" << fixed.c1 / PF
              << " pF, L1=" << fixed.l1 / NH
              << " nH, L2=" << fixed.l2 / NH
              << " nH, L3=" << fixed.l3 / NH
              << " nH, L4=" << fixed.l4 / NH
              << " nH, L5=" << fixed.l5 / NH << " nH\n";
    std::cout << "GA config: population=" << cfg.population_size
              << ", generations=" << cfg.max_generations
              << ", restarts=" << cfg.max_restarts
              << ", seed=" << cfg.seed << "\n";
    for (const auto& band : bands) {
        std::cout << band.name << " passband: " << band.pass_lo_hz / GHZ
                  << "-" << band.pass_hi_hz / GHZ << " GHz\n";
        std::cout << band.name << " stopbands: lower "
                  << band.lower_stop_lo_hz / GHZ
                  << "-" << band.lower_stop_hi_hz / GHZ
                  << " GHz, upper " << band.upper_stop_lo_hz / GHZ
                  << "-" << band.upper_stop_hi_hz / GHZ << " GHz\n";
    }
}

bool parse_int_arg(const char* text, int& out) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool parse_u64_arg(const char* text, std::uint64_t& out) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    out = static_cast<std::uint64_t>(value);
    return true;
}

GAConfig parse_config(int argc, char** argv) {
    GAConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string& name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--population") {
            parse_int_arg(need_value(arg), cfg.population_size);
        } else if (arg == "--generations") {
            parse_int_arg(need_value(arg), cfg.max_generations);
        } else if (arg == "--restarts") {
            parse_int_arg(need_value(arg), cfg.max_restarts);
        } else if (arg == "--seed") {
            parse_u64_arg(need_value(arg), cfg.seed);
        } else if (arg == "--polish-generations") {
            parse_int_arg(need_value(arg), cfg.min_generations_after_feasible);
        } else if (arg == "--stagnation-generations") {
            parse_int_arg(need_value(arg), cfg.stagnation_generations);
        } else if (arg == "--help") {
            std::cout << "Options: --population N --generations N "
                      << "--restarts N --seed N "
                      << "--polish-generations N --stagnation-generations N\n";
            std::exit(0);
        }
    }

    cfg.population_size = std::max(cfg.population_size, 60);
    cfg.max_generations = std::max(cfg.max_generations, 1);
    cfg.max_restarts = std::max(cfg.max_restarts, 1);
    cfg.min_generations_after_feasible =
        std::max(cfg.min_generations_after_feasible, 0);
    cfg.stagnation_generations = std::max(cfg.stagnation_generations, 0);
    cfg.elite_count = clamp_value(cfg.elite_count, 2, cfg.population_size / 3);
    return cfg;
}

} // namespace

int main(int argc, char** argv) {
    const FixedValues fixed;
    /*
    const BandSpec n77 = [] {
        BandSpec band = make_band("N77", 3.30 * GHZ, 4.20 * GHZ);
        band.pass_return_loss_min_db = 17.8;
        band.lower_stop_hi_hz = 3.20 * GHZ;
        band.upper_stop_lo_hz = 4.30 * GHZ;
        return band;
    }();
    */
    const BandSpec n78 = [] {
        BandSpec band = make_band("N78", 3.30 * GHZ, 3.80 * GHZ);
        band.pass_return_loss_min_db = 17.8;
        band.lower_stop_hi_hz = 3.20 * GHZ;
        band.upper_stop_lo_hz = 3.90 * GHZ;
        return band;
    }();
    /*
    const BandSpec n79 = [] {
        BandSpec band = make_band("N79", 4.40 * GHZ, 5.00 * GHZ);
        band.pass_return_loss_min_db = 17.8;
        band.lower_stop_hi_hz = 4.30 * GHZ;
        band.upper_stop_lo_hz = 5.10 * GHZ;
        return band;
    }();
    */
    const std::vector<BandSpec> bands = {n78};
    const GAConfig cfg = parse_config(argc, argv);

    print_startup(fixed, bands, cfg);

    ResultPublisher publisher(fixed, bands);
    std::vector<std::thread> threads;
    threads.reserve(bands.size());
    for (const auto& band : bands) {
        threads.emplace_back([&publisher, &fixed, &cfg, band] {
            const SearchResult result = run_ga_for_band(band, fixed, cfg);
            publisher.publish(result);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    return 0;
}

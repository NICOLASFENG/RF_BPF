#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double PF = 1e-12;
constexpr double NH = 1e-9;
constexpr double GHZ = 1e9;
constexpr double Z0 = 50.0;
constexpr double CAP_STEP = 0.005 * PF;
constexpr double INF = 1e100;
constexpr double TINY_DELTA_LOW = 0.005 * PF;
constexpr double TINY_DELTA_HIGH = 0.050 * PF;
constexpr double LARGE_DELTA = 2.000 * PF;

using Complex = std::complex<double>;

enum ModeIndex {
    MODE_N77 = 0,
    MODE_N78 = 1,
    MODE_N79 = 2,
    MODE_COUNT = 3
};

enum CellIndex {
    CELL_C1 = 0,
    CELL_C2 = 1,
    CELL_C3 = 2,
    CELL_C4 = 3,
    CELL_C5 = 4,
    CELL_C6 = 5,
    CELL_C7 = 6,
    CELL_C8 = 7,
    CELL_C9 = 8,
    CELL_COUNT = 9
};

enum SharePattern {
    SHARE_NONE = 0,
    SHARE_77_78 = 1,
    SHARE_77_79 = 2,
    SHARE_78_79 = 3,
    SHARE_ALL = 4
};

enum EvalTier {
    EVAL_COARSE = 0,
    EVAL_MID = 1,
    EVAL_DENSE = 2,
    EVAL_TIER_COUNT = 3
};

enum OptimizerKind {
    OPT_GA = 0,
    OPT_JDE = 1,
    OPT_DE = 2,
    OPT_HYBRID = 3
};

constexpr std::array<const char*, MODE_COUNT> kModeNames = {
    "N77", "N78", "N79"
};

constexpr std::array<const char*, CELL_COUNT> kCellNames = {
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"
};

using CapRow = std::array<double, CELL_COUNT>;
using CapTable = std::array<CapRow, MODE_COUNT>;

struct FixedValues {
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

    double pass_return_loss_min_db = 18.0;
    double pass_insertion_loss_max_db = 0.20;
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

struct Bounds {
    std::array<double, CELL_COUNT> lower{};
    std::array<double, CELL_COUNT> upper{};
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

struct ModeMetrics {
    bool valid = true;
    bool pass_return_loss_ok = false;
    bool pass_insertion_loss_ok = false;
    bool lower_stop_ok = false;
    bool upper_stop_ok = false;
    bool harmonic2_ok = false;
    bool harmonic3_ok = false;
    bool transmission_zero_order_ok = false;
    bool constraints_ok = false;

    double rf_penalty = INF;
    double rf_objective = INF;
    double min_pass_return_loss_db = INF;
    double max_pass_insertion_loss_db = -INF;
    double avg_pass_return_loss_db = 0.0;
    double avg_pass_insertion_loss_db = 0.0;
    double center_return_loss_db = 0.0;
    double center_insertion_loss_db = 0.0;
    double lower_stop_min_rejection_db = INF;
    double upper_stop_min_rejection_db = INF;
    double harmonic2_min_rejection_db = INF;
    double harmonic3_min_rejection_db = INF;
    double deepest_stop_rejection_db = -INF;
    double deepest_stop_frequency_hz = 0.0;
    double rf_margin = -INF;

    TransmissionZeros zeros;
};

struct SharingReport {
    int total_unique_count = 0;
    int sharing_cost = 0;
    int all_three_count = 0;
    int pair_77_78_count = 0;
    int pair_77_79_count = 0;
    int pair_78_79_count = 0;
    std::array<int, CELL_COUNT> unique_count{};
    std::array<SharePattern, CELL_COUNT> pattern{};
};

struct CellDecomposition {
    int cell = 0;
    std::array<double, MODE_COUNT> values{};
    std::array<int, MODE_COUNT> sorted_modes{};
    std::array<double, MODE_COUNT> sorted_values{};
    double fixed_f = 0.0;
    double delta_a_f = 0.0;
    double delta_b_f = 0.0;
    int branch_count = 0;
    int tiny_branch_count = 0;
    int large_branch_count = 0;
    double cost = 0.0;
};

struct HardwareReport {
    std::array<CellDecomposition, CELL_COUNT> cells{};
    double total_cost = 0.0;
    double total_fixed_f = 0.0;
    double total_increment_f = 0.0;
    double max_increment_f = 0.0;
    int nonzero_branch_count = 0;
    int tiny_branch_count = 0;
    int large_branch_count = 0;
    int huge_branch_count = 0;
};

struct Evaluation {
    bool valid = true;
    bool all_modes_feasible = false;
    bool rf_only_rank = false;
    double fitness = INF;
    double rf_penalty = INF;
    double rf_objective = INF;
    double worst_rf_violation = INF;
    double rf_margin = -INF;
    double sharing_cost = INF;
    double hardware_cost = INF;
    double baseline_distance_cost = INF;
    int total_unique_count = 0;
    std::array<ModeMetrics, MODE_COUNT> modes{};
    SharingReport sharing;
    HardwareReport hardware;
};

struct Candidate {
    CapTable cap{};
    Evaluation eval;
};

struct MaskConstraints {
    bool enabled = false;
    std::array<SharePattern, CELL_COUNT> pattern{};
    std::array<bool, CELL_COUNT> specified{};
};

struct Config {
    int population_size = 1800;
    int max_generations = 4000;
    int max_restarts = 8;
    int elite_count = 40;
    int tournament_size = 4;
    int dense_check_interval = 25;
    int elite_dense_interval = 5;
    int mid_check_interval = 5;
    int mid_top_count = 140;
    int dense_top_count = 48;
    int progress_interval = 100;
    int archive_top_scan = 80;
    int archive_per_sharing = 8;
    int threads = 0;
    double crossover_rate = 0.92;
    double mutation_rate = 0.30;
    double reset_mutation_rate = 0.04;
    double compression_mutation_rate = 0.08;
    double decompression_mutation_rate = 0.05;
    double mutation_scale = 0.08;
    double lambda_compression = 10.0;
    double lambda_hardware = 1.0;
    double lambda_base = 0.0;   // was 0.20
    double local_jitter_scale = 1.0;
    bool require_feasible_output = true;
    bool rf_only = false;
    bool two_stage = false;
    bool semi_random_init = true;
    double init_seed_fraction = 0.15;
    double init_local_jitter_fraction = 0.15;
    double init_wide_jitter_fraction = 0.25;
    double init_random_fraction = 0.35;
    double init_share_pattern_fraction = 0.10;
    double min_random_fraction = 0.25;
    double local_jitter_sigma = 0.05;
    double wide_jitter_sigma = 0.35;
    double random_seed_blend_min = 0.0;
    double random_seed_blend_max = 0.20;
    int stagnation_threshold = 320;
    double stagnation_inject_fraction = 0.30;
    bool jde_mode = false;
    double jde_F_init = 0.5;
    double jde_CR_init = 0.9;
    double jde_tau1 = 0.1;
    double jde_tau2 = 0.1;
    OptimizerKind optimizer = OPT_GA;
    bool island_model = false;
    int islands = 4;
    int migration_interval = 50;
    int migration_count = 2;
    bool lhs_random_init = false;
    bool progressive_sharing = false;
    int pattern_failure_limit = 3;
    bool pattern_first = false;
    bool local_polish = false;
    int local_polish_steps = 180;
    bool enable_early_exit = false;
    bool benchmark_short = false;
    bool use_share_patterns = false;
    bool sweep_lambda_share = false;
    bool eval_baseline_only = false;
    bool single_cell_mask_scan = false;
    bool legacy_fixed_c1_c5 = false;
    bool quiet = false;
    bool lambda_base_explicit = false;
    std::string baseline_csv_path;
    std::string output_prefix = "ctc_ga";
    std::uint64_t seed = 20260503ULL;
    MaskConstraints mask;
};

struct RuntimeCounters {
    std::uint64_t coarse_evaluations = 0;
    std::uint64_t mid_evaluations = 0;
    std::uint64_t dense_evaluations = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t local_repair_calls = 0;
    std::uint64_t accepted_sharing_moves = 0;
    std::uint64_t rolled_back_sharing_moves = 0;
    std::uint64_t feasible_candidates_found = 0;

    void add(const RuntimeCounters& other) {
        coarse_evaluations += other.coarse_evaluations;
        mid_evaluations += other.mid_evaluations;
        dense_evaluations += other.dense_evaluations;
        cache_hits += other.cache_hits;
        local_repair_calls += other.local_repair_calls;
        accepted_sharing_moves += other.accepted_sharing_moves;
        rolled_back_sharing_moves += other.rolled_back_sharing_moves;
        feasible_candidates_found += other.feasible_candidates_found;
    }
};

struct PopulationComposition {
    int seed = 0;
    int local_jitter = 0;
    int wide_jitter = 0;
    int random_full = 0;
    int random_blend = 0;
    int share_pattern = 0;

    int total() const {
        return seed + local_jitter + wide_jitter +
               random_full + random_blend + share_pattern;
    }
};

struct SearchResult {
    double lambda_compression = 0.0;
    Candidate best;
    Candidate baseline;
    std::map<int, std::vector<Candidate>> archive_by_sharing;
    Candidate best_rf_feasible;
    bool has_best = false;
    bool has_best_rf_feasible = false;
    int restart = 0;
    int generation = 0;
    double elapsed_seconds = 0.0;
    RuntimeCounters counters;
};

double sqr(double x) {
    return x * x;
}

double clamp_value(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

bool finite_positive(double x) {
    return std::isfinite(x) && x > 0.0;
}

double pf(double value_f) {
    return value_f / PF;
}

double ghz(double value_hz) {
    return value_hz / GHZ;
}

double db_from_magnitude(double mag) {
    return -20.0 * std::log10(std::max(mag, 1e-15));
}

[[maybe_unused]] double cap_for_tz(double f_hz, double inductance_h) {
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

std::string yes_no(bool value) {
    return value ? "YES" : "NO";
}

std::string pattern_name(SharePattern pattern) {
    switch (pattern) {
    case SHARE_NONE:
        return "ALL_DIFFERENT";
    case SHARE_77_78:
        return "N77_N78_SHARED";
    case SHARE_77_79:
        return "N77_N79_SHARED";
    case SHARE_78_79:
        return "N78_N79_SHARED";
    case SHARE_ALL:
        return "ALL_SHARED";
    }
    return "UNKNOWN";
}

std::string mode_logic_string(const CellDecomposition& d) {
    std::ostringstream oss;
    oss << "FIX=" << kModeNames[d.sorted_modes[0]];
    if (d.delta_a_f > TINY_DELTA_LOW) {
        oss << ", +A -> " << kModeNames[d.sorted_modes[1]];
    }
    if (d.delta_b_f > TINY_DELTA_LOW) {
        oss << ", +B -> " << kModeNames[d.sorted_modes[2]];
    }
    return oss.str();
}

int quantized_cap_code(double c_f) {
    return static_cast<int>(std::llround(c_f / CAP_STEP));
}

double quantize_cap(double c_f) {
    return static_cast<double>(quantized_cap_code(c_f)) * CAP_STEP;
}

int unique_count_3(int a, int b, int c) {
    int n = 1;
    if (b != a) {
        ++n;
    }
    if (c != a && c != b) {
        ++n;
    }
    return n;
}

BandSpec make_band(std::string name, double pass_lo_hz, double pass_hi_hz) {
    BandSpec band;
    band.name = std::move(name);
    band.pass_lo_hz = pass_lo_hz;
    band.pass_hi_hz = pass_hi_hz;
    band.center_hz = 0.5 * (pass_lo_hz + pass_hi_hz);

    band.lower_stop_hi_hz = std::max(0.05 * GHZ, pass_lo_hz - 0.10 * GHZ);
    band.upper_stop_lo_hz = pass_hi_hz + 0.10 * GHZ;
    band.upper_stop_hi_hz = 15.0 * GHZ;

    const double harmonic_window = 0.035;
    band.harmonic2_lo_hz = 2.0 * band.center_hz * (1.0 - harmonic_window);
    band.harmonic2_hi_hz = 2.0 * band.center_hz * (1.0 + harmonic_window);
    band.harmonic3_lo_hz = 3.0 * band.center_hz * (1.0 - harmonic_window);
    band.harmonic3_hi_hz = std::min(15.0 * GHZ,
                                    3.0 * band.center_hz * (1.0 + harmonic_window));
    return band;
}

std::array<BandSpec, MODE_COUNT> make_bands() {
    std::array<BandSpec, MODE_COUNT> bands = {{
        make_band("N77", 3.30 * GHZ, 4.20 * GHZ),
        make_band("N78", 3.30 * GHZ, 3.80 * GHZ),
        make_band("N79", 4.40 * GHZ, 5.00 * GHZ)
    }};
    bands[MODE_N77].lower_stop_hi_hz = 3.20 * GHZ;
    bands[MODE_N77].upper_stop_lo_hz = 4.30 * GHZ;
    bands[MODE_N78].lower_stop_hi_hz = 3.20 * GHZ;
    bands[MODE_N78].upper_stop_lo_hz = 3.90 * GHZ;
    bands[MODE_N79].lower_stop_hi_hz = 4.30 * GHZ;
    bands[MODE_N79].upper_stop_lo_hz = 5.10 * GHZ;
    return bands;
}

FrequencyPlan make_frequency_plan_tier(const BandSpec& band, EvalTier tier) {
    FrequencyPlan plan;
    const bool dense = tier == EVAL_DENSE;
    const bool mid = tier == EVAL_MID;
    plan.pass = linspace(band.pass_lo_hz, band.pass_hi_hz,
                         dense ? 111 : (mid ? 63 : 31));
    plan.lower_stop = linspace(band.lower_stop_lo_hz, band.lower_stop_hi_hz,
                               dense ? 111 : (mid ? 63 : 31));
    plan.upper_stop = linspace(band.upper_stop_lo_hz, band.upper_stop_hi_hz,
                               dense ? 171 : (mid ? 91 : 45));
    plan.harmonic2 = linspace(band.harmonic2_lo_hz, band.harmonic2_hi_hz,
                              dense ? 31 : (mid ? 17 : 9));
    plan.harmonic3 = linspace(band.harmonic3_lo_hz, band.harmonic3_hi_hz,
                              dense ? 31 : (mid ? 17 : 9));
    return plan;
}

std::array<FrequencyPlan, MODE_COUNT> make_frequency_plans(
    const std::array<BandSpec, MODE_COUNT>& bands,
    bool dense) {
    return [&]() {
        std::array<FrequencyPlan, MODE_COUNT> plans;
        const EvalTier tier = dense ? EVAL_DENSE : EVAL_COARSE;
        for (int mode = 0; mode < MODE_COUNT; ++mode) {
            plans[mode] = make_frequency_plan_tier(bands[mode], tier);
        }
        return plans;
    }();
}

std::array<FrequencyPlan, MODE_COUNT> make_frequency_plans_tier(
    const std::array<BandSpec, MODE_COUNT>& bands,
    EvalTier tier) {
    std::array<FrequencyPlan, MODE_COUNT> plans;
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        plans[mode] = make_frequency_plan_tier(bands[mode], tier);
    }
    return plans;
}

Bounds make_bounds() {
    Bounds bounds;
    bounds.lower = {{
        0.05 * PF, 0.25 * PF, 0.50 * PF, 0.20 * PF, 0.30 * PF,
        0.50 * PF, 0.30 * PF, 0.15 * PF, 0.20 * PF
    }};
    bounds.upper = {{
        0.80 * PF, 2.50 * PF, 6.50 * PF, 2.50 * PF, 2.50 * PF,
        6.50 * PF, 2.80 * PF, 1.20 * PF, 1.20 * PF
    }};
    return bounds;
}

CapTable make_baseline_table() {
    return {{
        {{0.20 * PF, 1.49 * PF, 5.00 * PF, 1.12 * PF, 1.14 * PF,
          5.00 * PF, 1.43 * PF, 0.67 * PF, 0.54 * PF}},
        {{0.46 * PF, 1.76 * PF, 4.18 * PF, 0.78 * PF, 1.87 * PF,
          2.91 * PF, 0.64 * PF, 0.92 * PF, 0.37 * PF}},
        {{0.31 * PF, 2.08 * PF, 1.27 * PF, 2.21 * PF, 0.74 * PF,
          5.62 * PF, 2.36 * PF, 0.58 * PF, 1.03 * PF}}
    }};
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

SParameters calculate_s_parameters(const CapRow& cap_row,
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

    // 0 = port 1, 8 = port 2, 1..7 are internal nodes.
    // L1-L5 remain fixed; all C1-C9 values come from the mode table.
    cap(0, 1, cap_row[CELL_C1]);
    ind(0, 1, fixed.l1);
    cap(1, 2, cap_row[CELL_C2]);
    cap(2, 3, cap_row[CELL_C3]);
    ind(3, -1, fixed.l3);
    cap(3, 6, cap_row[CELL_C4]);
    ind(6, -1, fixed.l5);
    cap(6, 5, cap_row[CELL_C7]);
    cap(4, 5, cap_row[CELL_C6]);
    ind(5, -1, fixed.l4);
    cap(2, 4, cap_row[CELL_C5]);
    cap(4, 7, cap_row[CELL_C8]);
    cap(7, 8, cap_row[CELL_C9]);
    ind(7, 8, fixed.l2);

    constexpr std::array<int, 2> ports = {{0, 8}};
    constexpr std::array<int, 7> internal = {{1, 2, 3, 4, 5, 6, 7}};

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
            const Complex identity = (r == c)
                ? Complex(1.0, 0.0)
                : Complex(0.0, 0.0);
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

    if (!std::isfinite(std::abs(s[0][0])) ||
        !std::isfinite(std::abs(s[1][0]))) {
        return {};
    }
    return {s[0][0], s[1][0], true};
}

TransmissionZeros compute_transmission_zeros(const CapRow& cap_row,
                                             const FixedValues& fixed) {
    TransmissionZeros z;
    z.tz1_hz = tz_from_lc(cap_row[CELL_C3] + cap_row[CELL_C4], fixed.l3);
    z.tz2_hz = tz_from_lc(cap_row[CELL_C6] + cap_row[CELL_C7], fixed.l4);
    z.tz3_hz = tz_from_lc(cap_row[CELL_C9], fixed.l2);
    z.tz4_hz = tz_from_lc(cap_row[CELL_C1], fixed.l1);
    return z;
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

double soft_target(double x, double target, double width) {
    return sqr((x - target) / width);
}

double lower_target_zero_ratio() {
    return 2.20 / 3.75;
}

double upper_target_zero_ratio() {
    return 5.59 / 3.75;
}

ModeMetrics evaluate_mode(const CapRow& cap_row,
                          const FixedValues& fixed,
                          const BandSpec& band,
                          const FrequencyPlan& plan,
                          double penalty_cutoff = INF,
                          bool allow_early_exit = false) {
    ModeMetrics m;
    m.zeros = compute_transmission_zeros(cap_row, fixed);
    double penalty = 0.0;
    double objective = 0.0;
    bool stopped_early = false;

    auto invalid = [&]() {
        m.valid = false;
        m.constraints_ok = false;
        m.rf_penalty = INF;
        m.rf_objective = INF;
        return m;
    };
    auto early = [&]() {
        m.constraints_ok = false;
        m.rf_penalty = penalty;
        m.rf_objective = objective;
        stopped_early = true;
        return m;
    };
    auto should_stop = [&]() {
        return allow_early_exit &&
               std::isfinite(penalty_cutoff) &&
               penalty > penalty_cutoff;
    };

    double pass_rl_sum = 0.0;
    double pass_il_sum = 0.0;
    int pass_count = 0;
    for (double f : plan.pass) {
        const auto sp = calculate_s_parameters(cap_row, fixed, f);
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

        penalty += 4500.0 * sqr(std::max(0.0, band.pass_return_loss_min_db - rl_db));
        penalty += 2800.0 * sqr(std::max(0.0, il_db - band.pass_insertion_loss_max_db));
        if (should_stop()) {
            return early();
        }
    }

    if (pass_count == 0) {
        return invalid();
    }
    m.avg_pass_return_loss_db = pass_rl_sum / static_cast<double>(pass_count);
    m.avg_pass_insertion_loss_db = pass_il_sum / static_cast<double>(pass_count);

    const auto center_sp = calculate_s_parameters(cap_row, fixed, band.center_hz);
    if (center_sp.valid) {
        m.center_return_loss_db = db_from_magnitude(std::abs(center_sp.s11));
        m.center_insertion_loss_db = db_from_magnitude(std::abs(center_sp.s21));
    }

    auto scan_rejection = [&](const std::vector<double>& frequencies,
                              double target_db,
                              double weight,
                              double& min_rejection_db) {
        min_rejection_db = frequencies.empty() ? INF : INF;
        for (double f : frequencies) {
            if (stopped_early) {
                return;
            }
            const auto sp = calculate_s_parameters(cap_row, fixed, f);
            if (!sp.valid) {
                m.valid = false;
                continue;
            }
            const double rejection_db = db_from_magnitude(std::abs(sp.s21));
            if (!std::isfinite(rejection_db)) {
                m.valid = false;
                continue;
            }
            min_rejection_db = std::min(min_rejection_db, rejection_db);
            if (rejection_db > m.deepest_stop_rejection_db) {
                m.deepest_stop_rejection_db = rejection_db;
                m.deepest_stop_frequency_hz = f;
            }
            penalty += weight * sqr(std::max(0.0, target_db - rejection_db));
            if (should_stop()) {
                early();
                return;
            }
        }
    };

    scan_rejection(plan.lower_stop, band.stop_rejection_min_db, 850.0,
                   m.lower_stop_min_rejection_db);
    if (stopped_early) {
        return m;
    }
    scan_rejection(plan.upper_stop, band.stop_rejection_min_db, 850.0,
                   m.upper_stop_min_rejection_db);
    if (stopped_early) {
        return m;
    }
    scan_rejection(plan.harmonic2, band.harmonic_rejection_min_db, 1400.0,
                   m.harmonic2_min_rejection_db);
    if (stopped_early) {
        return m;
    }
    scan_rejection(plan.harmonic3, band.harmonic_rejection_min_db, 1400.0,
                   m.harmonic3_min_rejection_db);
    if (stopped_early) {
        return m;
    }

    if (!m.valid) {
        return invalid();
    }

    const double r1 = m.zeros.tz1_hz / band.center_hz;
    const double r2 = m.zeros.tz2_hz / band.center_hz;
    const double r3 = m.zeros.tz3_hz / band.center_hz;
    const double r4 = m.zeros.tz4_hz / band.center_hz;

    penalty += 600.0 * sqr(outside_window(r1, 0.45, 0.82));
    penalty += 600.0 * sqr(outside_window(r2, 0.45, 0.82));
    penalty += 450.0 * sqr(outside_window(r3, 1.05, 2.10));
    objective += 3.0 * soft_target(r1, lower_target_zero_ratio(), 0.18);
    objective += 3.0 * soft_target(r2, lower_target_zero_ratio(), 0.18);
    objective += 2.0 * soft_target(r3, upper_target_zero_ratio(), 0.25);

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

    m.rf_margin = INF;
    m.rf_margin = std::min(m.rf_margin,
                           m.min_pass_return_loss_db -
                           band.pass_return_loss_min_db);
    m.rf_margin = std::min(m.rf_margin,
                           band.pass_insertion_loss_max_db -
                           m.max_pass_insertion_loss_db);
    m.rf_margin = std::min(m.rf_margin,
                           m.lower_stop_min_rejection_db -
                           band.stop_rejection_min_db);
    m.rf_margin = std::min(m.rf_margin,
                           m.upper_stop_min_rejection_db -
                           band.stop_rejection_min_db);
    m.rf_margin = std::min(m.rf_margin,
                           m.harmonic2_min_rejection_db -
                           band.harmonic_rejection_min_db);
    m.rf_margin = std::min(m.rf_margin,
                           m.harmonic3_min_rejection_db -
                           band.harmonic_rejection_min_db);
    m.rf_margin = std::min(m.rf_margin,
                           (band.pass_lo_hz - m.zeros.tz1_hz) / GHZ);
    m.rf_margin = std::min(m.rf_margin,
                           (band.pass_lo_hz - m.zeros.tz2_hz) / GHZ);
    m.rf_margin = std::min(m.rf_margin,
                           (m.zeros.tz3_hz - band.pass_hi_hz) / GHZ);
    m.rf_margin = std::min(m.rf_margin,
                           (m.zeros.tz4_hz - band.pass_hi_hz) / GHZ);
    m.rf_margin = std::min(m.rf_margin, r4 - 1.40);

    objective += 0.25 * std::max(0.0, 22.0 - m.min_pass_return_loss_db);
    objective += 8.0 * m.max_pass_insertion_loss_db;
    objective += 1.5 * m.avg_pass_insertion_loss_db;
    objective += 0.05 * std::max(0.0, 35.0 - m.lower_stop_min_rejection_db);
    objective += 0.05 * std::max(0.0, 35.0 - m.upper_stop_min_rejection_db);
    objective += 0.04 * std::max(0.0, 40.0 - m.harmonic2_min_rejection_db);
    objective += 0.04 * std::max(0.0, 40.0 - m.harmonic3_min_rejection_db);

    m.rf_penalty = penalty;
    m.rf_objective = objective;
    return m;
}

SharingReport compute_sharing_report(const CapTable& cap) {
    SharingReport report;
    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        const int a = quantized_cap_code(cap[MODE_N77][cell]);
        const int b = quantized_cap_code(cap[MODE_N78][cell]);
        const int c = quantized_cap_code(cap[MODE_N79][cell]);
        const int unique = unique_count_3(a, b, c);
        report.unique_count[cell] = unique;
        report.total_unique_count += unique;
        report.sharing_cost += unique - 1;

        if (a == b) {
            ++report.pair_77_78_count;
        }
        if (a == c) {
            ++report.pair_77_79_count;
        }
        if (b == c) {
            ++report.pair_78_79_count;
        }

        if (a == b && b == c) {
            ++report.all_three_count;
            report.pattern[cell] = SHARE_ALL;
        } else if (a == b) {
            report.pattern[cell] = SHARE_77_78;
        } else if (a == c) {
            report.pattern[cell] = SHARE_77_79;
        } else if (b == c) {
            report.pattern[cell] = SHARE_78_79;
        } else {
            report.pattern[cell] = SHARE_NONE;
        }
    }
    return report;
}

double tiny_delta_penalty(double delta_f) {
    if (delta_f <= TINY_DELTA_LOW || delta_f >= TINY_DELTA_HIGH) {
        return 0.0;
    }
    const double normalized = (TINY_DELTA_HIGH - delta_f) /
                              (TINY_DELTA_HIGH - TINY_DELTA_LOW);
    return sqr(normalized);
}

double large_delta_penalty(double delta_f) {
    return sqr(std::max(0.0, (delta_f - LARGE_DELTA) / PF));
}

HardwareReport compute_hardware_report(const CapTable& cap) {
    HardwareReport report;
    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        CellDecomposition d;
        d.cell = cell;
        for (int mode = 0; mode < MODE_COUNT; ++mode) {
            d.values[mode] = cap[mode][cell];
        }

        std::array<std::pair<double, int>, MODE_COUNT> sorted = {{
            {cap[MODE_N77][cell], MODE_N77},
            {cap[MODE_N78][cell], MODE_N78},
            {cap[MODE_N79][cell], MODE_N79}
        }};
        std::stable_sort(sorted.begin(), sorted.end(),
                         [](const auto& lhs, const auto& rhs) {
                             if (lhs.first != rhs.first) {
                                 return lhs.first < rhs.first;
                             }
                             return lhs.second < rhs.second;
                         });

        for (int i = 0; i < MODE_COUNT; ++i) {
            d.sorted_values[i] = sorted[i].first;
            d.sorted_modes[i] = sorted[i].second;
        }
        d.fixed_f = d.sorted_values[0];
        d.delta_a_f = std::max(0.0, d.sorted_values[1] - d.sorted_values[0]);
        d.delta_b_f = std::max(0.0, d.sorted_values[2] - d.sorted_values[1]);

        const std::array<double, 2> deltas = {{d.delta_a_f, d.delta_b_f}};
        double tiny_cost = 0.0;
        double large_cost = 0.0;
        for (double delta : deltas) {
            if (delta > TINY_DELTA_LOW) {
                ++d.branch_count;
                ++report.nonzero_branch_count;
            }
            if (delta > TINY_DELTA_LOW && delta < TINY_DELTA_HIGH) {
                ++d.tiny_branch_count;
                ++report.tiny_branch_count;
            }
            if (delta > 1.0 * PF) {
                ++d.large_branch_count;
                ++report.large_branch_count;
            }
            if (delta > LARGE_DELTA) {
                ++report.huge_branch_count;
            }
            tiny_cost += tiny_delta_penalty(delta);
            large_cost += large_delta_penalty(delta);
            report.max_increment_f = std::max(report.max_increment_f, delta);
        }

        const double increment = d.delta_a_f + d.delta_b_f;
        report.total_fixed_f += d.fixed_f;
        report.total_increment_f += increment;

        const double fixed_area_cost = d.fixed_f / PF;
        const double increment_area_cost = increment / PF;
        d.cost =
            0.20 * fixed_area_cost +
            0.60 * increment_area_cost +
            0.50 * static_cast<double>(d.branch_count) +
            2.00 * tiny_cost +
            3.00 * large_cost;
        report.total_cost += d.cost;
        report.cells[cell] = d;
    }
    return report;
}

double compute_baseline_distance_cost(const CapTable& cap,
                                      const CapTable& baseline) {
    double cost = 0.0;
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            const double scale = std::max(0.10 * PF, 0.25 * baseline[mode][cell]);
            cost += sqr((cap[mode][cell] - baseline[mode][cell]) / scale);
        }
    }
    return cost;
}

double mode_worst_rf_violation(const ModeMetrics& m, const BandSpec& band) {
    if (!m.valid) {
        return INF;
    }

    double worst = 0.0;
    auto add = [&](double violation) {
        if (!std::isfinite(violation)) {
            worst = INF;
        } else if (std::isfinite(worst)) {
            worst = std::max(worst, std::max(0.0, violation));
        }
    };

    add(band.pass_return_loss_min_db - m.min_pass_return_loss_db);
    add(m.max_pass_insertion_loss_db - band.pass_insertion_loss_max_db);
    add(band.stop_rejection_min_db - m.lower_stop_min_rejection_db);
    add(band.stop_rejection_min_db - m.upper_stop_min_rejection_db);
    add(band.harmonic_rejection_min_db - m.harmonic2_min_rejection_db);
    add(band.harmonic_rejection_min_db - m.harmonic3_min_rejection_db);

    add((m.zeros.tz1_hz - band.pass_lo_hz) / GHZ);
    add((m.zeros.tz2_hz - band.pass_lo_hz) / GHZ);
    add((band.pass_hi_hz - m.zeros.tz3_hz) / GHZ);
    add((band.pass_hi_hz - m.zeros.tz4_hz) / GHZ);
    if (band.center_hz > 0.0) {
        add(1.40 - m.zeros.tz4_hz / band.center_hz);
    }
    return worst;
}

Evaluation evaluate_candidate(const CapTable& cap,
                              const FixedValues& fixed,
                              const std::array<BandSpec, MODE_COUNT>& bands,
                              const std::array<FrequencyPlan, MODE_COUNT>& plans,
                              const CapTable& baseline,
                              const Config& cfg,
                              double penalty_cutoff = INF,
                              bool allow_early_exit = false) {
    Evaluation e;
    e.valid = true;
    e.rf_only_rank = cfg.rf_only;
    e.all_modes_feasible = true;
    e.rf_penalty = 0.0;
    e.rf_objective = 0.0;
    e.worst_rf_violation = 0.0;
    e.rf_margin = INF;

    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        const double remaining_cutoff =
            std::isfinite(penalty_cutoff)
            ? std::max(0.0, penalty_cutoff - e.rf_penalty)
            : INF;
        e.modes[mode] = evaluate_mode(cap[mode], fixed, bands[mode],
                                      plans[mode], remaining_cutoff,
                                      allow_early_exit);
        if (!e.modes[mode].valid) {
            e.valid = false;
        }
        e.rf_penalty += e.modes[mode].rf_penalty;
        e.rf_objective += e.modes[mode].rf_objective;
        e.rf_margin = std::min(e.rf_margin, e.modes[mode].rf_margin);
        e.worst_rf_violation =
            std::max(e.worst_rf_violation,
                     mode_worst_rf_violation(e.modes[mode], bands[mode]));
        e.all_modes_feasible = e.all_modes_feasible &&
                               e.modes[mode].constraints_ok;
        if (allow_early_exit &&
            std::isfinite(penalty_cutoff) &&
            e.rf_penalty > penalty_cutoff) {
            e.all_modes_feasible = false;
            break;
        }
    }

    e.sharing = compute_sharing_report(cap);
    e.total_unique_count = e.sharing.total_unique_count;
    e.sharing_cost = static_cast<double>(e.sharing.sharing_cost);
    e.hardware = compute_hardware_report(cap);
    e.hardware_cost = e.hardware.total_cost;
    e.baseline_distance_cost = compute_baseline_distance_cost(cap, baseline);

    if (!e.valid) {
        e.all_modes_feasible = false;
        e.fitness = INF;
        return e;
    }

    if (!e.all_modes_feasible) {
        e.fitness = e.rf_penalty + e.rf_objective;
    } else {
        e.fitness =
            e.rf_penalty +
            e.rf_objective +
            cfg.lambda_compression * e.sharing_cost +
            cfg.lambda_hardware * e.hardware_cost +
            cfg.lambda_base * e.baseline_distance_cost;
    }
    return e;
}

bool nearly_less(double a, double b, double eps = 1e-9) {
    return a < b - eps;
}

bool better_eval(const Evaluation& a, const Evaluation& b) {
    if (a.all_modes_feasible != b.all_modes_feasible) {
        return a.all_modes_feasible;
    }

    if (a.all_modes_feasible && b.all_modes_feasible) {
        if (a.rf_only_rank || b.rf_only_rank) {
            if (nearly_less(a.rf_penalty, b.rf_penalty)) {
                return true;
            }
            if (nearly_less(b.rf_penalty, a.rf_penalty)) {
                return false;
            }
            if (nearly_less(a.rf_objective, b.rf_objective)) {
                return true;
            }
            if (nearly_less(b.rf_objective, a.rf_objective)) {
                return false;
            }
            if (nearly_less(a.worst_rf_violation, b.worst_rf_violation)) {
                return true;
            }
            if (nearly_less(b.worst_rf_violation, a.worst_rf_violation)) {
                return false;
            }
            return a.fitness < b.fitness;
        }
        if (a.sharing.sharing_cost != b.sharing.sharing_cost) {
            return a.sharing.sharing_cost < b.sharing.sharing_cost;
        }
        if (nearly_less(a.hardware_cost, b.hardware_cost)) {
            return true;
        }
        if (nearly_less(b.hardware_cost, a.hardware_cost)) {
            return false;
        }
        if (nearly_less(b.rf_margin, a.rf_margin)) {
            return true;
        }
        if (nearly_less(a.rf_margin, b.rf_margin)) {
            return false;
        }
        if (nearly_less(a.rf_objective, b.rf_objective)) {
            return true;
        }
        if (nearly_less(b.rf_objective, a.rf_objective)) {
            return false;
        }
        return a.fitness < b.fitness;
    }

    if (nearly_less(a.rf_penalty, b.rf_penalty)) {
        return true;
    }
    if (nearly_less(b.rf_penalty, a.rf_penalty)) {
        return false;
    }
    if (nearly_less(a.rf_objective, b.rf_objective)) {
        return true;
    }
    if (nearly_less(b.rf_objective, a.rf_objective)) {
        return false;
    }
    if (nearly_less(a.worst_rf_violation, b.worst_rf_violation)) {
        return true;
    }
    if (nearly_less(b.worst_rf_violation, a.worst_rf_violation)) {
        return false;
    }
    return a.fitness < b.fitness;
}

bool better_candidate(const Candidate& a, const Candidate& b) {
    return better_eval(a.eval, b.eval);
}

struct CapKey {
    std::array<int, MODE_COUNT * CELL_COUNT> code{};

    bool operator==(const CapKey& other) const {
        return code == other.code;
    }
};

struct CapKeyHash {
    std::size_t operator()(const CapKey& key) const {
        std::size_t h = 1469598103934665603ULL;
        for (int v : key.code) {
            h ^= static_cast<std::size_t>(v + 0x9e3779b9);
            h *= 1099511628211ULL;
        }
        return h;
    }
};

CapKey make_cap_key(const CapTable& cap) {
    CapKey key;
    int idx = 0;
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            key.code[idx++] = quantized_cap_code(cap[mode][cell]);
        }
    }
    return key;
}

const char* tier_name(EvalTier tier) {
    switch (tier) {
    case EVAL_COARSE:
        return "coarse";
    case EVAL_MID:
        return "mid";
    case EVAL_DENSE:
        return "dense";
    case EVAL_TIER_COUNT:
        break;
    }
    return "unknown";
}

std::uint64_t& counter_for_tier(RuntimeCounters& counters, EvalTier tier) {
    if (tier == EVAL_COARSE) {
        return counters.coarse_evaluations;
    }
    if (tier == EVAL_MID) {
        return counters.mid_evaluations;
    }
    return counters.dense_evaluations;
}

struct SearchContext {
    const Config& cfg;
    const FixedValues& fixed;
    const std::array<BandSpec, MODE_COUNT>& bands;
    const std::array<FrequencyPlan, MODE_COUNT>& coarse_plans;
    const std::array<FrequencyPlan, MODE_COUNT>& mid_plans;
    const std::array<FrequencyPlan, MODE_COUNT>& dense_plans;
    const Bounds& bounds;
    const CapTable& baseline;
    std::array<std::unordered_map<CapKey, Evaluation, CapKeyHash>,
               EVAL_TIER_COUNT> cache;
    RuntimeCounters counters;
    double early_exit_cutoff = INF;

    const std::array<FrequencyPlan, MODE_COUNT>& plans(EvalTier tier) const {
        if (tier == EVAL_COARSE) {
            return coarse_plans;
        }
        if (tier == EVAL_MID) {
            return mid_plans;
        }
        return dense_plans;
    }

    Evaluation evaluate(const CapTable& cap, EvalTier tier) {
        return evaluate_with_baseline(cap, tier, baseline);
    }

    Evaluation evaluate_with_baseline(const CapTable& cap,
                                      EvalTier tier,
                                      const CapTable& eval_baseline) {
        const CapKey key = make_cap_key(cap);
        auto& bucket = cache[static_cast<int>(tier)];
        auto it = bucket.find(key);
        if (it != bucket.end()) {
            ++counters.cache_hits;
            return it->second;
        }

        const bool allow_early =
            cfg.enable_early_exit && tier != EVAL_DENSE;
        const double cutoff = allow_early ? early_exit_cutoff : INF;
        Evaluation eval = evaluate_candidate(cap, fixed, bands, plans(tier),
                                             eval_baseline, cfg, cutoff,
                                             allow_early);
        ++counter_for_tier(counters, tier);
        if (eval.all_modes_feasible) {
            ++counters.feasible_candidates_found;
        } else if (tier != EVAL_DENSE &&
                   eval.rf_penalty > 0.0 &&
                   eval.rf_penalty < early_exit_cutoff) {
            early_exit_cutoff = eval.rf_penalty;
        }
        bucket.emplace(key, eval);
        return eval;
    }

    Candidate evaluate_candidate_at(const Candidate& c, EvalTier tier) {
        Candidate out = c;
        out.eval = evaluate(out.cap, tier);
        return out;
    }
};

void sort_population_by_fitness(
    std::vector<Candidate>& population,
    std::vector<std::pair<double, double>>* jde_params = nullptr) {
    if (jde_params == nullptr || jde_params->size() != population.size()) {
        std::sort(population.begin(), population.end(), better_candidate);
        return;
    }

    std::vector<int> order(population.size());
    for (int i = 0; i < static_cast<int>(order.size()); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&](int lhs, int rhs) {
                  return better_candidate(population[lhs], population[rhs]);
              });

    std::vector<Candidate> sorted_population;
    std::vector<std::pair<double, double>> sorted_params;
    sorted_population.reserve(population.size());
    sorted_params.reserve(jde_params->size());
    for (int idx : order) {
        sorted_population.push_back(population[idx]);
        sorted_params.push_back((*jde_params)[idx]);
    }
    population = std::move(sorted_population);
    *jde_params = std::move(sorted_params);
}

bool better_rf_quality(const Candidate& a, const Candidate& b) {
    if (nearly_less(a.eval.rf_penalty, b.eval.rf_penalty)) {
        return true;
    }
    if (nearly_less(b.eval.rf_penalty, a.eval.rf_penalty)) {
        return false;
    }
    if (nearly_less(a.eval.rf_objective, b.eval.rf_objective)) {
        return true;
    }
    if (nearly_less(b.eval.rf_objective, a.eval.rf_objective)) {
        return false;
    }
    if (nearly_less(a.eval.worst_rf_violation, b.eval.worst_rf_violation)) {
        return true;
    }
    if (nearly_less(b.eval.worst_rf_violation, a.eval.worst_rf_violation)) {
        return false;
    }
    if (nearly_less(b.eval.rf_margin, a.eval.rf_margin)) {
        return true;
    }
    if (nearly_less(a.eval.rf_margin, b.eval.rf_margin)) {
        return false;
    }
    return a.eval.fitness < b.eval.fitness;
}

void repair_table(CapTable& cap, const Bounds& bounds, const MaskConstraints& mask) {
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            cap[mode][cell] = clamp_value(cap[mode][cell],
                                          bounds.lower[cell],
                                          bounds.upper[cell]);
            cap[mode][cell] = quantize_cap(cap[mode][cell]);
            cap[mode][cell] = clamp_value(cap[mode][cell],
                                          bounds.lower[cell],
                                          bounds.upper[cell]);
        }
    }

    if (!mask.enabled) {
        return;
    }

    auto set_pair_average = [&](int cell, int m0, int m1) {
        const double avg = quantize_cap(0.5 * (cap[m0][cell] + cap[m1][cell]));
        cap[m0][cell] = clamp_value(avg, bounds.lower[cell], bounds.upper[cell]);
        cap[m1][cell] = clamp_value(avg, bounds.lower[cell], bounds.upper[cell]);
    };

    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        if (!mask.specified[cell]) {
            continue;
        }
        switch (mask.pattern[cell]) {
        case SHARE_ALL: {
            const double avg = quantize_cap((cap[MODE_N77][cell] +
                                             cap[MODE_N78][cell] +
                                             cap[MODE_N79][cell]) / 3.0);
            for (int mode = 0; mode < MODE_COUNT; ++mode) {
                cap[mode][cell] = clamp_value(avg,
                                              bounds.lower[cell],
                                              bounds.upper[cell]);
            }
            break;
        }
        case SHARE_77_78:
            set_pair_average(cell, MODE_N77, MODE_N78);
            break;
        case SHARE_77_79:
            set_pair_average(cell, MODE_N77, MODE_N79);
            break;
        case SHARE_78_79:
            set_pair_average(cell, MODE_N78, MODE_N79);
            break;
        case SHARE_NONE:
            break;
        }
    }
}

void enforce_pattern(CapTable& cap,
                     int cell,
                     SharePattern pattern,
                     const Bounds& bounds,
                     const MaskConstraints& mask) {
    auto avg_pair = [&](int m0, int m1) {
        const double avg = 0.5 * (cap[m0][cell] + cap[m1][cell]);
        cap[m0][cell] = avg;
        cap[m1][cell] = avg;
    };

    switch (pattern) {
    case SHARE_ALL: {
        const double avg = (cap[MODE_N77][cell] +
                            cap[MODE_N78][cell] +
                            cap[MODE_N79][cell]) / 3.0;
        for (int mode = 0; mode < MODE_COUNT; ++mode) {
            cap[mode][cell] = avg;
        }
        break;
    }
    case SHARE_77_78:
        avg_pair(MODE_N77, MODE_N78);
        break;
    case SHARE_77_79:
        avg_pair(MODE_N77, MODE_N79);
        break;
    case SHARE_78_79:
        avg_pair(MODE_N78, MODE_N79);
        break;
    case SHARE_NONE:
        break;
    }
    repair_table(cap, bounds, mask);
}

Candidate make_candidate(CapTable cap,
                         const Bounds& bounds,
                         const MaskConstraints& mask) {
    Candidate c;
    c.cap = cap;
    repair_table(c.cap, bounds, mask);
    return c;
}

struct InitialGroupTargets {
    int seed = 0;
    int local_jitter = 0;
    int wide_jitter = 0;
    int random_bucket = 0;
    int share_pattern = 0;
};

InitialGroupTargets compute_initial_group_targets(const Config& cfg) {
    std::array<double, 5> fractions = {{
        std::max(0.0, cfg.init_seed_fraction),
        std::max(0.0, cfg.init_local_jitter_fraction),
        std::max(0.0, cfg.init_wide_jitter_fraction),
        std::max(0.0, cfg.init_random_fraction),
        std::max(0.0, cfg.init_share_pattern_fraction)
    }};

    double sum = 0.0;
    for (double f : fractions) {
        sum += f;
    }
    if (sum <= 0.0) {
        fractions = {{0.25, 0.25, 0.20, 0.20, 0.10}};
        sum = 1.0;
    }

    std::array<int, 5> counts{};
    std::array<double, 5> remainder{};
    int assigned = 0;
    for (int i = 0; i < 5; ++i) {
        const double exact = fractions[i] / sum *
                             static_cast<double>(cfg.population_size);
        counts[i] = static_cast<int>(std::floor(exact));
        remainder[i] = exact - static_cast<double>(counts[i]);
        assigned += counts[i];
    }

    while (assigned < cfg.population_size) {
        int best = 0;
        for (int i = 1; i < 5; ++i) {
            if (remainder[i] > remainder[best]) {
                best = i;
            }
        }
        ++counts[best];
        remainder[best] = -1.0;
        ++assigned;
    }

    if (counts[0] == 0 && cfg.population_size > 0) {
        counts[0] = 1;
        ++assigned;
    }
    while (assigned > cfg.population_size) {
        int best = -1;
        for (int i = 1; i < 5; ++i) {
            if (counts[i] > 0 && (best < 0 || counts[i] > counts[best])) {
                best = i;
            }
        }
        if (best < 0) {
            break;
        }
        --counts[best];
        --assigned;
    }

    // Enforce minimum random fraction.
    const int min_random = static_cast<int>(
        cfg.min_random_fraction * static_cast<double>(cfg.population_size));
    const int current_random = counts[3];  // index 3 = random_bucket
    if (current_random < min_random) {
        const int deficit = min_random - current_random;
        counts[3] += deficit;
        // Take from the largest non-random bucket to compensate.
        // Priority for reduction: local_jitter (1), seed (0), wide_jitter (2).
        static const std::array<int, 3> reduce_priority = {{1, 0, 2}};
        int remaining = deficit;
        for (int idx : reduce_priority) {
            const int cut = std::min(remaining, counts[idx]);
            counts[idx] -= cut;
            remaining -= cut;
            if (remaining == 0) break;
        }
        if (remaining > 0) {
            const int cut = std::min(remaining, counts[4]);
            counts[4] -= cut;
            remaining -= cut;
        }
    }

    InitialGroupTargets targets;
    targets.seed = counts[0];
    targets.local_jitter = counts[1];
    targets.wide_jitter = counts[2];
    targets.random_bucket = counts[3];
    targets.share_pattern = counts[4];
    return targets;
}

CapTable make_random_table(const Bounds& bounds, std::mt19937_64& rng) {
    CapTable table{};
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            std::uniform_real_distribution<double> dist(bounds.lower[cell],
                                                        bounds.upper[cell]);
            table[mode][cell] = dist(rng);
        }
    }
    return table;
}

std::vector<CapTable> make_lhs_random_tables(const Bounds& bounds,
                                             int count,
                                             std::mt19937_64& rng) {
    std::vector<CapTable> tables(static_cast<std::size_t>(std::max(0, count)));
    if (count <= 0) {
        return tables;
    }

    std::uniform_real_distribution<double> urand(0.0, 1.0);
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            std::vector<int> strata(static_cast<std::size_t>(count));
            std::iota(strata.begin(), strata.end(), 0);
            std::shuffle(strata.begin(), strata.end(), rng);
            const double lo = bounds.lower[cell];
            const double hi = bounds.upper[cell];
            for (int i = 0; i < count; ++i) {
                const double u =
                    (static_cast<double>(strata[static_cast<std::size_t>(i)]) +
                     urand(rng)) / static_cast<double>(count);
                tables[static_cast<std::size_t>(i)][mode][cell] =
                    lo + u * (hi - lo);
            }
        }
    }
    return tables;
}

CapTable make_local_jitter_table(const Config& cfg,
                                 const CapTable& baseline,
                                 std::mt19937_64& rng) {
    CapTable table = baseline;
    std::normal_distribution<double> nrand(0.0, 1.0);
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            const double sigma = cfg.local_jitter_sigma *
                                 cfg.local_jitter_scale *
                                 std::max(baseline[mode][cell], 0.20 * PF);
            table[mode][cell] += sigma * nrand(rng);
        }
    }
    return table;
}

CapTable make_wide_jitter_table(const Config& cfg,
                                const Bounds& bounds,
                                const CapTable& baseline,
                                std::mt19937_64& rng) {
    CapTable table = baseline;
    std::normal_distribution<double> nrand(0.0, 1.0);
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            const double span = bounds.upper[cell] - bounds.lower[cell];
            table[mode][cell] += cfg.wide_jitter_sigma * span * nrand(rng);
        }
    }
    return table;
}

CapTable make_random_blend_table(const Config& cfg,
                                 const Bounds& bounds,
                                 const CapTable& baseline,
                                 std::mt19937_64& rng) {
    CapTable random_table = make_random_table(bounds, rng);
    CapTable table{};
    std::uniform_real_distribution<double> alpha_dist(
        cfg.random_seed_blend_min, cfg.random_seed_blend_max);
    const double alpha = alpha_dist(rng);
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            table[mode][cell] =
                alpha * baseline[mode][cell] +
                (1.0 - alpha) * random_table[mode][cell];
        }
    }
    return table;
}

void apply_random_share_patterns(CapTable& table,
                                 const Bounds& bounds,
                                 const Config& cfg,
                                 std::mt19937_64& rng) {
    std::uniform_real_distribution<double> urand(0.0, 1.0);
    std::uniform_int_distribution<int> cell_dist(0, CELL_COUNT - 1);
    const int max_cells = cfg.rf_only ? 1 : (cfg.use_share_patterns ? 5 : 3);
    std::uniform_int_distribution<int> count_dist(1, std::max(1, max_cells));
    const int pattern_count = count_dist(rng);

    for (int i = 0; i < pattern_count; ++i) {
        const int cell = cell_dist(rng);
        SharePattern pattern = SHARE_NONE;
        if (cfg.rf_only) {
            std::uniform_int_distribution<int> pair_dist(1, 3);
            pattern = static_cast<SharePattern>(pair_dist(rng));
            if (cfg.use_share_patterns && urand(rng) < 0.10) {
                pattern = SHARE_ALL;
            }
        } else {
            std::uniform_int_distribution<int> pattern_dist(1, 4);
            pattern = static_cast<SharePattern>(pattern_dist(rng));
            if (!cfg.use_share_patterns && pattern == SHARE_ALL &&
                urand(rng) < 0.70) {
                std::uniform_int_distribution<int> pair_dist(1, 3);
                pattern = static_cast<SharePattern>(pair_dist(rng));
            }
        }
        enforce_pattern(table, cell, pattern, bounds, cfg.mask);
    }
}

CapTable make_seed_variant(const Config& cfg,
                           const Bounds& bounds,
                           const CapTable& baseline,
                           int index,
                           std::mt19937_64& rng) {
    if (index == 0) {
        return baseline;
    }

    CapTable table = baseline;
    std::normal_distribution<double> nrand(0.0, 1.0);
    std::uniform_int_distribution<int> cell_dist(0, CELL_COUNT - 1);
    const int mode = 1 + (index % 3);
    if (mode == 1) {
        for (int m = 0; m < MODE_COUNT; ++m) {
            for (int cell = 0; cell < CELL_COUNT; ++cell) {
                const double sigma = 0.50 * cfg.local_jitter_sigma *
                                     std::max(baseline[m][cell], 0.20 * PF);
                table[m][cell] += sigma * nrand(rng);
            }
        }
    } else if (mode == 2) {
        std::uniform_int_distribution<int> pattern_dist(1, 3);
        enforce_pattern(table, cell_dist(rng),
                        static_cast<SharePattern>(pattern_dist(rng)),
                        bounds, cfg.mask);
    } else if (mode == 3) {
        const std::array<double, MODE_COUNT> centers = {{
            3.75 * GHZ, 3.55 * GHZ, 4.70 * GHZ
        }};
        const double blend =
            0.10 + 0.10 * static_cast<double>((index / 4) % 5);
        for (int m = 0; m < MODE_COUNT; ++m) {
            const double scale_lc = sqr(3.75 * GHZ / centers[m]);
            const double scale_linear = 3.75 * GHZ / centers[m];
            for (int cell = 0; cell < CELL_COUNT; ++cell) {
                const bool mostly_frequency_setter =
                    (cell == CELL_C3 || cell == CELL_C4 ||
                     cell == CELL_C6 || cell == CELL_C7 || cell == CELL_C9);
                const double scale =
                    mostly_frequency_setter ? scale_lc : scale_linear;
                table[m][cell] =
                    (1.0 - blend) * baseline[m][cell] +
                    blend * baseline[MODE_N77][cell] * scale;
            }
        }
    }
    return table;
}

std::vector<Candidate> make_initial_population(const Config& cfg,
                                               const Bounds& bounds,
                                               const CapTable& baseline,
                                               std::mt19937_64& rng,
                                               PopulationComposition* composition) {
    std::vector<Candidate> population;
    population.reserve(static_cast<std::size_t>(cfg.population_size));
    PopulationComposition actual;
    const InitialGroupTargets targets = compute_initial_group_targets(cfg);

    auto add = [&](const CapTable& table, int& counter) {
        if (static_cast<int>(population.size()) < cfg.population_size) {
            population.push_back(make_candidate(table, bounds, cfg.mask));
            ++counter;
        }
    };

    for (int i = 0; i < targets.seed; ++i) {
        add(make_seed_variant(cfg, bounds, baseline, i, rng), actual.seed);
    }

    for (int i = 0; i < targets.local_jitter; ++i) {
        add(make_local_jitter_table(cfg, baseline, rng), actual.local_jitter);
    }

    for (int i = 0; i < targets.wide_jitter; ++i) {
        add(make_wide_jitter_table(cfg, bounds, baseline, rng),
            actual.wide_jitter);
    }

    const int random_blend_target = targets.random_bucket / 3;
    const int random_full_target =
        targets.random_bucket - random_blend_target;
    if (cfg.lhs_random_init) {
        const std::vector<CapTable> lhs_tables =
            make_lhs_random_tables(bounds, random_full_target, rng);
        for (const CapTable& table : lhs_tables) {
            add(table, actual.random_full);
        }
    } else {
        for (int i = 0; i < random_full_target; ++i) {
            add(make_random_table(bounds, rng), actual.random_full);
        }
    }
    for (int i = 0; i < random_blend_target; ++i) {
        add(make_random_blend_table(cfg, bounds, baseline, rng),
            actual.random_blend);
    }

    for (int i = 0; i < targets.share_pattern; ++i) {
        CapTable table = baseline;
        if (i % 3 == 0) {
            table = make_local_jitter_table(cfg, baseline, rng);
        } else if (i % 3 == 1) {
            table = make_wide_jitter_table(cfg, bounds, baseline, rng);
        } else {
            table = make_random_blend_table(cfg, bounds, baseline, rng);
        }
        apply_random_share_patterns(table, bounds, cfg, rng);
        add(table, actual.share_pattern);
    }

    while (static_cast<int>(population.size()) < cfg.population_size) {
        add(make_random_blend_table(cfg, bounds, baseline, rng),
            actual.random_blend);
    }

    if (composition != nullptr) {
        *composition = actual;
    }
    return population;
}

void evaluate_population(std::vector<Candidate>& population,
                         const FixedValues& fixed,
                         const std::array<BandSpec, MODE_COUNT>& bands,
                         const std::array<FrequencyPlan, MODE_COUNT>& plans,
                         const CapTable& baseline,
                         const Config& cfg) {
    // The GA is still CPU-based. OpenMP parallelizes independent candidate
    // evaluations. It does not change the fitness definition or the
    // optimization objective.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < static_cast<int>(population.size()); ++i) {
        population[i].eval = evaluate_candidate(population[i].cap,
                                                fixed,
                                                bands,
                                                plans,
                                                baseline,
                                                cfg);
    }
}

void evaluate_population(std::vector<Candidate>& population,
                         SearchContext& ctx,
                         EvalTier tier) {
    for (Candidate& c : population) {
        c.eval = ctx.evaluate(c.cap, tier);
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
        if (better_candidate(population[idx], population[best])) {
            best = idx;
        }
    }
    return population[best];
}

std::pair<double, double> random_de_params(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> fdist(0.1, 1.0);
    std::uniform_real_distribution<double> crdist(0.0, 1.0);
    return {fdist(rng), crdist(rng)};
}

void apply_compression_mutation(CapTable& cap,
                                const Bounds& bounds,
                                const MaskConstraints& mask,
                                bool use_share_patterns,
                                std::mt19937_64& rng) {
    std::uniform_int_distribution<int> pattern_dist(1, 4);
    std::uniform_real_distribution<double> urand(0.0, 1.0);
    // Prefer lower-RF-sensitivity cells for compression attempts.
    // High-sensitivity cells (TZ setters) get 1 weight; others get 3 weight.
    static const std::array<int, CELL_COUNT> cell_weights = {{
        3, 3, 1, 1, 3, 1, 1, 3, 1   // C1 C2 C3 C4 C5 C6 C7 C8 C9
    }};
    std::discrete_distribution<int> weighted_cell_dist(
        cell_weights.begin(), cell_weights.end());
    const int cell = weighted_cell_dist(rng);

    SharePattern pattern = static_cast<SharePattern>(pattern_dist(rng));
    if (use_share_patterns && urand(rng) < 0.20) {
        pattern = SHARE_ALL;
    }
    enforce_pattern(cap, cell, pattern, bounds, mask);
    // TODO: Add evaluator-backed partial RF repair for the forced cell when
    // apply_compression_mutation has access to RF plans and fixed values.
}

void apply_decompression_mutation(CapTable& cap,
                                  const Bounds& bounds,
                                  const MaskConstraints& mask,
                                  std::mt19937_64& rng) {
    std::vector<std::pair<int, int>> shared_modes;
    const SharingReport sharing = compute_sharing_report(cap);
    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        if (mask.enabled && mask.specified[cell]) {
            continue;
        }
        if (sharing.unique_count[cell] < MODE_COUNT) {
            for (int mode = 0; mode < MODE_COUNT; ++mode) {
                shared_modes.push_back({cell, mode});
            }
        }
    }
    if (shared_modes.empty()) {
        return;
    }
    std::uniform_int_distribution<int> pick(0,
                                            static_cast<int>(shared_modes.size()) - 1);
    const auto [cell, mode] = shared_modes[pick(rng)];
    std::normal_distribution<double> nrand(0.0, 1.0);
    const double sigma = 0.05 * (bounds.upper[cell] - bounds.lower[cell]);
    cap[mode][cell] += sigma * nrand(rng);
    repair_table(cap, bounds, mask);
}

Candidate crossover_and_mutate(const Candidate& a,
                               const Candidate& b,
                               const Bounds& bounds,
                               const Config& cfg,
                               int generation,
                               std::mt19937_64& rng) {
    Candidate child;
    std::uniform_real_distribution<double> urand(0.0, 1.0);
    std::normal_distribution<double> nrand(0.0, 1.0);
    const double progress = static_cast<double>(generation) /
                            static_cast<double>(std::max(1, cfg.max_generations));
    const double anneal = std::max(0.15, 1.0 - progress);

    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            double value = a.cap[mode][cell];
            if (urand(rng) < cfg.crossover_rate) {
                const double alpha = 1.30 * urand(rng) - 0.15;
                value = alpha * a.cap[mode][cell] +
                        (1.0 - alpha) * b.cap[mode][cell];
            }
            if (urand(rng) < cfg.mutation_rate) {
                const double span = bounds.upper[cell] - bounds.lower[cell];
                value += cfg.mutation_scale * anneal * span * nrand(rng);
            }
            if (urand(rng) < cfg.reset_mutation_rate) {
                std::uniform_real_distribution<double> reset(bounds.lower[cell],
                                                             bounds.upper[cell]);
                value = reset(rng);
            }
            child.cap[mode][cell] = value;
        }
    }

    // Mode-row swap crossover: 15% chance, swap one complete mode row from b.
    {
        std::uniform_real_distribution<double> urow(0.0, 1.0);
        if (urow(rng) < 0.15) {
            std::uniform_int_distribution<int> mode_pick(0, MODE_COUNT - 1);
            const int swap_mode = mode_pick(rng);
            for (int cell = 0; cell < CELL_COUNT; ++cell) {
                child.cap[swap_mode][cell] = b.cap[swap_mode][cell];
            }
        }
    }

    if (urand(rng) < cfg.compression_mutation_rate) {
        apply_compression_mutation(child.cap, bounds, cfg.mask,
                                   cfg.use_share_patterns, rng);
    }
    if (urand(rng) < cfg.decompression_mutation_rate) {
        apply_decompression_mutation(child.cap, bounds, cfg.mask, rng);
    }

    repair_table(child.cap, bounds, cfg.mask);
    return child;
}

bool same_cap_table(const Candidate& a, const Candidate& b) {
    return make_cap_key(a.cap) == make_cap_key(b.cap);
}

double lowest_insertion_loss(const Candidate& c) {
    double value = -INF;
    for (const auto& m : c.eval.modes) {
        value = std::max(value, m.max_pass_insertion_loss_db);
    }
    return value;
}

void prune_archive_bucket(std::vector<Candidate>& bucket, int limit) {
    if (static_cast<int>(bucket.size()) <= limit) {
        return;
    }

    std::vector<Candidate> kept;
    auto add_index = [&](int idx) {
        if (idx < 0 || idx >= static_cast<int>(bucket.size())) {
            return;
        }
        for (const Candidate& c : kept) {
            if (same_cap_table(c, bucket[idx])) {
                return;
            }
        }
        kept.push_back(bucket[idx]);
    };

    auto best_by = [&](auto pred) {
        int best = 0;
        for (int i = 1; i < static_cast<int>(bucket.size()); ++i) {
            if (pred(bucket[i], bucket[best])) {
                best = i;
            }
        }
        return best;
    };

    add_index(best_by([](const Candidate& a, const Candidate& b) {
        return better_candidate(a, b);
    }));
    add_index(best_by([](const Candidate& a, const Candidate& b) {
        return nearly_less(a.eval.hardware_cost, b.eval.hardware_cost);
    }));
    add_index(best_by([](const Candidate& a, const Candidate& b) {
        return nearly_less(b.eval.rf_margin, a.eval.rf_margin);
    }));
    add_index(best_by([](const Candidate& a, const Candidate& b) {
        return nearly_less(a.eval.rf_objective, b.eval.rf_objective);
    }));
    add_index(best_by([](const Candidate& a, const Candidate& b) {
        return nearly_less(b.eval.baseline_distance_cost,
                           a.eval.baseline_distance_cost);
    }));
    add_index(best_by([](const Candidate& a, const Candidate& b) {
        return nearly_less(lowest_insertion_loss(a),
                           lowest_insertion_loss(b));
    }));

    std::sort(bucket.begin(), bucket.end(), better_candidate);
    for (const Candidate& c : bucket) {
        if (static_cast<int>(kept.size()) >= limit) {
            break;
        }
        bool exists = false;
        for (const Candidate& k : kept) {
            if (same_cap_table(c, k)) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            kept.push_back(c);
        }
    }

    bucket = std::move(kept);
}

void update_feasible_archive(SearchResult& result,
                             const Candidate& c,
                             const Config& cfg) {
    if (!c.eval.all_modes_feasible) {
        return;
    }

    const int key = c.eval.sharing.sharing_cost;
    std::vector<Candidate>& bucket = result.archive_by_sharing[key];
    bool duplicate = false;
    for (const Candidate& existing : bucket) {
        if (same_cap_table(existing, c)) {
            duplicate = true;
            break;
        }
    }
    if (!duplicate) {
        bucket.push_back(c);
        prune_archive_bucket(bucket, std::max(1, cfg.archive_per_sharing));
    }

    if (!result.has_best_rf_feasible ||
        better_rf_quality(c, result.best_rf_feasible)) {
        result.best_rf_feasible = c;
        result.has_best_rf_feasible = true;
    }
}

void update_archive(SearchResult& result,
                    const std::vector<Candidate>& population,
                    const Config& cfg) {
    for (const Candidate& c : population) {
        update_feasible_archive(result, c, cfg);
    }
}

std::vector<const Candidate*> feasible_archive_refs(const SearchResult& result) {
    std::vector<const Candidate*> refs;
    for (const auto& kv : result.archive_by_sharing) {
        for (const Candidate& c : kv.second) {
            if (c.eval.all_modes_feasible) {
                refs.push_back(&c);
            }
        }
    }
    return refs;
}

Candidate select_stage2_parent(const std::vector<Candidate>& population,
                               const SearchResult& result,
                               const Config& cfg,
                               std::mt19937_64& rng) {
    std::uniform_real_distribution<double> urand(0.0, 1.0);
    const double r = urand(rng);
    std::vector<const Candidate*> pool;

    if (!cfg.rf_only && r < 0.70) {
        for (const Candidate& c : population) {
            if (c.eval.all_modes_feasible) {
                pool.push_back(&c);
            }
        }
        const std::vector<const Candidate*> archive = feasible_archive_refs(result);
        pool.insert(pool.end(), archive.begin(), archive.end());
    } else if (!cfg.rf_only && r >= 0.90) {
        for (const Candidate& c : population) {
            if (!c.eval.all_modes_feasible) {
                pool.push_back(&c);
            }
        }
    }

    if (pool.empty()) {
        for (const Candidate& c : population) {
            pool.push_back(&c);
        }
    }

    std::uniform_int_distribution<int> pick(
        0, static_cast<int>(pool.size()) - 1);
    return *pool[static_cast<std::size_t>(pick(rng))];
}

int feasible_count(const std::vector<Candidate>& population) {
    int count = 0;
    for (const Candidate& c : population) {
        if (c.eval.all_modes_feasible) {
            ++count;
        }
    }
    return count;
}

Candidate dense_evaluate_candidate(const Candidate& c,
                                   const FixedValues& fixed,
                                   const std::array<BandSpec, MODE_COUNT>& bands,
                                   const std::array<FrequencyPlan, MODE_COUNT>& dense_plans,
                                   const CapTable& baseline,
                                   const Config& cfg) {
    Candidate out = c;
    out.eval = evaluate_candidate(out.cap, fixed, bands, dense_plans, baseline, cfg);
    return out;
}

Candidate dense_evaluate_candidate(const Candidate& c, SearchContext& ctx) {
    return ctx.evaluate_candidate_at(c, EVAL_DENSE);
}

Candidate local_polish_candidate(const Candidate& start,
                                 SearchContext& ctx,
                                 int max_steps);
Candidate mode_wise_rf_repair(const Candidate& start, SearchContext& ctx);
bool apply_feasibility_preserving_sharing_mutation(
    Candidate& child,
    const Candidate& reference,
    SearchContext& ctx,
    std::mt19937_64& rng);

void check_top_dense(const std::vector<Candidate>& population,
                     int top_count,
                     const FixedValues& fixed,
                     const std::array<BandSpec, MODE_COUNT>& bands,
                     const std::array<FrequencyPlan, MODE_COUNT>& dense_plans,
                     const CapTable& baseline,
                     const Config& cfg,
                     SearchResult& result,
                     int restart,
                     int generation) {
    const int n = std::min(top_count, static_cast<int>(population.size()));
    for (int i = 0; i < n; ++i) {
        Candidate dense = dense_evaluate_candidate(population[i], fixed, bands,
                                                   dense_plans, baseline, cfg);
        update_feasible_archive(result, dense, cfg);
        if (!result.has_best || better_candidate(dense, result.best)) {
            result.best = dense;
            result.has_best = true;
            result.restart = restart;
            result.generation = generation;
        }
    }
}

void check_top_tier(std::vector<Candidate>& population,
                    int top_count,
                    EvalTier tier,
                    SearchContext& ctx,
                    SearchResult& result,
                    int restart,
                    int generation) {
    const int n = std::min(top_count, static_cast<int>(population.size()));
    for (int i = 0; i < n; ++i) {
        Candidate verified = ctx.evaluate_candidate_at(population[i], tier);
        if (tier == EVAL_DENSE &&
            ctx.cfg.local_polish &&
            verified.eval.all_modes_feasible) {
            verified = local_polish_candidate(verified, ctx,
                                              ctx.cfg.local_polish_steps);
        }
        if (verified.eval.all_modes_feasible ||
            !population[i].eval.all_modes_feasible ||
            better_candidate(verified, population[i])) {
            population[i] = verified;
        }
        if (tier == EVAL_DENSE) {
            update_feasible_archive(result, population[i], ctx.cfg);
        }
        if (!result.has_best || better_candidate(population[i], result.best)) {
            result.best = population[i];
            result.has_best = true;
            result.restart = restart;
            result.generation = generation;
        }
    }
}

double worst_rl(const Candidate& c);
double worst_il(const Candidate& c);
double worst_stop(const Candidate& c);
double worst_harmonic(const Candidate& c);

void print_population_composition(const std::string& stage,
                                  int restart,
                                  const PopulationComposition& composition) {
    std::cout << "[stage=" << stage << " restart=" << restart
              << "] population_composition:"
              << " seed=" << composition.seed
              << " local_jitter=" << composition.local_jitter
              << " wide_jitter=" << composition.wide_jitter
              << " random=" << composition.random_full
              << " random_blend=" << composition.random_blend
              << " share_pattern=" << composition.share_pattern
              << " total=" << composition.total() << "\n";
}

void print_progress(const SearchResult& result,
                    const std::vector<Candidate>& population,
                    const std::string& stage,
                    int restart,
                    int generation) {
    const double best_rf_penalty =
        result.has_best ? result.best.eval.rf_penalty : INF;
    std::cout << "[stage=" << stage
              << " restart=" << restart
              << " gen=" << generation << "]\n"
              << "best_feasible="
              << yes_no(result.has_best_rf_feasible) << "\n"
              << "best_rf_penalty=" << std::scientific
              << std::setprecision(3) << best_rf_penalty << std::fixed
              << "\n"
              << "population_feasible_count="
              << feasible_count(population) << "\n";

    const Candidate* best_feasible = nullptr;
    if (result.has_best && result.best.eval.all_modes_feasible) {
        best_feasible = &result.best;
    } else if (result.has_best_rf_feasible) {
        best_feasible = &result.best_rf_feasible;
    }

    if (best_feasible != nullptr) {
        const Candidate& c = *best_feasible;
        std::cout << "sharing_cost=" << c.eval.sharing.sharing_cost
                  << "\n"
                  << "unique=" << c.eval.total_unique_count << "\n"
                  << "hw=" << std::fixed << std::setprecision(3)
                  << c.eval.hardware_cost << "\n"
                  << "worst_RL=" << worst_rl(c) << "\n"
                  << "worst_IL=" << worst_il(c) << "\n";
    }
}

SearchResult run_ctc_ga(const Config& cfg,
                        const FixedValues& fixed,
                        const std::array<BandSpec, MODE_COUNT>& bands,
                        const std::array<FrequencyPlan, MODE_COUNT>& coarse_plans,
                        const std::array<FrequencyPlan, MODE_COUNT>& mid_plans,
                        const std::array<FrequencyPlan, MODE_COUNT>& dense_plans,
                        const Bounds& bounds,
                        const CapTable& baseline,
                        const std::string& stage) {
    const auto started = std::chrono::steady_clock::now();
    SearchContext ctx{cfg, fixed, bands, coarse_plans, mid_plans,
                      dense_plans, bounds, baseline, {}, {}};
    SearchResult result;
    result.lambda_compression = cfg.lambda_compression;
    result.baseline = make_candidate(baseline, bounds, cfg.mask);
    result.baseline.eval = ctx.evaluate(result.baseline.cap, EVAL_DENSE);
    result.best = result.baseline;
    result.has_best = true;
    result.restart = 0;
    result.generation = 0;
    update_feasible_archive(result, result.baseline, cfg);

    for (int restart = 1; restart <= cfg.max_restarts; ++restart) {
        const std::uint64_t salt =
            0x9e3779b97f4a7c15ULL *
            static_cast<std::uint64_t>(restart + 131);
        std::mt19937_64 rng(cfg.seed ^ salt ^
                            static_cast<std::uint64_t>(cfg.lambda_compression * 1009.0));

        PopulationComposition composition;
        const CapTable& seed_table =
            (restart > 1 && result.has_best_rf_feasible)
            ? result.best_rf_feasible.cap
            : baseline;
        std::vector<Candidate> population =
            make_initial_population(cfg, bounds, seed_table, rng, &composition);
        std::vector<std::pair<double, double>> jde_params(population.size());
        for (auto& p : jde_params) {
            p = random_de_params(rng);
        }
        if (!cfg.quiet) {
            print_population_composition(stage, restart, composition);
        }
        OptimizerKind active_optimizer = cfg.optimizer;
        if (cfg.jde_mode) {
            active_optimizer = OPT_JDE;
        }
        if (active_optimizer == OPT_HYBRID) {
            active_optimizer = cfg.rf_only ? OPT_DE : OPT_GA;
        }
        const bool de_like =
            active_optimizer == OPT_JDE || active_optimizer == OPT_DE;

        evaluate_population(population, ctx, EVAL_COARSE);
        sort_population_by_fitness(population, de_like ? &jde_params : nullptr);
        update_archive(result, population, cfg);

        double best_stagnation_metric = std::numeric_limits<double>::infinity();
        int stagnation_counter = 0;

        for (int generation = 0; generation <= cfg.max_generations; ++generation) {
            if (generation % cfg.mid_check_interval == 0 ||
                generation == cfg.max_generations) {
                check_top_tier(population,
                               std::min(cfg.mid_top_count,
                                        cfg.population_size),
                               EVAL_MID, ctx, result, restart, generation);
                sort_population_by_fitness(population,
                                           de_like ? &jde_params : nullptr);
            }
            if (generation % cfg.dense_check_interval == 0 ||
                generation == cfg.max_generations) {
                check_top_tier(population,
                               std::min(cfg.dense_top_count,
                                        cfg.population_size),
                               EVAL_DENSE, ctx, result, restart, generation);
                sort_population_by_fitness(population,
                                           de_like ? &jde_params : nullptr);

                if (!cfg.quiet &&
                    (generation % cfg.progress_interval == 0 ||
                     generation == cfg.max_generations)) {
                    print_progress(result, population, stage, restart,
                                   generation);
                }
            }

            if (generation == cfg.max_generations) {
                break;
            }

            if (active_optimizer == OPT_JDE || active_optimizer == OPT_DE) {
                std::uniform_real_distribution<double> urand(0.0, 1.0);
                const int n = static_cast<int>(population.size());
                std::uniform_int_distribution<int> index_pick(0, n - 1);
                std::uniform_int_distribution<int> gene_pick(
                    0, MODE_COUNT * CELL_COUNT - 1);
                const int pbest_count = std::max(2, n / 5);
                std::uniform_int_distribution<int> pbest_pick(0, pbest_count - 1);

                for (int i = 0; i < n; ++i) {
                    double f_new = cfg.jde_F_init;
                    double cr_new = cfg.jde_CR_init;
                    if (active_optimizer == OPT_JDE) {
                        f_new = jde_params[i].first;
                        cr_new = jde_params[i].second;
                    }
                    if (active_optimizer == OPT_JDE && urand(rng) < cfg.jde_tau1) {
                        f_new = 0.1 + 0.9 * urand(rng);
                    }
                    if (active_optimizer == OPT_JDE && urand(rng) < cfg.jde_tau2) {
                        cr_new = urand(rng);
                    }

                    int r1 = i;
                    int r2 = i;
                    while (r1 == i) {
                        r1 = index_pick(rng);
                    }
                    while (r2 == i || r2 == r1) {
                        r2 = index_pick(rng);
                    }
                    const int pbest = pbest_pick(rng);

                    CapTable trial_table = population[i].cap;
                    const int j_rand = gene_pick(rng);
                    int gene = 0;
                    for (int mode = 0; mode < MODE_COUNT; ++mode) {
                        for (int cell = 0; cell < CELL_COUNT; ++cell) {
                            const double mutant =
                                population[i].cap[mode][cell] +
                                f_new * (population[pbest].cap[mode][cell] -
                                         population[i].cap[mode][cell]) +
                                f_new * (population[r1].cap[mode][cell] -
                                         population[r2].cap[mode][cell]);
                            if (urand(rng) < cr_new || gene == j_rand) {
                                trial_table[mode][cell] = mutant;
                            }
                            ++gene;
                        }
                    }

                    Candidate trial =
                        make_candidate(trial_table, bounds, cfg.mask);
                    trial.eval = ctx.evaluate(trial.cap, EVAL_COARSE);
                    if (!trial.eval.all_modes_feasible) {
                        trial = mode_wise_rf_repair(trial, ctx);
                    }
                    if (better_candidate(trial, population[i])) {
                        population[i] = trial;
                        if (active_optimizer == OPT_JDE) {
                            jde_params[i] = {f_new, cr_new};
                        }
                    }
                }
            } else {
                std::vector<Candidate> next;
                next.reserve(static_cast<std::size_t>(cfg.population_size));
                const int elite_count =
                    std::min(cfg.elite_count, static_cast<int>(population.size()));

                if (generation % cfg.elite_dense_interval == 0 ||
                    generation == cfg.max_generations) {
                    // Dense-verify the elite before promoting them.
                    for (int i = 0; i < elite_count &&
                                    i < static_cast<int>(population.size()); ++i) {
                        Candidate dense = dense_evaluate_candidate(
                            population[i], ctx);
                        update_feasible_archive(result, dense, cfg);
                        if (dense.eval.all_modes_feasible ||
                            !population[i].eval.all_modes_feasible) {
                            population[i] = dense;
                        }
                    }
                    sort_population_by_fitness(population);
                }

                for (int i = 0; i < elite_count; ++i) {
                    next.push_back(population[i]);
                }

                while (static_cast<int>(next.size()) < cfg.population_size) {
                    const Candidate p1 =
                        (!cfg.rf_only && result.has_best_rf_feasible)
                        ? select_stage2_parent(population, result, cfg, rng)
                        : tournament_select(population, cfg.tournament_size, rng);
                    const Candidate p2 =
                        (!cfg.rf_only && result.has_best_rf_feasible)
                        ? select_stage2_parent(population, result, cfg, rng)
                        : tournament_select(population, cfg.tournament_size, rng);
                    next.push_back(crossover_and_mutate(p1, p2, bounds, cfg,
                                                        generation, rng));
                    apply_feasibility_preserving_sharing_mutation(
                        next.back(), p1, ctx, rng);
                    if (!next.back().eval.all_modes_feasible &&
                        result.has_best_rf_feasible) {
                        next.back().eval = ctx.evaluate(next.back().cap,
                                                        EVAL_COARSE);
                        next.back() = mode_wise_rf_repair(next.back(), ctx);
                    }
                }

                population = std::move(next);

                // Elites [0..elite_count-1] already hold dense evaluations from this
                // generation's elite dense-verify block (or coarse evals on other
                // generations). Only coarse-evaluate the newly generated offspring.
                const bool elites_already_evaluated =
                    (generation % cfg.elite_dense_interval == 0 ||
                     generation == cfg.max_generations);
                const int eval_start = elites_already_evaluated ? elite_count : 0;
                for (int i = eval_start; i < static_cast<int>(population.size()); ++i) {
                    population[i].eval = ctx.evaluate(population[i].cap, EVAL_COARSE);
                }
            }

            sort_population_by_fitness(population,
                                       de_like ? &jde_params : nullptr);
            update_archive(result, population, cfg);

            if (cfg.island_model &&
                generation > 0 &&
                generation % cfg.migration_interval == 0) {
                const int island_count =
                    std::min(cfg.islands,
                             std::max(1, static_cast<int>(population.size())));
                const int island_size =
                    std::max(1, static_cast<int>(population.size()) / island_count);
                for (int island = 0; island < island_count; ++island) {
                    const int src_start = island * island_size;
                    const int dst = (island + 1) % island_count;
                    const int dst_start = dst * island_size;
                    const int dst_end =
                        (dst == island_count - 1)
                        ? static_cast<int>(population.size())
                        : std::min(static_cast<int>(population.size()),
                                   dst_start + island_size);
                    const int count =
                        std::min(cfg.migration_count,
                                 std::min(island_size, dst_end - dst_start));
                    for (int m = 0; m < count; ++m) {
                        population[dst_end - 1 - m] = population[src_start + m];
                    }
                }
                sort_population_by_fitness(population,
                                           de_like ? &jde_params : nullptr);
            }

            const double stagnation_metric =
                result.has_best_rf_feasible
                ? static_cast<double>(
                      result.best_rf_feasible.eval.sharing.sharing_cost)
                : (result.has_best ? result.best.eval.rf_penalty : INF);

            if (stagnation_metric < best_stagnation_metric - 1e-6) {
                best_stagnation_metric = stagnation_metric;
                stagnation_counter = 0;
            } else {
                ++stagnation_counter;
            }

            if (stagnation_counter >= cfg.stagnation_threshold) {
                const int inject_count =
                    static_cast<int>(cfg.stagnation_inject_fraction *
                                     static_cast<double>(cfg.population_size));
                const int start =
                    static_cast<int>(population.size()) - inject_count;
                for (int k = std::max(0, start);
                     k < static_cast<int>(population.size()); ++k) {
                    population[k] = make_candidate(
                        make_random_table(bounds, rng), bounds, cfg.mask);
                    if (cfg.jde_mode &&
                        k < static_cast<int>(jde_params.size())) {
                        jde_params[k] = random_de_params(rng);
                    }
                }
                // Re-evaluate injected candidates.
                for (int k = std::max(0, start);
                     k < static_cast<int>(population.size()); ++k) {
                    population[k].eval = ctx.evaluate(population[k].cap,
                                                      EVAL_COARSE);
                }
                sort_population_by_fitness(population,
                                           de_like ? &jde_params : nullptr);
                stagnation_counter = 0;
                if (!cfg.quiet) {
                    std::cout << "[stage=" << stage << " restart=" << restart
                              << " gen=" << generation
                              << "] stagnation triggered: injected "
                              << inject_count << " random candidates\n";
                }
            }
        }
    }

    if (!result.has_best) {
        result.best = result.baseline;
        result.has_best = true;
    }

    std::vector<Candidate> archive_candidates;
    archive_candidates.reserve(result.archive_by_sharing.size() *
                               static_cast<std::size_t>(
                                   std::max(1, cfg.archive_per_sharing)) + 1);
    for (const auto& kv : result.archive_by_sharing) {
        for (const Candidate& c : kv.second) {
            archive_candidates.push_back(c);
        }
    }
    if (result.has_best) {
        archive_candidates.push_back(result.best);
    }
    if (result.has_best_rf_feasible) {
        archive_candidates.push_back(result.best_rf_feasible);
    }
    result.archive_by_sharing.clear();
    result.has_best_rf_feasible = false;
    for (const Candidate& c : archive_candidates) {
        Candidate dense = dense_evaluate_candidate(c, ctx);
        update_feasible_archive(result, dense, cfg);
        if (!result.has_best || better_candidate(dense, result.best)) {
            result.best = dense;
            result.has_best = true;
        }
    }

    if (!result.has_best_rf_feasible && !cfg.quiet) {
        std::cout << "\n*** NO RF-FEASIBLE CANDIDATE FOUND ***\n";
        std::cout << "Stage: " << stage
                  << ", Restarts: " << cfg.max_restarts
                  << ", Generations/restart: " << cfg.max_generations
                  << ", Population: " << cfg.population_size << "\n";
        std::cout << "Best rf_penalty achieved: "
                  << std::scientific << std::setprecision(4)
                  << result.best.eval.rf_penalty << "\n";
        std::cout << "Best worst_rf_violation: "
                  << result.best.eval.worst_rf_violation << "\n";
        std::cout << "Suggestions:\n"
                  << "  1. Run with --rf-only --init-random-fraction 0.60 "
                     "--lambda-base 0.0\n"
                  << "  2. Increase --restarts and --generations\n"
                  << "  3. Run --eval-baseline-only to check if baseline "
                     "itself is feasible\n"
                  << "  4. Run --single-cell-mask-scan to identify which "
                     "cells are most constrained\n";
    }

    // Prefer the best RF-feasible candidate from the archive over the
    // best overall candidate if they differ in feasibility.
    if (result.has_best_rf_feasible) {
        if (!result.best.eval.all_modes_feasible ||
            better_candidate(result.best_rf_feasible, result.best)) {
            result.best = result.best_rf_feasible;
        }
    }

    const auto ended = std::chrono::steady_clock::now();
    result.elapsed_seconds =
        std::chrono::duration<double>(ended - started).count();
    result.counters = ctx.counters;
    return result;
}

double worst_rl(const Candidate& c) {
    double value = INF;
    for (const auto& m : c.eval.modes) {
        value = std::min(value, m.min_pass_return_loss_db);
    }
    return value;
}

double worst_il(const Candidate& c) {
    double value = -INF;
    for (const auto& m : c.eval.modes) {
        value = std::max(value, m.max_pass_insertion_loss_db);
    }
    return value;
}

double worst_stop(const Candidate& c) {
    double value = INF;
    for (const auto& m : c.eval.modes) {
        value = std::min(value, m.lower_stop_min_rejection_db);
        value = std::min(value, m.upper_stop_min_rejection_db);
    }
    return value;
}

double worst_harmonic(const Candidate& c) {
    double value = INF;
    for (const auto& m : c.eval.modes) {
        value = std::min(value, m.harmonic2_min_rejection_db);
        value = std::min(value, m.harmonic3_min_rejection_db);
    }
    return value;
}

void print_cap_table(const Candidate& c) {
    std::cout << "\nCompressed capacitance table (pF)\n";
    std::cout << "Cell      N77      N78      N79      Pattern             Unique\n";
    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        std::cout << std::left << std::setw(5) << kCellNames[cell]
                  << std::right << std::fixed << std::setprecision(4)
                  << std::setw(9) << pf(c.cap[MODE_N77][cell])
                  << std::setw(9) << pf(c.cap[MODE_N78][cell])
                  << std::setw(9) << pf(c.cap[MODE_N79][cell])
                  << "  " << std::left << std::setw(20)
                  << pattern_name(c.eval.sharing.pattern[cell])
                  << std::right << c.eval.sharing.unique_count[cell] << "\n";
    }
}

void print_metrics_table(const Candidate& c) {
    std::cout << "\nRF metrics by mode\n";
    std::cout << "Mode Feas  RLmin  ILmax  LowStop  UpStop  H2min  H3min  "
              << "TZ1   TZ2   TZ3   TZ4\n";
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        const ModeMetrics& m = c.eval.modes[mode];
        std::cout << std::left << std::setw(4) << kModeNames[mode]
                  << " " << std::setw(4) << yes_no(m.constraints_ok)
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(7) << m.min_pass_return_loss_db
                  << std::setw(7) << m.max_pass_insertion_loss_db
                  << std::setw(9) << m.lower_stop_min_rejection_db
                  << std::setw(8) << m.upper_stop_min_rejection_db
                  << std::setw(7) << m.harmonic2_min_rejection_db
                  << std::setw(7) << m.harmonic3_min_rejection_db
                  << std::setw(6) << ghz(m.zeros.tz1_hz)
                  << std::setw(6) << ghz(m.zeros.tz2_hz)
                  << std::setw(6) << ghz(m.zeros.tz3_hz)
                  << std::setw(6) << ghz(m.zeros.tz4_hz)
                  << "\n";
    }
}

void print_failure_diagnostics(
    const Candidate& c,
    const std::array<BandSpec, MODE_COUNT>& bands) {
    std::cout << "\nConstraint diagnostics\n";
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        const ModeMetrics& m = c.eval.modes[mode];
        const BandSpec& band = bands[mode];
        std::cout << kModeNames[mode] << ":\n";
        auto line = [](const char* name,
                       double actual,
                       const char* relation,
                       double target,
                       double margin,
                       bool pass) {
            std::cout << "  " << std::left << std::setw(16) << name
                      << std::right << std::fixed << std::setprecision(3)
                      << " actual=" << std::setw(8) << actual
                      << " target " << relation << " " << std::setw(7) << target
                      << " margin=" << std::setw(8) << margin
                      << " " << yes_no(pass) << "\n";
        };
        line("Pass RL min", m.min_pass_return_loss_db, ">=",
             band.pass_return_loss_min_db,
             m.min_pass_return_loss_db - band.pass_return_loss_min_db,
             m.pass_return_loss_ok);
        line("Pass IL max", m.max_pass_insertion_loss_db, "<=",
             band.pass_insertion_loss_max_db,
             band.pass_insertion_loss_max_db - m.max_pass_insertion_loss_db,
             m.pass_insertion_loss_ok);
        line("Lower stop", m.lower_stop_min_rejection_db, ">=",
             band.stop_rejection_min_db,
             m.lower_stop_min_rejection_db - band.stop_rejection_min_db,
             m.lower_stop_ok);
        line("Upper stop", m.upper_stop_min_rejection_db, ">=",
             band.stop_rejection_min_db,
             m.upper_stop_min_rejection_db - band.stop_rejection_min_db,
             m.upper_stop_ok);
        line("H2 rejection", m.harmonic2_min_rejection_db, ">=",
             band.harmonic_rejection_min_db,
             m.harmonic2_min_rejection_db - band.harmonic_rejection_min_db,
             m.harmonic2_ok);
        line("H3 rejection", m.harmonic3_min_rejection_db, ">=",
             band.harmonic_rejection_min_db,
             m.harmonic3_min_rejection_db - band.harmonic_rejection_min_db,
             m.harmonic3_ok);
        std::cout << "  " << std::left << std::setw(16) << "TZ ordering"
                  << std::right << " TZ="
                  << std::fixed << std::setprecision(3)
                  << ghz(m.zeros.tz1_hz) << "/"
                  << ghz(m.zeros.tz2_hz) << "/"
                  << ghz(m.zeros.tz3_hz) << "/"
                  << ghz(m.zeros.tz4_hz) << " GHz "
                  << yes_no(m.transmission_zero_order_ok) << "\n";
    }
}

void print_sharing_summary(const Candidate& c) {
    const SharingReport& s = c.eval.sharing;
    const int max_unique_targets = MODE_COUNT * CELL_COUNT;
    std::cout << "\nSharing summary\n";
    std::cout << "  Total unique target count : " << s.total_unique_count
              << " / " << max_unique_targets << "\n";
    std::cout << "  Sharing cost              : " << s.sharing_cost << "\n";
    std::cout << "  All-three shared cells    : " << s.all_three_count
              << " / " << CELL_COUNT << "\n";
    std::cout << "  N77/N78 shared cells      : " << s.pair_77_78_count
              << " / " << CELL_COUNT << "\n";
    std::cout << "  N77/N79 shared cells      : " << s.pair_77_79_count
              << " / " << CELL_COUNT << "\n";
    std::cout << "  N78/N79 shared cells      : " << s.pair_78_79_count
              << " / " << CELL_COUNT << "\n";
}

void print_hardware_summary(const Candidate& c) {
    const HardwareReport& h = c.eval.hardware;
    std::cout << "\nSwitched-cap implementation heuristic summary\n";
    std::cout << "  Implementation cost       : " << std::fixed
              << std::setprecision(4) << h.total_cost << "\n";
    std::cout << "  Nonzero switched branches : " << h.nonzero_branch_count
              << "\n";
    std::cout << "  Total switched increment  : " << pf(h.total_increment_f)
              << " pF\n";
    std::cout << "  Max branch increment      : " << pf(h.max_increment_f)
              << " pF\n";
    std::cout << "  Large branches > 1 pF     : " << h.large_branch_count
              << "\n";
    std::cout << "  Huge branches > 2 pF      : " << h.huge_branch_count
              << "\n";
    std::cout << "  Tiny branches 0.005-0.05pF: " << h.tiny_branch_count
              << "\n";

    std::cout << "\nDecomposition by cell\n";
    std::cout << "Cell  Cfix     DeltaA   DeltaB   Branches  Cost    Logic\n";
    for (const auto& d : h.cells) {
        std::cout << std::left << std::setw(5) << kCellNames[d.cell]
                  << std::right << std::fixed << std::setprecision(4)
                  << std::setw(8) << pf(d.fixed_f)
                  << std::setw(9) << pf(d.delta_a_f)
                  << std::setw(9) << pf(d.delta_b_f)
                  << std::setw(9) << d.branch_count
                  << std::setw(8) << d.cost
                  << "  " << mode_logic_string(d) << "\n";
    }
}

void print_comparison(const Candidate& baseline, const Candidate& best) {
    std::cout << "\nBaseline vs CTC-GA result\n";
    std::cout << "Metric                         Baseline        Optimized\n";
    std::cout << "Feasible                       "
              << std::setw(8) << yes_no(baseline.eval.all_modes_feasible)
              << std::setw(15) << yes_no(best.eval.all_modes_feasible) << "\n";
    std::cout << "Unique target count            "
              << std::setw(8) << baseline.eval.total_unique_count
              << std::setw(15) << best.eval.total_unique_count << "\n";
    std::cout << "Sharing cost                   "
              << std::setw(8) << baseline.eval.sharing.sharing_cost
              << std::setw(15) << best.eval.sharing.sharing_cost << "\n";
    std::cout << "Hardware cost                  "
              << std::fixed << std::setprecision(4)
              << std::setw(8) << baseline.eval.hardware_cost
              << std::setw(15) << best.eval.hardware_cost << "\n";
    std::cout << "Max increment pF               "
              << std::setw(8) << pf(baseline.eval.hardware.max_increment_f)
              << std::setw(15) << pf(best.eval.hardware.max_increment_f) << "\n";
    std::cout << "Worst pass RL dB               "
              << std::setw(8) << worst_rl(baseline)
              << std::setw(15) << worst_rl(best) << "\n";
    std::cout << "Worst pass IL dB               "
              << std::setw(8) << worst_il(baseline)
              << std::setw(15) << worst_il(best) << "\n";
    std::cout << "Worst stop rejection dB        "
              << std::setw(8) << worst_stop(baseline)
              << std::setw(15) << worst_stop(best) << "\n";
    std::cout << "Worst harmonic rejection dB    "
              << std::setw(8) << worst_harmonic(baseline)
              << std::setw(15) << worst_harmonic(best) << "\n";
}

void print_archive(const SearchResult& result) {
    std::cout << "\nFeasible archive: best candidate by sharing cost\n";
    std::cout << "ShareCost  Unique  HWCost   WorstRL  WorstIL  WorstStop  WorstHarm\n";
    if (result.archive_by_sharing.empty()) {
        std::cout << "(no feasible candidates archived under the current evaluator)\n";
        return;
    }
    for (const auto& kv : result.archive_by_sharing) {
        for (const Candidate& c : kv.second) {
            if (!c.eval.all_modes_feasible) {
                continue;
            }
            std::cout << std::setw(9) << kv.first
                      << std::setw(8) << c.eval.total_unique_count
                      << std::fixed << std::setprecision(3)
                      << std::setw(8) << c.eval.hardware_cost
                      << std::setw(9) << worst_rl(c)
                      << std::setw(9) << worst_il(c)
                      << std::setw(11) << worst_stop(c)
                      << std::setw(11) << worst_harmonic(c)
                      << "\n";
        }
    }
    if (result.has_best_rf_feasible) {
        const Candidate& c = result.best_rf_feasible;
        std::cout << std::setw(9) << c.eval.sharing.sharing_cost
                  << std::setw(8) << c.eval.total_unique_count
                  << std::fixed << std::setprecision(3)
                  << std::setw(8) << c.eval.hardware_cost
                  << std::setw(9) << worst_rl(c)
                  << std::setw(9) << worst_il(c)
                  << std::setw(11) << worst_stop(c)
                  << std::setw(11) << worst_harmonic(c)
                  << "  RF-best\n";
    }
}

void print_result(const SearchResult& result,
                  const std::string& title,
                  const std::array<BandSpec, MODE_COUNT>& bands,
                  const Config& cfg) {
    const Candidate& best = result.best;
    const bool feasible = best.eval.all_modes_feasible;
    const bool block_design_output =
        cfg.require_feasible_output && !feasible;

    std::cout << "\n============================================================\n";
    std::cout << title << "\n";
    if (block_design_output) {
        std::cout << "NO FEASIBLE SOLUTION FOUND.\n";
        std::cout << "Program will exit with code 1.\n";
        std::cout << "No design CSV files have been written.\n";
        std::cout << "Diagnostic CSVs are written with prefix '"
                  << cfg.output_prefix << "'.\n";
    } else if (!feasible) {
        std::cout << "INFEASIBLE DEBUG CANDIDATE "
                  << "\xE2\x80\x94"
                  << " NOT A VALID DESIGN\n";
    }
    std::cout << "CTC-GA status: "
              << (feasible ? "FEASIBLE"
                           : "INFEASIBLE DEBUG CANDIDATE")
              << " | lambda_compression=" << result.lambda_compression
              << " | restart/gen=" << result.restart << "/"
              << result.generation
              << " | elapsed=" << std::fixed << std::setprecision(2)
              << result.elapsed_seconds << " s\n";
    std::cout << "Fitness components: rf_penalty=" << std::scientific
              << std::setprecision(4) << best.eval.rf_penalty
              << " rf_objective=" << best.eval.rf_objective
              << std::fixed
              << " sharing=" << best.eval.sharing_cost
              << " hardware=" << best.eval.hardware_cost
              << " baseline_distance=" << best.eval.baseline_distance_cost
              << " rf_margin=" << best.eval.rf_margin
              << "\n";
    std::cout << "Runtime counters: coarse=" << result.counters.coarse_evaluations
              << " mid=" << result.counters.mid_evaluations
              << " dense=" << result.counters.dense_evaluations
              << " cache_hits=" << result.counters.cache_hits
              << " local_repairs=" << result.counters.local_repair_calls
              << " sharing_accept=" << result.counters.accepted_sharing_moves
              << " sharing_rollback=" << result.counters.rolled_back_sharing_moves
              << " feasible_seen=" << result.counters.feasible_candidates_found
              << "\n";

    if (block_design_output) {
        print_metrics_table(best);
        print_failure_diagnostics(best, bands);
        print_archive(result);
        return;
    }

    print_cap_table(best);
    print_sharing_summary(best);
    print_metrics_table(best);
    print_failure_diagnostics(best, bands);
    print_hardware_summary(best);
    print_comparison(result.baseline, best);
    print_archive(result);
}

void write_best_cap_table_csv(const std::string& path, const Candidate& best) {
    std::ofstream out(path);
    if (!best.eval.all_modes_feasible) {
        out << "# INFEASIBLE DEBUG CANDIDATE -- NOT A VALID DESIGN\n";
    }
    out << "IsFeasible,Cell,N77_pF,N78_pF,N79_pF,Pattern,UniqueCount\n";
    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        out << yes_no(best.eval.all_modes_feasible) << ","
            << kCellNames[cell] << ","
            << pf(best.cap[MODE_N77][cell]) << ","
            << pf(best.cap[MODE_N78][cell]) << ","
            << pf(best.cap[MODE_N79][cell]) << ","
            << pattern_name(best.eval.sharing.pattern[cell]) << ","
            << best.eval.sharing.unique_count[cell] << "\n";
    }
}

void write_decomposition_csv(const std::string& path, const Candidate& best) {
    std::ofstream out(path);
    if (!best.eval.all_modes_feasible) {
        out << "# INFEASIBLE DEBUG CANDIDATE -- NOT A VALID DESIGN\n";
    }
    out << "Cell,N77_pF,N78_pF,N79_pF,Sorted0Mode,Cfix_pF,"
        << "Sorted1Mode,DeltaA_pF,Sorted2Mode,DeltaB_pF,"
        << "BranchCount,TinyBranchCount,LargeBranchCount,CellHardwareCost,Logic\n";
    for (const auto& d : best.eval.hardware.cells) {
        out << kCellNames[d.cell] << ","
            << pf(d.values[MODE_N77]) << ","
            << pf(d.values[MODE_N78]) << ","
            << pf(d.values[MODE_N79]) << ","
            << kModeNames[d.sorted_modes[0]] << ","
            << pf(d.fixed_f) << ","
            << kModeNames[d.sorted_modes[1]] << ","
            << pf(d.delta_a_f) << ","
            << kModeNames[d.sorted_modes[2]] << ","
            << pf(d.delta_b_f) << ","
            << d.branch_count << ","
            << d.tiny_branch_count << ","
            << d.large_branch_count << ","
            << d.cost << ","
            << "\"" << mode_logic_string(d) << "\"\n";
    }
}

void write_metrics_csv(const std::string& path, const Candidate& best) {
    std::ofstream out(path);
    out << "Mode,CandidateFeasible,ModeFeasible,PassRLMin_dB,PassILMax_dB,PassRLAvg_dB,"
        << "PassILAvg_dB,LowerStopMin_dB,UpperStopMin_dB,H2Min_dB,H3Min_dB,"
        << "TZ1_GHz,TZ2_GHz,TZ3_GHz,TZ4_GHz,RFPenalty,RFObjective,RFMargin\n";
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        const ModeMetrics& m = best.eval.modes[mode];
        out << kModeNames[mode] << ","
            << yes_no(best.eval.all_modes_feasible) << ","
            << yes_no(m.constraints_ok) << ","
            << m.min_pass_return_loss_db << ","
            << m.max_pass_insertion_loss_db << ","
            << m.avg_pass_return_loss_db << ","
            << m.avg_pass_insertion_loss_db << ","
            << m.lower_stop_min_rejection_db << ","
            << m.upper_stop_min_rejection_db << ","
            << m.harmonic2_min_rejection_db << ","
            << m.harmonic3_min_rejection_db << ","
            << ghz(m.zeros.tz1_hz) << ","
            << ghz(m.zeros.tz2_hz) << ","
            << ghz(m.zeros.tz3_hz) << ","
            << ghz(m.zeros.tz4_hz) << ","
            << m.rf_penalty << ","
            << m.rf_objective << ","
            << m.rf_margin << "\n";
    }
}

void write_failure_diagnostics_csv(
    const std::string& path,
    const Candidate& best,
    const std::array<BandSpec, MODE_COUNT>& bands) {
    std::ofstream out(path);
    out << "Mode,Metric,Actual,Target,Margin,Pass\n";
    auto row = [&](int mode,
                   const std::string& metric,
                   double actual,
                   double target,
                   double margin,
                   bool pass) {
        out << kModeNames[mode] << ","
            << metric << ","
            << actual << ","
            << target << ","
            << margin << ","
            << yes_no(pass) << "\n";
    };
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        const ModeMetrics& m = best.eval.modes[mode];
        const BandSpec& band = bands[mode];
        row(mode, "Pass_RL_min",
            m.min_pass_return_loss_db,
            band.pass_return_loss_min_db,
            m.min_pass_return_loss_db - band.pass_return_loss_min_db,
            m.pass_return_loss_ok);
        row(mode, "Pass_IL_max",
            m.max_pass_insertion_loss_db,
            band.pass_insertion_loss_max_db,
            band.pass_insertion_loss_max_db - m.max_pass_insertion_loss_db,
            m.pass_insertion_loss_ok);
        row(mode, "Lower_stop_min",
            m.lower_stop_min_rejection_db,
            band.stop_rejection_min_db,
            m.lower_stop_min_rejection_db - band.stop_rejection_min_db,
            m.lower_stop_ok);
        row(mode, "Upper_stop_min",
            m.upper_stop_min_rejection_db,
            band.stop_rejection_min_db,
            m.upper_stop_min_rejection_db - band.stop_rejection_min_db,
            m.upper_stop_ok);
        row(mode, "H2_min",
            m.harmonic2_min_rejection_db,
            band.harmonic_rejection_min_db,
            m.harmonic2_min_rejection_db - band.harmonic_rejection_min_db,
            m.harmonic2_ok);
        row(mode, "H3_min",
            m.harmonic3_min_rejection_db,
            band.harmonic_rejection_min_db,
            m.harmonic3_min_rejection_db - band.harmonic_rejection_min_db,
            m.harmonic3_ok);
        out << kModeNames[mode] << ",TZ_order,"
            << "\"tz1=" << ghz(m.zeros.tz1_hz)
            << ";tz2=" << ghz(m.zeros.tz2_hz)
            << ";tz3=" << ghz(m.zeros.tz3_hz)
            << ";tz4=" << ghz(m.zeros.tz4_hz) << "\","
            << "\"tz1,tz2<pass_lo;tz3,tz4>pass_hi;r4>1.40\",,"
            << yes_no(m.transmission_zero_order_ok) << "\n";
    }
}

void write_lambda_sweep_csv(const std::string& path,
                            const std::vector<SearchResult>& results) {
    std::ofstream out(path);
    out << "LambdaCompression,Feasible,UniqueTargetCount,SharingCost,"
        << "AllThreeShared,N77N78Shared,N77N79Shared,N78N79Shared,"
        << "HardwareCost,MaxIncrement_pF,NonzeroBranches,WorstRL_dB,"
        << "WorstIL_dB,WorstStop_dB,WorstHarmonic_dB,RFPenalty,RFObjective,"
        << "ElapsedSeconds\n";
    for (const auto& result : results) {
        const Candidate& c = result.best;
        out << result.lambda_compression << ","
            << yes_no(c.eval.all_modes_feasible) << ","
            << c.eval.total_unique_count << ","
            << c.eval.sharing.sharing_cost << ","
            << c.eval.sharing.all_three_count << ","
            << c.eval.sharing.pair_77_78_count << ","
            << c.eval.sharing.pair_77_79_count << ","
            << c.eval.sharing.pair_78_79_count << ","
            << c.eval.hardware_cost << ","
            << pf(c.eval.hardware.max_increment_f) << ","
            << c.eval.hardware.nonzero_branch_count << ","
            << worst_rl(c) << ","
            << worst_il(c) << ","
            << worst_stop(c) << ","
            << worst_harmonic(c) << ","
            << c.eval.rf_penalty << ","
            << c.eval.rf_objective << ","
            << result.elapsed_seconds << "\n";
    }
}

std::vector<std::string> write_csv_outputs(
    const Config& cfg,
    const Candidate& best,
    const std::vector<SearchResult>& results,
    const std::array<BandSpec, MODE_COUNT>& bands) {
    std::vector<std::string> written;
    const bool export_design =
        best.eval.all_modes_feasible || !cfg.require_feasible_output;

    if (export_design) {
        const std::string best_path =
            best.eval.all_modes_feasible
            ? cfg.output_prefix + "_best_cap_table.csv"
            : cfg.output_prefix + "_INFEASIBLE_DEBUG_cap_table.csv";
        const std::string decomp_path =
            best.eval.all_modes_feasible
            ? cfg.output_prefix + "_decomposition.csv"
            : cfg.output_prefix + "_INFEASIBLE_DEBUG_decomposition.csv";
        write_best_cap_table_csv(best_path, best);
        write_decomposition_csv(decomp_path, best);
        written.push_back(best_path);
        written.push_back(decomp_path);
    } else {
        std::cout << "\nSkipping final design CSV exports because no feasible "
                  << "solution was found.\n";
    }

    const std::string metrics_path = cfg.output_prefix + "_metrics.csv";
    const std::string diagnostics_path =
        cfg.output_prefix + "_failure_diagnostics.csv";
    const std::string sweep_path = cfg.output_prefix + "_lambda_sweep.csv";
    write_metrics_csv(metrics_path, best);
    write_failure_diagnostics_csv(diagnostics_path, best, bands);
    write_lambda_sweep_csv(sweep_path, results);
    written.push_back(metrics_path);
    written.push_back(diagnostics_path);
    written.push_back(sweep_path);
    return written;
}

void print_eval_baseline_only(
    const Config& cfg,
    const FixedValues& fixed,
    const std::array<BandSpec, MODE_COUNT>& bands,
    const std::array<FrequencyPlan, MODE_COUNT>& dense_plans,
    const Bounds& bounds,
    const CapTable& baseline) {
    Candidate c = make_candidate(baseline, bounds, cfg.mask);
    c.eval = evaluate_candidate(c.cap, fixed, bands, dense_plans, baseline, cfg);

    std::cout << "\n============================================================\n";
    std::cout << "Dense baseline evaluation only\n";
    std::cout << "Feasible: " << yes_no(c.eval.all_modes_feasible)
              << " | rf_penalty=" << std::scientific << std::setprecision(4)
              << c.eval.rf_penalty
              << " | rf_objective=" << c.eval.rf_objective
              << std::fixed << "\n";
    print_cap_table(c);
    print_sharing_summary(c);
    print_metrics_table(c);
    print_failure_diagnostics(c, bands);
    print_hardware_summary(c);
}

void run_single_cell_mask_scan(
    const Config& cfg,
    const FixedValues& fixed,
    const std::array<BandSpec, MODE_COUNT>& bands,
    const std::array<FrequencyPlan, MODE_COUNT>& dense_plans,
    const Bounds& bounds,
    const CapTable& baseline) {
    const std::array<SharePattern, 5> patterns = {{
        SHARE_NONE, SHARE_77_78, SHARE_77_79, SHARE_78_79, SHARE_ALL
    }};

    std::cout << "\nSingle-cell mask scan from baseline (dense evaluation, no GA polish)\n";
    std::cout << "Cell Pattern             Feas  RFPenalty      WorstRL  WorstIL"
              << "  Sharing  ImplCost\n";
    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        for (SharePattern pattern : patterns) {
            CapTable table = baseline;
            enforce_pattern(table, cell, pattern, bounds, cfg.mask);
            Candidate c = make_candidate(table, bounds, cfg.mask);
            c.eval = evaluate_candidate(c.cap, fixed, bands, dense_plans,
                                        baseline, cfg);
            std::cout << std::left << std::setw(5) << kCellNames[cell]
                      << std::setw(20) << pattern_name(pattern)
                      << std::right << std::setw(5)
                      << yes_no(c.eval.all_modes_feasible)
                      << std::scientific << std::setprecision(3)
                      << std::setw(13) << c.eval.rf_penalty
                      << std::fixed << std::setprecision(3)
                      << std::setw(9) << worst_rl(c)
                      << std::setw(9) << worst_il(c)
                      << std::setw(9) << c.eval.sharing.sharing_cost
                      << std::setw(10) << c.eval.hardware_cost
                      << "\n";
        }
    }
}

double rf_repair_score(const Candidate& c) {
    return c.eval.rf_penalty + c.eval.rf_objective;
}

void local_cap_repair(
    CapTable& cap,
    int cell,
    SearchContext& ctx,
    int max_iter = 30) {
    ++ctx.counters.local_repair_calls;
    const CapTable start_cap = cap;
    Candidate best = make_candidate(cap, ctx.bounds, ctx.cfg.mask);
    best.eval = ctx.evaluate_with_baseline(best.cap, EVAL_MID, start_cap);
    double best_score = rf_repair_score(best);
    const SharePattern locked_pattern =
        compute_sharing_report(best.cap).pattern[cell];

    static const std::array<int, 6> step_codes = {{-5, -2, -1, 1, 2, 5}};
    for (int iter = 0; iter < max_iter; ++iter) {
        bool improved = false;
        for (int mode = 0; mode < MODE_COUNT; ++mode) {
            if (improved) {
                break;
            }
            for (int step : step_codes) {
                CapTable trial_cap = best.cap;
                trial_cap[mode][cell] +=
                    static_cast<double>(step) * CAP_STEP;
                enforce_pattern(trial_cap, cell, locked_pattern, ctx.bounds,
                                ctx.cfg.mask);
                Candidate trial = make_candidate(trial_cap, ctx.bounds,
                                                 ctx.cfg.mask);
                trial.eval = ctx.evaluate_with_baseline(trial.cap, EVAL_MID,
                                                        start_cap);
                const double trial_score = rf_repair_score(trial);
                if (trial_score < best_score - 1e-12) {
                    best = trial;
                    best_score = trial_score;
                    improved = true;
                    break;
                }
            }
        }
        if (!improved) {
            break;
        }
    }

    cap = best.cap;
}

bool exactly_one_mode_failing(const Candidate& c, int& failing_mode) {
    failing_mode = -1;
    int fail_count = 0;
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        if (!c.eval.modes[mode].constraints_ok) {
            failing_mode = mode;
            ++fail_count;
        }
    }
    return fail_count == 1;
}

Candidate mode_wise_rf_repair(const Candidate& start, SearchContext& ctx) {
    int failing_mode = -1;
    if (!exactly_one_mode_failing(start, failing_mode)) {
        return start;
    }

    Candidate best = start;
    double best_score = rf_repair_score(best);
    static const std::array<int, 6> step_codes = {{-5, -2, -1, 1, 2, 5}};
    for (int pass = 0; pass < 10; ++pass) {
        bool improved = false;
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            if (ctx.cfg.mask.enabled && ctx.cfg.mask.specified[cell]) {
                continue;
            }
            for (int step : step_codes) {
                CapTable trial_cap = best.cap;
                trial_cap[failing_mode][cell] +=
                    static_cast<double>(step) * CAP_STEP;
                Candidate trial = make_candidate(trial_cap, ctx.bounds,
                                                 ctx.cfg.mask);
                trial.eval = ctx.evaluate(trial.cap, EVAL_MID);
                const double score = rf_repair_score(trial);
                if (score < best_score - 1e-12) {
                    best = trial;
                    best_score = score;
                    improved = true;
                    break;
                }
            }
            if (improved) {
                break;
            }
        }
        if (!improved) {
            break;
        }
    }
    return best;
}

Candidate local_polish_candidate(const Candidate& start,
                                 SearchContext& ctx,
                                 int max_steps) {
    if (!start.eval.all_modes_feasible) {
        return start;
    }

    Candidate best = start;
    static const std::array<int, 6> step_codes = {{-5, -2, -1, 1, 2, 5}};
    int accepted = 0;
    while (accepted < max_steps) {
        bool improved = false;
        for (int mode = 0; mode < MODE_COUNT && !improved; ++mode) {
            for (int cell = 0; cell < CELL_COUNT && !improved; ++cell) {
                if (ctx.cfg.mask.enabled && ctx.cfg.mask.specified[cell]) {
                    continue;
                }
                for (int step : step_codes) {
                    CapTable trial_cap = best.cap;
                    trial_cap[mode][cell] +=
                        static_cast<double>(step) * CAP_STEP;
                    Candidate trial = make_candidate(trial_cap, ctx.bounds,
                                                     ctx.cfg.mask);
                    trial.eval = ctx.evaluate(trial.cap, EVAL_MID);
                    if (!trial.eval.all_modes_feasible) {
                        continue;
                    }
                    trial = ctx.evaluate_candidate_at(trial, EVAL_DENSE);
                    if (trial.eval.all_modes_feasible &&
                        better_candidate(trial, best)) {
                        best = trial;
                        ++accepted;
                        improved = true;
                        break;
                    }
                }
            }
        }
        if (!improved) {
            break;
        }
    }
    return best;
}

bool apply_feasibility_preserving_sharing_mutation(
    Candidate& child,
    const Candidate& reference,
    SearchContext& ctx,
    std::mt19937_64& rng) {
    if (ctx.cfg.rf_only) {
        return false;
    }

    std::uniform_real_distribution<double> urand(0.0, 1.0);
    if (urand(rng) > ctx.cfg.compression_mutation_rate) {
        return false;
    }

    Candidate original = child;
    if (!std::isfinite(original.eval.rf_penalty) ||
        original.eval.rf_penalty >= INF * 0.5) {
        original.eval = ctx.evaluate(original.cap, EVAL_COARSE);
    }

    static const std::array<int, CELL_COUNT> cell_weights = {{
        3, 3, 1, 1, 3, 1, 1, 3, 1
    }};
    std::discrete_distribution<int> weighted_cell_dist(
        cell_weights.begin(), cell_weights.end());
    std::uniform_int_distribution<int> pattern_dist(1, 4);
    const int cell = weighted_cell_dist(rng);
    const SharePattern pattern =
        static_cast<SharePattern>(pattern_dist(rng));

    CapTable trial_cap = child.cap;
    enforce_pattern(trial_cap, cell, pattern, ctx.bounds, ctx.cfg.mask);
    Candidate trial = make_candidate(trial_cap, ctx.bounds, ctx.cfg.mask);
    trial.eval = ctx.evaluate_with_baseline(trial.cap, EVAL_COARSE,
                                            reference.cap);

    const double explode_threshold =
        original.eval.rf_penalty + std::max(1000.0, 0.20 * original.eval.rf_penalty);
    if (trial.eval.rf_penalty > explode_threshold) {
        ++ctx.counters.rolled_back_sharing_moves;
        child = original;
        return false;
    }

    local_cap_repair(trial.cap, cell, ctx);
    trial.eval = ctx.evaluate_with_baseline(trial.cap, EVAL_DENSE,
                                            reference.cap);
    const bool acceptable =
        trial.eval.all_modes_feasible ||
        trial.eval.rf_penalty <= original.eval.rf_penalty +
                                 std::max(10.0, 0.01 * original.eval.rf_penalty);
    if (acceptable) {
        child = trial;
        ++ctx.counters.accepted_sharing_moves;
        return true;
    }

    child = original;
    ++ctx.counters.rolled_back_sharing_moves;
    return false;
}

Candidate run_progressive_sharing(const Candidate& start, SearchContext& ctx) {
    if (!start.eval.all_modes_feasible) {
        return start;
    }

    Candidate working = make_candidate(start.cap, ctx.bounds, ctx.cfg.mask);
    working.eval = ctx.evaluate_with_baseline(working.cap, EVAL_DENSE,
                                              start.cap);
    if (!working.eval.all_modes_feasible) {
        return start;
    }

    struct SharingOption {
        int cell = 0;
        SharePattern pattern = SHARE_NONE;
        double rf_penalty_increase = 0.0;
    };

    const std::array<SharePattern, 4> patterns = {{
        SHARE_77_78, SHARE_77_79, SHARE_78_79, SHARE_ALL
    }};
    std::vector<SharingOption> options;
    options.reserve(CELL_COUNT * patterns.size());
    const double base_penalty = working.eval.rf_penalty;
    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        for (SharePattern pattern : patterns) {
            CapTable trial_cap = working.cap;
            enforce_pattern(trial_cap, cell, pattern, ctx.bounds,
                            ctx.cfg.mask);
            Candidate trial = make_candidate(trial_cap, ctx.bounds,
                                             ctx.cfg.mask);
            trial.eval = ctx.evaluate_with_baseline(trial.cap, EVAL_MID,
                                                    start.cap);
            options.push_back({cell, pattern,
                               trial.eval.rf_penalty - base_penalty});
        }
    }

    std::sort(options.begin(), options.end(),
              [](const SharingOption& a, const SharingOption& b) {
                  if (nearly_less(a.rf_penalty_increase,
                                  b.rf_penalty_increase)) {
                      return true;
                  }
                  if (nearly_less(b.rf_penalty_increase,
                                  a.rf_penalty_increase)) {
                      return false;
                  }
                  if (a.cell != b.cell) {
                      return a.cell < b.cell;
                  }
                  return static_cast<int>(a.pattern) <
                         static_cast<int>(b.pattern);
              });

    std::map<int, int> failed_patterns;
    for (const SharingOption& option : options) {
        const int failure_key =
            option.cell * 10 + static_cast<int>(option.pattern);
        if (failed_patterns[failure_key] >= ctx.cfg.pattern_failure_limit) {
            continue;
        }
        CapTable trial_cap = working.cap;
        enforce_pattern(trial_cap, option.cell, option.pattern, ctx.bounds,
                        ctx.cfg.mask);
        Candidate coarse = make_candidate(trial_cap, ctx.bounds,
                                          ctx.cfg.mask);
        coarse.eval = ctx.evaluate_with_baseline(coarse.cap, EVAL_COARSE,
                                                 working.cap);
        const double explode_threshold =
            working.eval.rf_penalty + std::max(1.0, 0.05 * working.eval.rf_penalty);
        if (coarse.eval.rf_penalty > explode_threshold) {
            ++ctx.counters.rolled_back_sharing_moves;
            ++failed_patterns[failure_key];
            continue;
        }

        local_cap_repair(trial_cap, option.cell, ctx);
        Candidate trial = make_candidate(trial_cap, ctx.bounds, ctx.cfg.mask);
        trial.eval = ctx.evaluate_with_baseline(trial.cap, EVAL_DENSE,
                                                start.cap);
        if (trial.eval.all_modes_feasible) {
            const int before_sharing = working.eval.sharing.sharing_cost;
            working = trial;
            ++ctx.counters.accepted_sharing_moves;
            if (!ctx.cfg.quiet) {
                std::cout << "[progressive-sharing] accepted "
                          << kCellNames[option.cell] << " "
                          << pattern_name(option.pattern)
                          << " sharing_cost " << before_sharing
                          << " -> " << working.eval.sharing.sharing_cost
                          << " rf_penalty=" << std::scientific
                          << std::setprecision(3)
                          << working.eval.rf_penalty << std::fixed << "\n";
            }
        } else {
            ++ctx.counters.rolled_back_sharing_moves;
            ++failed_patterns[failure_key];
        }
    }

    return working;
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

bool parse_double_arg(const char* text, double& out) {
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0') {
        return false;
    }
    out = value;
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

std::string uppercase(std::string text) {
    for (char& ch : text) {
        if (ch >= 'a' && ch <= 'z') {
            ch = static_cast<char>(ch - 'a' + 'A');
        } else if (ch == '-' || ch == '=') {
            ch = '_';
        }
    }
    return text;
}

std::string trim(std::string text) {
    const auto is_space = [](unsigned char ch) {
        return std::isspace(ch) != 0;
    };
    while (!text.empty() && is_space(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && is_space(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    return text;
}

int parse_cell_name(const std::string& text) {
    const std::string t = uppercase(trim(text));
    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        if (t == kCellNames[cell]) {
            return cell;
        }
    }
    return -1;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream stream(line);
    while (std::getline(stream, field, ',')) {
        fields.push_back(trim(field));
    }
    return fields;
}

bool load_baseline_csv(const std::string& path, CapTable& baseline) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot open --baseline-csv file: " << path << "\n";
        return false;
    }

    std::array<bool, CELL_COUNT> seen{};
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> fields = split_csv_line(line);
        if (fields.empty()) {
            continue;
        }
        if (uppercase(fields[0]) == "CELL") {
            continue;
        }
        if (fields.size() < 4) {
            std::cerr << "Bad baseline CSV line " << line_no
                      << ": expected Cell,N77,N78,N79\n";
            return false;
        }

        const int cell = parse_cell_name(fields[0]);
        double n77 = 0.0;
        double n78 = 0.0;
        double n79 = 0.0;
        if (cell < 0 ||
            !parse_double_arg(fields[1].c_str(), n77) ||
            !parse_double_arg(fields[2].c_str(), n78) ||
            !parse_double_arg(fields[3].c_str(), n79)) {
            std::cerr << "Bad baseline CSV line " << line_no << ": "
                      << line << "\n";
            return false;
        }
        baseline[MODE_N77][cell] = n77 * PF;
        baseline[MODE_N78][cell] = n78 * PF;
        baseline[MODE_N79][cell] = n79 * PF;
        seen[cell] = true;
    }

    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        if (!seen[cell]) {
            std::cerr << "Baseline CSV is missing row "
                      << kCellNames[cell] << "\n";
            return false;
        }
    }
    return true;
}

void apply_legacy_fixed_c1_c5(Bounds& bounds,
                              CapTable& baseline,
                              MaskConstraints& mask) {
    constexpr double legacy_c1 = 0.20 * PF;
    constexpr double legacy_c5 = 1.14 * PF;
    bounds.lower[CELL_C1] = legacy_c1;
    bounds.upper[CELL_C1] = legacy_c1;
    bounds.lower[CELL_C5] = legacy_c5;
    bounds.upper[CELL_C5] = legacy_c5;
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        baseline[mode][CELL_C1] = legacy_c1;
        baseline[mode][CELL_C5] = legacy_c5;
    }
    mask.enabled = true;
    mask.specified[CELL_C1] = true;
    mask.pattern[CELL_C1] = SHARE_ALL;
    mask.specified[CELL_C5] = true;
    mask.pattern[CELL_C5] = SHARE_ALL;
}

bool parse_share_pattern(const std::string& text, SharePattern& pattern) {
    const std::string t = uppercase(trim(text));
    if (t == "ALL" || t == "SHARE_ALL" || t == "ALL_SHARED") {
        pattern = SHARE_ALL;
        return true;
    }
    if (t == "77_78" || t == "N77_N78" || t == "N77_N78_SHARED") {
        pattern = SHARE_77_78;
        return true;
    }
    if (t == "77_79" || t == "N77_N79" || t == "N77_N79_SHARED") {
        pattern = SHARE_77_79;
        return true;
    }
    if (t == "78_79" || t == "N78_N79" || t == "N78_N79_SHARED") {
        pattern = SHARE_78_79;
        return true;
    }
    if (t == "NONE" || t == "DIFF" || t == "ALL_DIFFERENT") {
        pattern = SHARE_NONE;
        return true;
    }
    return false;
}

std::string optimizer_name(OptimizerKind optimizer) {
    switch (optimizer) {
    case OPT_GA:
        return "ga";
    case OPT_JDE:
        return "jde";
    case OPT_DE:
        return "de";
    case OPT_HYBRID:
        return "hybrid";
    }
    return "ga";
}

bool parse_optimizer_kind(const std::string& text, OptimizerKind& optimizer) {
    const std::string t = uppercase(trim(text));
    if (t == "GA") {
        optimizer = OPT_GA;
        return true;
    }
    if (t == "JDE") {
        optimizer = OPT_JDE;
        return true;
    }
    if (t == "DE") {
        optimizer = OPT_DE;
        return true;
    }
    if (t == "HYBRID") {
        optimizer = OPT_HYBRID;
        return true;
    }
    return false;
}

bool parse_mask_test(const std::string& text, MaskConstraints& mask) {
    mask.enabled = true;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token =
            trim(text.substr(start, comma == std::string::npos
                                        ? std::string::npos
                                        : comma - start));
        const std::size_t colon = token.find(':');
        if (colon == std::string::npos) {
            std::cerr << "Bad --mask-test token: " << token << "\n";
            return false;
        }
        const int cell = parse_cell_name(token.substr(0, colon));
        SharePattern pattern = SHARE_NONE;
        if (cell < 0 || !parse_share_pattern(token.substr(colon + 1), pattern)) {
            std::cerr << "Bad --mask-test token: " << token << "\n";
            return false;
        }
        mask.specified[cell] = true;
        mask.pattern[cell] = pattern;
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return true;
}

void print_help() {
    std::cout
        << "Heuristic RF-guided CTC-GA capacitance-table compression optimizer\n"
        << "Options:\n"
        << "  --population N\n"
        << "  --generations N\n"
        << "  --restarts N\n"
        << "  --seed N\n"
        << "  --threads N\n"
        << "  --dense-check-interval N\n"
        << "  --elite-dense-interval N\n"
        << "  --mid-check-interval N\n"
        << "  --mid-top-count N\n"
        << "  --dense-top-count N\n"
        << "  --optimizer ga|jde|de|hybrid\n"
        << "  --lambda-compression X   (alias: --lambda-share)\n"
        << "  --lambda-hardware X      (alias: --lambda-hw)\n"
        << "  --lambda-base X\n"
        << "  --local-jitter-scale X   (alias: --local-span)\n"
        << "  --require-feasible-output\n"
        << "  --allow-infeasible-output\n"
        << "  --rf-only\n"
        << "  --two-stage\n"
        << "  --semi-random-init\n"
        << "  --init-seed-fraction X\n"
        << "  --init-local-jitter-fraction X\n"
        << "  --init-wide-jitter-fraction X\n"
        << "  --init-random-fraction X\n"
        << "  --init-share-pattern-fraction X\n"
        << "  --min-random-fraction X\n"
        << "  --local-jitter-sigma X\n"
        << "  --wide-jitter-sigma X\n"
        << "  --random-seed-blend-min X\n"
        << "  --random-seed-blend-max X\n"
        << "  --stagnation-threshold N\n"
        << "  --stagnation-inject-fraction X\n"
        << "  --enable-early-exit\n"
        << "  --jde-mode\n"
        << "  --jde-F X\n"
        << "  --jde-CR X\n"
        << "  --island-model\n"
        << "  --islands N\n"
        << "  --migration-interval N\n"
        << "  --migration-count N\n"
        << "  --lhs-random-init\n"
        << "  --progressive-sharing\n"
        << "  --pattern-failure-limit N\n"
        << "  --pattern-first\n"
        << "  --local-polish\n"
        << "  --local-polish-steps N\n"
        << "  --archive-per-sharing N\n"
        << "  --benchmark-short\n"
        << "  --baseline-csv FILE\n"
        << "  --eval-baseline-only\n"
        << "  --single-cell-mask-scan\n"
        << "  --legacy-fixed-c1-c5\n"
        << "  --output-prefix NAME\n"
        << "  --use-share-patterns\n"
        << "  --mask-test C1:ALL,C2:ALL,C5:ALL,C9:77_79\n"
        << "  --sweep-lambda-share\n"
        << "  --quiet\n"
        << "\nCompile examples:\n"
        << "  Linux / WSL / MinGW with OpenMP:\n"
        << "    g++ -std=c++17 -O3 -march=native -fopenmp \"firstcheck(a2).cpp\" -o ctc_ga\n"
        << "  Without OpenMP:\n"
        << "    g++ -std=c++17 -O3 -march=native \"firstcheck(a2).cpp\" -o ctc_ga\n";
}

Config parse_config(int argc, char** argv) {
    Config cfg;
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
        } else if (arg == "--threads") {
            parse_int_arg(need_value(arg), cfg.threads);
        } else if (arg == "--dense-check-interval") {
            parse_int_arg(need_value(arg), cfg.dense_check_interval);
        } else if (arg == "--elite-dense-interval") {
            parse_int_arg(need_value(arg), cfg.elite_dense_interval);
        } else if (arg == "--mid-check-interval") {
            parse_int_arg(need_value(arg), cfg.mid_check_interval);
        } else if (arg == "--mid-top-count") {
            parse_int_arg(need_value(arg), cfg.mid_top_count);
        } else if (arg == "--dense-top-count") {
            parse_int_arg(need_value(arg), cfg.dense_top_count);
        } else if (arg == "--optimizer") {
            if (!parse_optimizer_kind(need_value(arg), cfg.optimizer)) {
                std::cerr << "Bad --optimizer value; use ga, jde, de, or hybrid\n";
                std::exit(2);
            }
        } else if (arg == "--lambda-compression" ||
                   arg == "--lambda-share") {
            parse_double_arg(need_value(arg), cfg.lambda_compression);
        } else if (arg == "--lambda-hardware" ||
                   arg == "--lambda-hw") {
            parse_double_arg(need_value(arg), cfg.lambda_hardware);
        } else if (arg == "--lambda-base") {
            parse_double_arg(need_value(arg), cfg.lambda_base);
            cfg.lambda_base_explicit = true;
        } else if (arg == "--local-jitter-scale" ||
                   arg == "--local-span") {
            parse_double_arg(need_value(arg), cfg.local_jitter_scale);
        } else if (arg == "--require-feasible-output") {
            cfg.require_feasible_output = true;
        } else if (arg == "--allow-infeasible-output") {
            cfg.require_feasible_output = false;
        } else if (arg == "--rf-only") {
            cfg.rf_only = true;
        } else if (arg == "--two-stage") {
            cfg.two_stage = true;
        } else if (arg == "--semi-random-init") {
            cfg.semi_random_init = true;
        } else if (arg == "--init-seed-fraction") {
            parse_double_arg(need_value(arg), cfg.init_seed_fraction);
        } else if (arg == "--init-local-jitter-fraction") {
            parse_double_arg(need_value(arg), cfg.init_local_jitter_fraction);
        } else if (arg == "--init-wide-jitter-fraction") {
            parse_double_arg(need_value(arg), cfg.init_wide_jitter_fraction);
        } else if (arg == "--init-random-fraction") {
            parse_double_arg(need_value(arg), cfg.init_random_fraction);
        } else if (arg == "--init-share-pattern-fraction") {
            parse_double_arg(need_value(arg), cfg.init_share_pattern_fraction);
        } else if (arg == "--min-random-fraction") {
            parse_double_arg(need_value(arg), cfg.min_random_fraction);
        } else if (arg == "--local-jitter-sigma") {
            parse_double_arg(need_value(arg), cfg.local_jitter_sigma);
        } else if (arg == "--wide-jitter-sigma") {
            parse_double_arg(need_value(arg), cfg.wide_jitter_sigma);
        } else if (arg == "--random-seed-blend-min") {
            parse_double_arg(need_value(arg), cfg.random_seed_blend_min);
        } else if (arg == "--random-seed-blend-max") {
            parse_double_arg(need_value(arg), cfg.random_seed_blend_max);
        } else if (arg == "--stagnation-threshold") {
            parse_int_arg(need_value(arg), cfg.stagnation_threshold);
        } else if (arg == "--stagnation-inject-fraction") {
            parse_double_arg(need_value(arg), cfg.stagnation_inject_fraction);
        } else if (arg == "--enable-early-exit") {
            cfg.enable_early_exit = true;
        } else if (arg == "--jde-mode") {
            cfg.jde_mode = true;
            cfg.optimizer = OPT_JDE;
        } else if (arg == "--jde-F") {
            parse_double_arg(need_value(arg), cfg.jde_F_init);
        } else if (arg == "--jde-CR") {
            parse_double_arg(need_value(arg), cfg.jde_CR_init);
        } else if (arg == "--island-model") {
            cfg.island_model = true;
        } else if (arg == "--islands") {
            parse_int_arg(need_value(arg), cfg.islands);
        } else if (arg == "--migration-interval") {
            parse_int_arg(need_value(arg), cfg.migration_interval);
        } else if (arg == "--migration-count") {
            parse_int_arg(need_value(arg), cfg.migration_count);
        } else if (arg == "--lhs-random-init") {
            cfg.lhs_random_init = true;
        } else if (arg == "--baseline-csv") {
            cfg.baseline_csv_path = need_value(arg);
        } else if (arg == "--eval-baseline-only") {
            cfg.eval_baseline_only = true;
        } else if (arg == "--single-cell-mask-scan") {
            cfg.single_cell_mask_scan = true;
        } else if (arg == "--progressive-sharing") {
            cfg.progressive_sharing = true;
        } else if (arg == "--pattern-failure-limit") {
            parse_int_arg(need_value(arg), cfg.pattern_failure_limit);
        } else if (arg == "--pattern-first") {
            cfg.pattern_first = true;
            cfg.progressive_sharing = true;
            cfg.two_stage = true;
        } else if (arg == "--local-polish") {
            cfg.local_polish = true;
        } else if (arg == "--local-polish-steps") {
            parse_int_arg(need_value(arg), cfg.local_polish_steps);
        } else if (arg == "--legacy-fixed-c1-c5") {
            cfg.legacy_fixed_c1_c5 = true;
        } else if (arg == "--output-prefix") {
            cfg.output_prefix = need_value(arg);
        } else if (arg == "--use-share-patterns") {
            cfg.use_share_patterns = true;
            cfg.compression_mutation_rate = std::max(cfg.compression_mutation_rate,
                                                     0.30);
        } else if (arg == "--mask-test") {
            if (!parse_mask_test(need_value(arg), cfg.mask)) {
                std::exit(2);
            }
        } else if (arg == "--sweep-lambda-share") {
            cfg.sweep_lambda_share = true;
        } else if (arg == "--archive-per-sharing") {
            parse_int_arg(need_value(arg), cfg.archive_per_sharing);
        } else if (arg == "--benchmark-short") {
            cfg.benchmark_short = true;
            cfg.population_size = 40;
            cfg.max_generations = 3;
            cfg.max_restarts = 1;
            cfg.seed = 20260508ULL;
            cfg.dense_top_count = 8;
            cfg.mid_top_count = 12;
            cfg.output_prefix = "ctc_ga_benchmark_short";
        } else if (arg == "--quiet") {
            cfg.quiet = true;
        } else if (arg == "--help") {
            print_help();
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_help();
            std::exit(2);
        }
    }

    cfg.population_size = std::max(cfg.population_size, 30);
    cfg.max_generations = std::max(cfg.max_generations, 0);
    cfg.max_restarts = std::max(cfg.max_restarts, 1);
    cfg.dense_check_interval = std::max(cfg.dense_check_interval, 1);
    cfg.elite_dense_interval = std::max(cfg.elite_dense_interval, 1);
    cfg.mid_check_interval = std::max(cfg.mid_check_interval, 1);
    cfg.mid_top_count = std::max(cfg.mid_top_count, 1);
    cfg.dense_top_count = std::max(cfg.dense_top_count, 1);
    cfg.archive_per_sharing = std::max(cfg.archive_per_sharing, 1);
    cfg.elite_count = std::min(std::max(2, cfg.elite_count),
                               std::max(2, cfg.population_size / 3));
    cfg.tournament_size = std::max(2, cfg.tournament_size);
    cfg.local_jitter_scale = clamp_value(cfg.local_jitter_scale, 0.0, 5.0);
    cfg.init_seed_fraction = std::max(0.0, cfg.init_seed_fraction);
    cfg.init_local_jitter_fraction =
        std::max(0.0, cfg.init_local_jitter_fraction);
    cfg.init_wide_jitter_fraction =
        std::max(0.0, cfg.init_wide_jitter_fraction);
    cfg.init_random_fraction = std::max(0.0, cfg.init_random_fraction);
    cfg.init_share_pattern_fraction =
        std::max(0.0, cfg.init_share_pattern_fraction);
    cfg.min_random_fraction = clamp_value(cfg.min_random_fraction, 0.0, 1.0);
    cfg.local_jitter_sigma = clamp_value(cfg.local_jitter_sigma, 0.0, 2.0);
    cfg.wide_jitter_sigma = clamp_value(cfg.wide_jitter_sigma, 0.0, 2.0);
    cfg.random_seed_blend_min =
        clamp_value(cfg.random_seed_blend_min, 0.0, 1.0);
    cfg.random_seed_blend_max =
        clamp_value(cfg.random_seed_blend_max, 0.0, 1.0);
    if (cfg.random_seed_blend_min > cfg.random_seed_blend_max) {
        std::swap(cfg.random_seed_blend_min, cfg.random_seed_blend_max);
    }
    cfg.lambda_compression = std::max(0.0, cfg.lambda_compression);
    cfg.lambda_hardware = std::max(0.0, cfg.lambda_hardware);
    cfg.lambda_base = std::max(0.0, cfg.lambda_base);
    cfg.stagnation_threshold = std::max(cfg.stagnation_threshold, 1);
    cfg.stagnation_inject_fraction =
        clamp_value(cfg.stagnation_inject_fraction, 0.0, 1.0);
    cfg.jde_F_init = clamp_value(cfg.jde_F_init, 0.1, 1.0);
    cfg.jde_CR_init = clamp_value(cfg.jde_CR_init, 0.0, 1.0);
    cfg.jde_tau1 = clamp_value(cfg.jde_tau1, 0.0, 1.0);
    cfg.jde_tau2 = clamp_value(cfg.jde_tau2, 0.0, 1.0);
    cfg.islands = std::max(cfg.islands, 1);
    cfg.migration_interval = std::max(cfg.migration_interval, 1);
    cfg.migration_count = std::max(cfg.migration_count, 1);
    cfg.pattern_failure_limit = std::max(cfg.pattern_failure_limit, 1);
    cfg.local_polish_steps = std::max(cfg.local_polish_steps, 1);
    if (cfg.rf_only) {
        cfg.lambda_compression = 0.0;
        cfg.lambda_hardware = 0.0;
        cfg.lambda_base = 0.0;
    }
    return cfg;
}

void print_startup(const Config& cfg,
                   const FixedValues& fixed,
                   const std::array<BandSpec, MODE_COUNT>& bands) {
    std::cout << "CTC-GA: heuristic RF-guided capacitance-table compression\n";
    std::cout << "Candidate shape: cap[3 modes][9 cells] for C1-C9\n";
    std::cout << "Fixed inductors: L1=" << fixed.l1 / NH
              << " nH, L2=" << fixed.l2 / NH
              << " nH, L3=" << fixed.l3 / NH
              << " nH, L4=" << fixed.l4 / NH
              << " nH, L5=" << fixed.l5 / NH << " nH\n";
    if (cfg.legacy_fixed_c1_c5) {
        std::cout << "Legacy comparison mode: C1=0.20 pF and C5=1.14 pF "
                  << "are fixed all-shared.\n";
    } else {
        std::cout << "C1-C9 are optimization variables; baseline C1/C5 are "
                  << "seeds, not fixed.\n";
    }
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        std::cout << "  " << bands[mode].name << " passband: "
                  << ghz(bands[mode].pass_lo_hz) << "-"
                  << ghz(bands[mode].pass_hi_hz) << " GHz"
                  << " | full-pass RL >= "
                  << bands[mode].pass_return_loss_min_db
                  << " dB, IL <= "
                  << bands[mode].pass_insertion_loss_max_db
                  << " dB\n";
    }
    std::cout << "GA config: population=" << cfg.population_size
              << ", generations=" << cfg.max_generations
              << ", restarts=" << cfg.max_restarts
              << ", cap_step=" << pf(CAP_STEP) << " pF"
              << ", seed=" << cfg.seed
              << ", rf_only=" << yes_no(cfg.rf_only)
              << ", two_stage=" << yes_no(cfg.two_stage)
              << ", optimizer=" << optimizer_name(cfg.optimizer)
              << ", jde_mode=" << yes_no(cfg.jde_mode)
              << ", progressive_sharing=" << yes_no(cfg.progressive_sharing)
              << ", pattern_first=" << yes_no(cfg.pattern_first)
              << ", local_polish=" << yes_no(cfg.local_polish)
              << ", require_feasible_output="
              << yes_no(cfg.require_feasible_output)
              << ", semi_random_init=" << yes_no(cfg.semi_random_init)
              << ", lambda_compression=" << cfg.lambda_compression
              << ", lambda_hardware=" << cfg.lambda_hardware
              << ", lambda_base=" << cfg.lambda_base
              << ", local_jitter_scale=" << cfg.local_jitter_scale
              << ", local_jitter_sigma=" << cfg.local_jitter_sigma
              << ", wide_jitter_sigma=" << cfg.wide_jitter_sigma
              << ", dense_check_interval=" << cfg.dense_check_interval
              << ", elite_dense_interval=" << cfg.elite_dense_interval
              << ", mid_check_interval=" << cfg.mid_check_interval
              << ", mid_top_count=" << cfg.mid_top_count
              << ", dense_top_count=" << cfg.dense_top_count
              << ", archive_per_sharing=" << cfg.archive_per_sharing
              << ", stagnation_threshold=" << cfg.stagnation_threshold
              << ", stagnation_inject_fraction="
              << cfg.stagnation_inject_fraction
              << ", early_exit=" << yes_no(cfg.enable_early_exit)
              << ", island_model=" << yes_no(cfg.island_model)
              << ", islands=" << cfg.islands
              << ", share_patterns=" << yes_no(cfg.use_share_patterns)
              << "\n";
    std::cout << "Initial population fractions: seed="
              << cfg.init_seed_fraction
              << ", local_jitter=" << cfg.init_local_jitter_fraction
              << ", wide_jitter=" << cfg.init_wide_jitter_fraction
              << ", random=" << cfg.init_random_fraction
              << ", share_pattern=" << cfg.init_share_pattern_fraction
              << ", min_random=" << cfg.min_random_fraction
              << ", lhs_random=" << yes_no(cfg.lhs_random_init)
              << ", random_seed_blend=" << cfg.random_seed_blend_min
              << "-" << cfg.random_seed_blend_max << "\n";
    if (cfg.jde_mode) {
        std::cout << "jDE config: F_init=" << cfg.jde_F_init
                  << ", CR_init=" << cfg.jde_CR_init
                  << ", tau1=" << cfg.jde_tau1
                  << ", tau2=" << cfg.jde_tau2 << "\n";
    }
#ifdef _OPENMP
    std::cout << "OpenMP enabled";
    if (cfg.threads > 0) {
        std::cout << ", requested threads=" << cfg.threads;
    }
    std::cout << ", max threads=" << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP not enabled; running single-threaded. "
              << "Compile with -fopenmp for parallel evaluation.\n";
#endif
    if (!cfg.baseline_csv_path.empty()) {
        std::cout << "Baseline CSV: " << cfg.baseline_csv_path << "\n";
    }
    if (cfg.mask.enabled) {
        std::cout << "Mask-test constraints:";
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            if (cfg.mask.specified[cell]) {
                std::cout << " " << kCellNames[cell] << ":"
                          << pattern_name(cfg.mask.pattern[cell]);
            }
        }
        std::cout << "\n";
    }
}

SearchResult choose_best_result(const std::vector<SearchResult>& results) {
    SearchResult best = results.front();
    for (std::size_t i = 1; i < results.size(); ++i) {
        if (better_candidate(results[i].best, best.best)) {
            best = results[i];
        }
    }
    // Prefer any feasible result over an infeasible best.
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (results[i].has_best_rf_feasible &&
            !best.best.eval.all_modes_feasible) {
            best = results[i];
            best.best = results[i].best_rf_feasible;
        }
    }
    return best;
}

} // namespace

int main(int argc, char** argv) {
    const FixedValues fixed;
    const auto bands = make_bands();
    const auto coarse_plans = make_frequency_plans(bands, false);
    const auto mid_plans = make_frequency_plans_tier(bands, EVAL_MID);
    const auto dense_plans = make_frequency_plans(bands, true);
    Config cfg = parse_config(argc, argv);
#ifdef _OPENMP
    if (cfg.threads > 0) {
        omp_set_num_threads(cfg.threads);
    }
#endif
    Bounds bounds = make_bounds();
    CapTable baseline = make_baseline_table();

    if (!cfg.baseline_csv_path.empty() &&
        !load_baseline_csv(cfg.baseline_csv_path, baseline)) {
        return 2;
    }
    if (cfg.legacy_fixed_c1_c5) {
        apply_legacy_fixed_c1_c5(bounds, baseline, cfg.mask);
    }
    repair_table(baseline, bounds, cfg.mask);

    print_startup(cfg, fixed, bands);

    if (cfg.eval_baseline_only) {
        print_eval_baseline_only(cfg, fixed, bands, dense_plans, bounds, baseline);
        return 0;
    }

    if (cfg.single_cell_mask_scan) {
        run_single_cell_mask_scan(cfg, fixed, bands, dense_plans, bounds, baseline);
        return 0;
    }

    std::vector<SearchResult> results;
    SearchResult best;

    if (cfg.two_stage && !cfg.rf_only) {
        Config rf_cfg = cfg;
        rf_cfg.rf_only = true;
        rf_cfg.require_feasible_output = true;
        rf_cfg.lambda_compression = 0.0;
        rf_cfg.lambda_hardware = 0.0;
        rf_cfg.lambda_base = 0.0;
        rf_cfg.sweep_lambda_share = false;

        SearchResult rf_result = run_ctc_ga(rf_cfg, fixed, bands,
                                            coarse_plans, mid_plans,
                                            dense_plans,
                                            bounds, baseline, "RF");
        results.push_back(rf_result);
        print_result(rf_result, "Stage 1 RF feasibility search", bands,
                     rf_cfg);

        if (!rf_result.best.eval.all_modes_feasible) {
            const std::vector<std::string> written =
                write_csv_outputs(rf_cfg, rf_result.best, results, bands);
            std::cout << "\nCSV files written with prefix '"
                      << cfg.output_prefix << "':\n";
            for (const std::string& path : written) {
                std::cout << "  " << path << "\n";
            }
            return 1;
        }

        Config compress_cfg = cfg;
        compress_cfg.rf_only = false;
        if (!compress_cfg.lambda_base_explicit) {
            compress_cfg.lambda_base = 0.02;
        }

        SearchResult compress_result = run_ctc_ga(
            compress_cfg, fixed, bands, coarse_plans, mid_plans, dense_plans,
            bounds, rf_result.best.cap, "COMPRESS");
        results.push_back(compress_result);
        print_result(compress_result, "Stage 2 sharing/hardware compression",
                     bands, compress_cfg);
        best = compress_result;
    } else {
        std::vector<double> lambdas;
        if (cfg.sweep_lambda_share && !cfg.rf_only) {
            lambdas = {1.0, 3.0, 10.0, 30.0, 100.0};
        } else {
            lambdas = {cfg.lambda_compression};
        }

        results.reserve(lambdas.size());
        for (double lambda : lambdas) {
            Config run_cfg = cfg;
            run_cfg.lambda_compression = cfg.rf_only ? 0.0 : lambda;
            if (run_cfg.rf_only) {
                run_cfg.lambda_hardware = 0.0;
                run_cfg.lambda_base = 0.0;
            }
            const std::string stage = run_cfg.rf_only ? "RF" : "COMPRESS";
            SearchResult result = run_ctc_ga(run_cfg, fixed, bands,
                                             coarse_plans, mid_plans,
                                             dense_plans,
                                             bounds, baseline, stage);
            results.push_back(result);
            print_result(result, "Lambda result", bands, run_cfg);
        }

        best = choose_best_result(results);
        if (results.size() > 1) {
            print_result(best, "Best trade-off selected from lambda sweep",
                         bands, cfg);
        }
    }

    if (cfg.progressive_sharing && best.best.eval.all_modes_feasible) {
        const Candidate before = best.best;
        SearchContext sharing_ctx{cfg, fixed, bands, coarse_plans, mid_plans,
                                  dense_plans, bounds, baseline, {}, {}};
        Candidate after = run_progressive_sharing(before, sharing_ctx);
        if (cfg.local_polish && after.eval.all_modes_feasible) {
            after = local_polish_candidate(after, sharing_ctx,
                                           cfg.local_polish_steps);
        }
        best.best = after;
        best.best_rf_feasible = after;
        best.has_best_rf_feasible = after.eval.all_modes_feasible;
        best.counters.add(sharing_ctx.counters);
        if (!cfg.quiet) {
            std::cout << "\nProgressive sharing summary\n"
                      << "  Sharing cost : "
                      << before.eval.sharing.sharing_cost << " -> "
                      << after.eval.sharing.sharing_cost << "\n"
                      << "  Unique count : "
                      << before.eval.total_unique_count << " -> "
                      << after.eval.total_unique_count << "\n";
        }
    }

    const std::vector<std::string> written =
        write_csv_outputs(cfg, best.best, results, bands);
    std::cout << "\nCSV files written with prefix '" << cfg.output_prefix
              << "':\n";
    for (const std::string& path : written) {
        std::cout << "  " << path << "\n";
    }

    if (!best.best.eval.all_modes_feasible && cfg.require_feasible_output) {
        std::cout << "FINAL_RESULT = NO_FEASIBLE_SOLUTION\n";
        return 1;
    }
    if (best.best.eval.all_modes_feasible) {
        const bool compressed =
            best.best.eval.sharing.sharing_cost <
            best.baseline.eval.sharing.sharing_cost;
        std::cout << "FINAL_RESULT = "
                  << (compressed
                      ? "FEASIBLE_COMPRESSED_DESIGN"
                      : "FEASIBLE_UNCOMPRESSED_DESIGN")
                  << "\n";
    } else {
        std::cout << "FINAL_RESULT = NO_FEASIBLE_SOLUTION\n";
    }
    return 0;
}

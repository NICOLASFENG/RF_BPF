#define main ctc_ga_original_main
#include "../firstcheck(a2).cpp"
#undef main

#include <omp.h>

namespace {

struct ModeSearchConfig {
    int population = 1600;
    int generations = 1600;
    int dense_interval = 25;
    int dense_top = 32;
    int threads = 0;
    std::uint64_t seed = 20260503ULL;
};

struct ScoredRow {
    CapRow row{};
    ModeMetrics coarse;
    ModeMetrics dense;
    double score = INF;
    double dense_score = INF;
    bool dense_checked = false;
};

double feasibility_score(const ModeMetrics& m, const BandSpec& band) {
    if (!m.valid) {
        return INF;
    }
    std::array<double, 9> deficits = {{
        std::max(0.0, band.pass_return_loss_min_db - m.min_pass_return_loss_db),
        25.0 * std::max(0.0, m.max_pass_insertion_loss_db -
                              band.pass_insertion_loss_max_db),
        std::max(0.0, band.stop_rejection_min_db -
                        m.lower_stop_min_rejection_db),
        std::max(0.0, band.stop_rejection_min_db -
                        m.upper_stop_min_rejection_db),
        std::max(0.0, band.harmonic_rejection_min_db -
                        m.harmonic2_min_rejection_db),
        std::max(0.0, band.harmonic_rejection_min_db -
                        m.harmonic3_min_rejection_db),
        std::max(0.0, (m.zeros.tz1_hz - band.pass_lo_hz) / GHZ),
        std::max(0.0, (m.zeros.tz2_hz - band.pass_lo_hz) / GHZ),
        std::max(0.0, (band.pass_hi_hz - m.zeros.tz3_hz) / GHZ)
    }};
    double max_deficit = 0.0;
    double sum_square = 0.0;
    for (double deficit : deficits) {
        max_deficit = std::max(max_deficit, deficit);
        sum_square += deficit * deficit;
    }
    const double tz4_deficit =
        std::max(0.0, 1.40 - (m.zeros.tz4_hz / band.center_hz));
    max_deficit = std::max(max_deficit, 8.0 * tz4_deficit);
    sum_square += 64.0 * tz4_deficit * tz4_deficit;
    return 1.0e6 * max_deficit + 1.0e3 * sum_square + m.rf_objective;
}

bool better_scored_values(bool a_ok, double a_score, double a_rf,
                          bool b_ok, double b_score, double b_rf) {
    if (a_ok != b_ok) {
        return a_ok;
    }
    if (nearly_less(a_score, b_score)) {
        return true;
    }
    if (nearly_less(b_score, a_score)) {
        return false;
    }
    return a_rf < b_rf;
}

bool better_row(const ScoredRow& a, const ScoredRow& b) {
    return better_scored_values(a.coarse.constraints_ok, a.score,
                                a.coarse.rf_penalty,
                                b.coarse.constraints_ok, b.score,
                                b.coarse.rf_penalty);
}

bool better_dense_row(const ScoredRow& a, const ScoredRow& b) {
    if (a.dense_checked != b.dense_checked) {
        return a.dense_checked;
    }
    if (!a.dense_checked) {
        return better_row(a, b);
    }
    return better_scored_values(a.dense.constraints_ok, a.dense_score,
                                a.dense.rf_penalty,
                                b.dense.constraints_ok, b.dense_score,
                                b.dense.rf_penalty);
}

CapRow repair_row(CapRow row, const Bounds& bounds) {
    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        row[cell] = clamp_value(row[cell], bounds.lower[cell], bounds.upper[cell]);
        row[cell] = quantize_cap(row[cell]);
        row[cell] = clamp_value(row[cell], bounds.lower[cell], bounds.upper[cell]);
    }
    return row;
}

ScoredRow evaluate_row(CapRow row,
                       int mode,
                       const FixedValues& fixed,
                       const std::array<BandSpec, MODE_COUNT>& bands,
                       const std::array<FrequencyPlan, MODE_COUNT>& plans,
                       const Bounds& bounds) {
    ScoredRow out;
    out.row = repair_row(row, bounds);
    out.coarse = evaluate_mode(out.row, fixed, bands[mode], plans[mode]);
    out.score = feasibility_score(out.coarse, bands[mode]);
    return out;
}

void dense_check(ScoredRow& row,
                 int mode,
                 const FixedValues& fixed,
                 const std::array<BandSpec, MODE_COUNT>& bands,
    const std::array<FrequencyPlan, MODE_COUNT>& dense_plans) {
    row.dense = evaluate_mode(row.row, fixed, bands[mode], dense_plans[mode]);
    row.dense_score = feasibility_score(row.dense, bands[mode]);
    row.dense_checked = true;
}

double uniform_real(std::mt19937_64& rng, double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(rng);
}

int uniform_int(std::mt19937_64& rng, int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng);
}

void set_pair_sum(CapRow& row,
                  int a,
                  int b,
                  double sum,
                  double split,
                  const Bounds& bounds) {
    split = clamp_value(split, 0.08, 0.92);
    row[a] = sum * split;
    row[b] = sum * (1.0 - split);
    row[a] = clamp_value(row[a], bounds.lower[a], bounds.upper[a]);
    row[b] = clamp_value(row[b], bounds.lower[b], bounds.upper[b]);
}

CapRow make_zero_seed(int mode,
                      int variant,
                      const FixedValues& fixed,
                      const std::array<BandSpec, MODE_COUNT>& bands,
                      const Bounds& bounds,
                      const CapTable& baseline,
                      std::mt19937_64& rng) {
    CapRow row = baseline[mode];
    const BandSpec& band = bands[mode];
    const double lower_offsets[] = {0.06, 0.10, 0.16, 0.24, 0.34, 0.46};
    const double upper_offsets[] = {0.06, 0.10, 0.16, 0.24, 0.34, 0.50};
    const double split_choices[] = {0.50, 0.58, 0.66, 0.74, 0.42, 0.34};

    const double lower_offset = lower_offsets[variant % 6] * GHZ;
    const double upper_offset = upper_offsets[(variant / 6) % 6] * GHZ;
    const double split1 = split_choices[(variant / 36) % 6];
    const double split2 = split_choices[(variant / 216) % 6];
    const double f_low = std::max(0.20 * GHZ, band.pass_lo_hz - lower_offset);
    const double f_up = band.pass_hi_hz + upper_offset;

    set_pair_sum(row, CELL_C3, CELL_C4, cap_for_tz(f_low, fixed.l3),
                 split1, bounds);
    set_pair_sum(row, CELL_C6, CELL_C7, cap_for_tz(f_low, fixed.l4),
                 split2, bounds);
    row[CELL_C9] = cap_for_tz(f_up, fixed.l2);

    const double c2_scale = uniform_real(rng, 0.65, 1.35);
    const double c5_scale = uniform_real(rng, 0.65, 1.35);
    const double c8_scale = uniform_real(rng, 0.65, 1.35);
    row[CELL_C2] *= c2_scale;
    row[CELL_C5] *= c5_scale;
    row[CELL_C8] *= c8_scale;
    row[CELL_C1] *= uniform_real(rng, 0.70, 1.45);

    return repair_row(row, bounds);
}

std::vector<ScoredRow> make_initial_rows(
    int mode,
    const ModeSearchConfig& cfg,
    const FixedValues& fixed,
    const std::array<BandSpec, MODE_COUNT>& bands,
    const std::array<FrequencyPlan, MODE_COUNT>& coarse_plans,
    const Bounds& bounds,
    const CapTable& baseline) {
    std::vector<CapRow> rows;
    rows.reserve(static_cast<std::size_t>(cfg.population));
    std::mt19937_64 rng(cfg.seed ^ (0x9e3779b97f4a7c15ULL +
                                    static_cast<std::uint64_t>(mode)));
    std::normal_distribution<double> normal(0.0, 1.0);

    rows.push_back(repair_row(baseline[mode], bounds));

    for (int i = 0; i < cfg.population / 5; ++i) {
        CapRow row = baseline[mode];
        const double level = (i % 4 == 0) ? 0.06 : (i % 4 == 1) ? 0.12
                           : (i % 4 == 2) ? 0.22 : 0.35;
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            const double sigma = level * std::max(row[cell], 0.20 * PF);
            row[cell] += sigma * normal(rng);
        }
        rows.push_back(repair_row(row, bounds));
    }

    for (int i = 0; i < cfg.population / 2; ++i) {
        rows.push_back(make_zero_seed(mode, i, fixed, bands, bounds, baseline, rng));
    }

    while (static_cast<int>(rows.size()) < cfg.population) {
        CapRow row{};
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            if (uniform_real(rng, 0.0, 1.0) < 0.45) {
                row[cell] = uniform_real(rng, bounds.lower[cell], bounds.upper[cell]);
            } else {
                const double sigma = 0.35 *
                    std::max(baseline[mode][cell], 0.20 * PF);
                row[cell] = baseline[mode][cell] + sigma * normal(rng);
            }
        }
        rows.push_back(repair_row(row, bounds));
    }

    std::vector<ScoredRow> scored(rows.size());
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        scored[i] = evaluate_row(rows[i], mode, fixed, bands, coarse_plans, bounds);
    }
    std::sort(scored.begin(), scored.end(), better_row);
    return scored;
}

std::array<int, 3> distinct_indices(std::mt19937_64& rng, int n, int avoid) {
    std::array<int, 3> idx{};
    for (int k = 0; k < 3; ++k) {
        bool ok = false;
        while (!ok) {
            idx[k] = uniform_int(rng, 0, n - 1);
            ok = idx[k] != avoid;
            for (int j = 0; j < k; ++j) {
                ok = ok && idx[k] != idx[j];
            }
        }
    }
    return idx;
}

CapRow make_trial(const std::vector<ScoredRow>& pop,
                  int i,
                  int generation,
                  const Bounds& bounds,
                  std::uint64_t seed) {
    const int n = static_cast<int>(pop.size());
    std::mt19937_64 rng(seed ^
                        (0x9e3779b97f4a7c15ULL *
                         static_cast<std::uint64_t>(generation + 17)) ^
                        (0xbf58476d1ce4e5b9ULL *
                         static_cast<std::uint64_t>(i + 31)));
    const auto idx = distinct_indices(rng, n, i);
    const double progress = static_cast<double>(generation) /
                            static_cast<double>(std::max(1, generation + 1));
    const double f = uniform_real(rng, 0.45, 0.95);
    const double cr = uniform_real(rng, 0.60, 0.96);
    const int forced = uniform_int(rng, 0, CELL_COUNT - 1);
    CapRow trial = pop[i].row;

    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        if (uniform_real(rng, 0.0, 1.0) < cr || cell == forced) {
            const double best_pull = uniform_real(rng, 0.15, 0.45);
            double value = pop[idx[0]].row[cell] +
                f * (pop[idx[1]].row[cell] - pop[idx[2]].row[cell]) +
                best_pull * (pop[0].row[cell] - pop[i].row[cell]);
            if (uniform_real(rng, 0.0, 1.0) < 0.08) {
                const double span = bounds.upper[cell] - bounds.lower[cell];
                std::normal_distribution<double> normal(0.0, 1.0);
                const double anneal = std::max(0.15, 1.0 - progress);
                value += 0.05 * anneal * span * normal(rng);
            }
            trial[cell] = value;
        }
    }

    if (uniform_real(rng, 0.0, 1.0) < 0.02) {
        const int cell = uniform_int(rng, 0, CELL_COUNT - 1);
        trial[cell] = uniform_real(rng, bounds.lower[cell], bounds.upper[cell]);
    }

    return repair_row(trial, bounds);
}

void print_mode_summary(const char* label,
                        int mode,
                        const ScoredRow& row) {
    const ModeMetrics& m = row.dense_checked ? row.dense : row.coarse;
    std::cout << label << " " << kModeNames[mode]
              << " feasible=" << yes_no(m.constraints_ok)
              << " rf_penalty=" << std::scientific << std::setprecision(3)
              << m.rf_penalty << std::fixed
              << " RL=" << std::setprecision(2) << m.min_pass_return_loss_db
              << " IL=" << m.max_pass_insertion_loss_db
              << " LS=" << m.lower_stop_min_rejection_db
              << " US=" << m.upper_stop_min_rejection_db
              << " H2=" << m.harmonic2_min_rejection_db
              << " H3=" << m.harmonic3_min_rejection_db
              << "\n";
}

ScoredRow optimize_mode(
    int mode,
    const ModeSearchConfig& cfg,
    const FixedValues& fixed,
    const std::array<BandSpec, MODE_COUNT>& bands,
    const std::array<FrequencyPlan, MODE_COUNT>& coarse_plans,
    const std::array<FrequencyPlan, MODE_COUNT>& dense_plans,
    const Bounds& bounds,
    const CapTable& baseline) {
    std::vector<ScoredRow> pop =
        make_initial_rows(mode, cfg, fixed, bands, coarse_plans, bounds, baseline);

    ScoredRow best_dense = pop.front();
    dense_check(best_dense, mode, fixed, bands, dense_plans);
    print_mode_summary("initial", mode, best_dense);

    for (int gen = 1; gen <= cfg.generations; ++gen) {
        std::vector<ScoredRow> trials(pop.size());
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(pop.size()); ++i) {
            const CapRow row = make_trial(pop, i, gen, bounds, cfg.seed);
            trials[i] = evaluate_row(row, mode, fixed, bands, coarse_plans, bounds);
        }

        for (int i = 0; i < static_cast<int>(pop.size()); ++i) {
            if (better_row(trials[i], pop[i])) {
                pop[i] = trials[i];
            }
        }
        std::sort(pop.begin(), pop.end(), better_row);

        if (gen % cfg.dense_interval == 0 || gen == cfg.generations ||
            pop.front().coarse.constraints_ok) {
            const int top = std::min(cfg.dense_top, static_cast<int>(pop.size()));
            std::vector<ScoredRow> checked(pop.begin(), pop.begin() + top);
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < top; ++i) {
                dense_check(checked[i], mode, fixed, bands, dense_plans);
            }
            std::sort(checked.begin(), checked.end(), better_dense_row);
            if (better_dense_row(checked.front(), best_dense)) {
                best_dense = checked.front();
            }
        }

        if (gen % 50 == 0 || best_dense.dense.constraints_ok) {
            std::cout << "gen " << std::setw(5) << gen << " ";
            print_mode_summary("best-dense", mode, best_dense);
        }
        if (best_dense.dense.constraints_ok) {
            break;
        }
    }

    return best_dense;
}

void print_cap_row(int mode, const CapRow& row) {
    std::cout << kModeNames[mode] << ":";
    for (int cell = 0; cell < CELL_COUNT; ++cell) {
        std::cout << " " << kCellNames[cell] << "="
                  << std::fixed << std::setprecision(4) << pf(row[cell]);
    }
    std::cout << " pF\n";
}

int parse_mode_arg(const std::string& text) {
    const std::string t = uppercase(text);
    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        if (t == kModeNames[mode]) {
            return mode;
        }
    }
    return -1;
}

} // namespace

int main(int argc, char** argv) {
    ModeSearchConfig cfg;
    int only_mode = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--population") {
            parse_int_arg(need("--population"), cfg.population);
        } else if (arg == "--generations") {
            parse_int_arg(need("--generations"), cfg.generations);
        } else if (arg == "--seed") {
            parse_u64_arg(need("--seed"), cfg.seed);
        } else if (arg == "--dense-interval") {
            parse_int_arg(need("--dense-interval"), cfg.dense_interval);
        } else if (arg == "--dense-top") {
            parse_int_arg(need("--dense-top"), cfg.dense_top);
        } else if (arg == "--threads") {
            parse_int_arg(need("--threads"), cfg.threads);
        } else if (arg == "--mode") {
            only_mode = parse_mode_arg(need("--mode"));
            if (only_mode < 0) {
                std::cerr << "Bad --mode; use N77, N78, or N79\n";
                return 2;
            }
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return 2;
        }
    }

    cfg.population = std::max(20, cfg.population);
    cfg.generations = std::max(0, cfg.generations);
    cfg.dense_interval = std::max(1, cfg.dense_interval);
    cfg.dense_top = std::max(1, cfg.dense_top);
    if (cfg.threads > 0) {
        omp_set_num_threads(cfg.threads);
    }

    const FixedValues fixed;
    const auto bands = make_bands();
    const auto coarse_plans = make_frequency_plans(bands, false);
    const auto dense_plans = make_frequency_plans(bands, true);
    const Bounds bounds = make_bounds();
    CapTable table = make_baseline_table();

    std::cout << "Per-mode feasible search: population=" << cfg.population
              << " generations=" << cfg.generations
              << " seed=" << cfg.seed
              << " threads=" << omp_get_max_threads() << "\n";

    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        if (only_mode >= 0 && mode != only_mode) {
            continue;
        }
        ScoredRow result = optimize_mode(mode, cfg, fixed, bands, coarse_plans,
                                         dense_plans, bounds, table);
        table[mode] = result.row;
        print_cap_row(mode, result.row);
    }

    Config eval_cfg;
    eval_cfg.lambda_compression = 0.0;
    eval_cfg.lambda_hardware = 0.0;
    eval_cfg.lambda_base = 0.0;
    Candidate c = make_candidate(table, bounds, eval_cfg.mask);
    c.eval = evaluate_candidate(c.cap, fixed, bands, dense_plans, table, eval_cfg);

    std::cout << "\nCombined dense result feasible="
              << yes_no(c.eval.all_modes_feasible)
              << " rf_penalty=" << std::scientific << std::setprecision(4)
              << c.eval.rf_penalty << std::fixed << "\n";
    print_cap_table(c);
    print_metrics_table(c);
    print_failure_diagnostics(c, bands);

    return c.eval.all_modes_feasible ? 0 : 1;
}

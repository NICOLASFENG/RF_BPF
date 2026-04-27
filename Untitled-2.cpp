#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double PF = 1e-12;
constexpr double NH = 1e-9;
constexpr double EPS = 1e-30;
constexpr double HUGE_PENALTY = 1e12;

struct FixedInductors {
    double L1;
    double L2;
    double L3;
    double L4;
    double L5;
};

struct BandSpec {
    std::string name;
    double f_start_hz;
    double f_stop_hz;
    int sample_count;
};

struct Candidate {
    // Optimize only C2~C9, unit = F.
    std::array<double, 8> genes{};
    double fitness = HUGE_PENALTY;
};

struct GAConfig {
    int population_size = 260;
    int generations = 800;
    int elite_count = 14;
    int tournament_k = 4;
    double crossover_rate = 0.90;
    double mutation_rate = 0.28;
    double mutation_scale = 0.10;
    unsigned int random_seed = 2025;
};

struct BandResult {
    Candidate best;
    double avg_s11_mag = HUGE_PENALTY;
    double max_s11_mag = HUGE_PENALTY;

    // Passband metrics.
    double min_pass_return_loss_db = 1e9;   // larger is better
    double max_pass_insertion_loss_db = -1e9; // smaller is better

    // Stopband metrics.
    double min_stop_insertion_loss_db = 1e9; // larger is better
    double max_stop_s11_db = -1e9;           // smaller is better

    bool pass_constraints = false;
};

bool invalid_number(double x) {
    return !std::isfinite(x) || std::fabs(x) > 1e15;
}

double clamp_value(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

double safe_div(double numerator, double denominator) {
    if (std::fabs(denominator) < EPS) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return numerator / denominator;
}

double reflection_to_return_loss_db(double gamma_mag) {
    const double g = std::max(gamma_mag, 1e-15);
    return -20.0 * std::log10(g);
}

double s21_mag_from_lossless_network(double s11_mag) {
    const double value = std::max(0.0, 1.0 - s11_mag * s11_mag);
    return std::sqrt(value);
}

double insertion_loss_db_from_s21_mag(double s21_mag) {
    const double t = std::max(s21_mag, 1e-15);
    return -20.0 * std::log10(t);
}

std::vector<double> make_frequency_grid(const BandSpec& band) {
    std::vector<double> grid;
    grid.reserve(std::max(1, band.sample_count));

    if (band.sample_count <= 1) {
        grid.push_back(0.5 * (band.f_start_hz + band.f_stop_hz));
        return grid;
    }

    const double step = (band.f_stop_hz - band.f_start_hz) /
                        static_cast<double>(band.sample_count - 1);
    for (int i = 0; i < band.sample_count; ++i) {
        grid.push_back(band.f_start_hz + i * step);
    }
    return grid;
}

bool in_band(double f_hz, const BandSpec& band) {
    return (f_hz >= band.f_start_hz && f_hz <= band.f_stop_hz);
}

bool in_any_band(double f_hz, const std::vector<BandSpec>& bands) {
    for (const auto& band : bands) {
        if (in_band(f_hz, band)) {
            return true;
        }
    }
    return false;
}

void print_gene_assignment(const std::string& suffix, const std::array<double, 8>& genes) {
    std::cout << "\n" << suffix << " optimized C2~C9 (unit: pF)\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << "C" << (i + 2) << "_" << suffix << " = "
                  << std::fixed << std::setprecision(6)
                  << genes[i] / PF << " pF\n";
    }
}

// Keep user's original S11 formula with only numerical-safety guards.
double calculate_S11_magnitude(double C1, double C2, double C3, double C4, double C5,
                               double C6, double C7, double C8, double C9, double w,
                               double L1, double L2, double L3, double L4, double L5) {
    if (w <= 0.0) {
        return HUGE_PENALTY;
    }

    auto inv_wc = [&](double C) -> double {
        if (C <= 0.0) return std::numeric_limits<double>::quiet_NaN();
        return safe_div(1.0, w * C);
    };

    const double r3 = w * L5 + safe_div(w * C4 * L5, C7) - inv_wc(C7);
    const double r2 = w * L5 + safe_div(w * C7 * L5, C4) - inv_wc(C4);
    const double r1 = safe_div(1.0, w * w * w * C4 * C7 * L5) - inv_wc(C4) - inv_wc(C7);

    if (invalid_number(r1) || invalid_number(r2) || invalid_number(r3)) {
        return HUGE_PENALTY;
    }

    const double R2 = safe_div(r2 * w * L3, r2 + w * L3);
    const double R3 = safe_div(r3 * w * L4, r3 + w * L4);
    const double den123 = R2 + R3 + r1;

    const double R11 = safe_div(R2 * R3, den123);
    const double R22 = safe_div(r1 * R3, den123);
    const double R33 = safe_div(r1 * R2, den123);

    if (invalid_number(R2) || invalid_number(R3) || invalid_number(R11) ||
        invalid_number(R22) || invalid_number(R33)) {
        return HUGE_PENALTY;
    }

    const double r22 = R22 - inv_wc(C6);
    const double r33 = R33 - inv_wc(C3);
    const double den456 = r22 - inv_wc(C5) + r33;

    const double R111 = safe_div(r33 * r22, den456);
    const double R222 = -safe_div(r33 * inv_wc(C5), den456);
    const double R333 = -safe_div(r22 * inv_wc(C5), den456);

    if (invalid_number(r22) || invalid_number(r33) || invalid_number(R111) ||
        invalid_number(R222) || invalid_number(R333)) {
        return HUGE_PENALTY;
    }

    const double denL1 = safe_div(1.0, w) - w * L1 * C1;
    const double denL2 = safe_div(1.0, w) - w * L2 * C9;
    if (std::fabs(denL1) < EPS || std::fabs(denL2) < EPS) {
        return HUGE_PENALTY;
    }

    const double r111 = R111 + R11;
    const double r222 = R222 - inv_wc(C2) + safe_div(L1, denL1);
    const double r333 = R333 - inv_wc(C8) + safe_div(L2, denL2);

    if (invalid_number(r111) || invalid_number(r222) || invalid_number(r333)) {
        return HUGE_PENALTY;
    }

    const double den_z = 2500.0 + (r111 + r333) * (r111 + r333);
    if (std::fabs(den_z) < EPS) {
        return HUGE_PENALTY;
    }

    const double zin_real =
        (50.0 * r111 * (r111 + r333) - 50.0 * r111 * r333) / den_z;
    const double zin_imag =
        r222 + (r111 * 2500.0 + r333 * r111 * (r111 + r333)) / den_z;

    if (invalid_number(zin_real) || invalid_number(zin_imag)) {
        return HUGE_PENALTY;
    }

    const double num = (zin_real - 50.0) * (zin_real - 50.0) + zin_imag * zin_imag;
    const double den = (zin_real + 50.0) * (zin_real + 50.0) + zin_imag * zin_imag;
    if (std::fabs(den) < EPS) {
        return HUGE_PENALTY;
    }

    const double gamma = std::sqrt(num / den);
    if (invalid_number(gamma)) {
        return HUGE_PENALTY;
    }

    return gamma;
}

// Constraints used in optimization:
// 1) In passbands (N78/N79): RL(S11) >= 15 dB, IL(S21) <= 0.3 dB.
// 2) In 0~15 GHz except passbands: IL(S21) >= 10 dB, S11 <= 0.3 dB.
double evaluate_fitness(const std::array<double, 8>& genes,
                        double fixed_C1,
                        const FixedInductors& Lfix,
                        const std::vector<BandSpec>& pass_bands,
                        const BandSpec& full_scan_band) {
    double total_penalty = 0.0;
    double sum_s11_pass = 0.0;
    double max_s11_pass = 0.0;
    int pass_points = 0;

    for (const auto& pass_band : pass_bands) {
        const auto grid = make_frequency_grid(pass_band);
        for (double f : grid) {
            const double w = 2.0 * PI * f;
            const double s11_mag = calculate_S11_magnitude(
                fixed_C1,
                genes[0], genes[1], genes[2], genes[3],
                genes[4], genes[5], genes[6], genes[7],
                w,
                Lfix.L1, Lfix.L2, Lfix.L3, Lfix.L4, Lfix.L5);

            if (!std::isfinite(s11_mag) || s11_mag >= HUGE_PENALTY) {
                return HUGE_PENALTY;
            }

            const double rl_db = reflection_to_return_loss_db(s11_mag);
            const double s21_mag = s21_mag_from_lossless_network(s11_mag);
            const double il_db = insertion_loss_db_from_s21_mag(s21_mag);

            if (rl_db < 15.0) {
                total_penalty += 1.0e6 + 1.0e5 * (15.0 - rl_db);
            }
            if (il_db > 0.3) {
                total_penalty += 1.0e6 + 1.0e5 * (il_db - 0.3);
            }

            sum_s11_pass += s11_mag;
            max_s11_pass = std::max(max_s11_pass, s11_mag);
            ++pass_points;
        }
    }

    const auto full_grid = make_frequency_grid(full_scan_band);
    for (double f : full_grid) {
        if (in_any_band(f, pass_bands)) {
            continue;
        }

        const double w = 2.0 * PI * f;
        const double s11_mag = calculate_S11_magnitude(
            fixed_C1,
            genes[0], genes[1], genes[2], genes[3],
            genes[4], genes[5], genes[6], genes[7],
            w,
            Lfix.L1, Lfix.L2, Lfix.L3, Lfix.L4, Lfix.L5);

        if (!std::isfinite(s11_mag) || s11_mag >= HUGE_PENALTY) {
            return HUGE_PENALTY;
        }

        const double s11_db = reflection_to_return_loss_db(s11_mag);
        const double s21_mag = s21_mag_from_lossless_network(s11_mag);
        const double s21_db = insertion_loss_db_from_s21_mag(s21_mag);

        if (s21_db < 10.0) {
            total_penalty += 1.0e6 + 1.0e5 * (10.0 - s21_db);
        }
        if (s11_db > 0.3) {
            total_penalty += 1.0e6 + 1.0e5 * (s11_db - 0.3);
        }
    }

    if (pass_points <= 0) {
        return HUGE_PENALTY;
    }

    const double avg_s11_pass = sum_s11_pass / static_cast<double>(pass_points);
    return total_penalty + 0.65 * avg_s11_pass + 0.35 * max_s11_pass;
}

BandResult summarize_result(const Candidate& cand,
                           double fixed_C1,
                           const FixedInductors& Lfix,
                           const std::vector<BandSpec>& pass_bands,
                           const BandSpec& full_scan_band) {
    BandResult result;
    result.best = cand;
    result.avg_s11_mag = 0.0;
    result.max_s11_mag = 0.0;
    result.min_pass_return_loss_db = 1e9;
    result.max_pass_insertion_loss_db = -1e9;
    result.min_stop_insertion_loss_db = 1e9;
    result.max_stop_s11_db = -1e9;
    result.pass_constraints = true;

    int pass_points = 0;
    for (const auto& pass_band : pass_bands) {
        const auto grid = make_frequency_grid(pass_band);
        for (double f : grid) {
            const double w = 2.0 * PI * f;
            const double s11_mag = calculate_S11_magnitude(
                fixed_C1,
                cand.genes[0], cand.genes[1], cand.genes[2], cand.genes[3],
                cand.genes[4], cand.genes[5], cand.genes[6], cand.genes[7],
                w,
                Lfix.L1, Lfix.L2, Lfix.L3, Lfix.L4, Lfix.L5);

            if (!std::isfinite(s11_mag) || s11_mag >= HUGE_PENALTY) {
                result.pass_constraints = false;
                continue;
            }

            const double rl_db = reflection_to_return_loss_db(s11_mag);
            const double s21_mag = s21_mag_from_lossless_network(s11_mag);
            const double il_db = insertion_loss_db_from_s21_mag(s21_mag);

            result.avg_s11_mag += s11_mag;
            result.max_s11_mag = std::max(result.max_s11_mag, s11_mag);
            result.min_pass_return_loss_db = std::min(result.min_pass_return_loss_db, rl_db);
            result.max_pass_insertion_loss_db = std::max(result.max_pass_insertion_loss_db, il_db);

            if (rl_db < 15.0 || il_db > 0.3) {
                result.pass_constraints = false;
            }
            ++pass_points;
        }
    }

    if (pass_points > 0) {
        result.avg_s11_mag /= static_cast<double>(pass_points);
    } else {
        result.avg_s11_mag = HUGE_PENALTY;
        result.pass_constraints = false;
    }

    int stop_points = 0;
    const auto full_grid = make_frequency_grid(full_scan_band);
    for (double f : full_grid) {
        if (in_any_band(f, pass_bands)) {
            continue;
        }

        const double w = 2.0 * PI * f;
        const double s11_mag = calculate_S11_magnitude(
            fixed_C1,
            cand.genes[0], cand.genes[1], cand.genes[2], cand.genes[3],
            cand.genes[4], cand.genes[5], cand.genes[6], cand.genes[7],
            w,
            Lfix.L1, Lfix.L2, Lfix.L3, Lfix.L4, Lfix.L5);

        if (!std::isfinite(s11_mag) || s11_mag >= HUGE_PENALTY) {
            result.pass_constraints = false;
            continue;
        }

        const double s11_db = reflection_to_return_loss_db(s11_mag);
        const double s21_mag = s21_mag_from_lossless_network(s11_mag);
        const double s21_db = insertion_loss_db_from_s21_mag(s21_mag);

        result.min_stop_insertion_loss_db = std::min(result.min_stop_insertion_loss_db, s21_db);
        result.max_stop_s11_db = std::max(result.max_stop_s11_db, s11_db);

        if (s21_db < 10.0 || s11_db > 0.3) {
            result.pass_constraints = false;
        }
        ++stop_points;
    }

    if (stop_points == 0) {
        result.min_stop_insertion_loss_db = 0.0;
        result.max_stop_s11_db = 0.0;
    }

    return result;
}

Candidate make_random_candidate(std::mt19937& rng,
                                const std::array<double, 8>& lower,
                                const std::array<double, 8>& upper) {
    Candidate c;
    for (int i = 0; i < 8; ++i) {
        std::uniform_real_distribution<double> dist(lower[i], upper[i]);
        c.genes[i] = dist(rng);
    }
    return c;
}

Candidate make_seeded_candidate(const std::array<double, 8>& seed,
                                std::mt19937& rng,
                                const std::array<double, 8>& lower,
                                const std::array<double, 8>& upper,
                                double jitter_ratio) {
    Candidate c;
    std::normal_distribution<double> nd(0.0, 1.0);

    for (int i = 0; i < 8; ++i) {
        const double range = upper[i] - lower[i];
        const double jitter = nd(rng) * range * jitter_ratio;
        c.genes[i] = clamp_value(seed[i] + jitter, lower[i], upper[i]);
    }
    return c;
}

void evaluate_population(std::vector<Candidate>& pop,
                         double fixed_C1,
                         const FixedInductors& Lfix,
                         const std::vector<BandSpec>& pass_bands,
                         const BandSpec& full_scan_band) {
    for (auto& ind : pop) {
        ind.fitness = evaluate_fitness(ind.genes, fixed_C1, Lfix, pass_bands, full_scan_band);
    }
}

const Candidate& tournament_select(const std::vector<Candidate>& pop,
                                   int k,
                                   std::mt19937& rng) {
    std::uniform_int_distribution<int> uid(0, static_cast<int>(pop.size()) - 1);
    int best_idx = uid(rng);
    for (int i = 1; i < k; ++i) {
        const int idx = uid(rng);
        if (pop[idx].fitness < pop[best_idx].fitness) {
            best_idx = idx;
        }
    }
    return pop[best_idx];
}

Candidate crossover_and_mutate(const Candidate& p1,
                               const Candidate& p2,
                               const std::array<double, 8>& lower,
                               const std::array<double, 8>& upper,
                               const GAConfig& cfg,
                               int generation,
                               std::mt19937& rng) {
    Candidate child;
    std::uniform_real_distribution<double> urand(0.0, 1.0);
    std::normal_distribution<double> nrand(0.0, 1.0);

    const double anneal = 1.0 - static_cast<double>(generation) /
                                  std::max(1, cfg.generations);
    const double cur_mut_scale = std::max(0.02, cfg.mutation_scale * (0.35 + 0.65 * anneal));

    for (int i = 0; i < 8; ++i) {
        double gene = p1.genes[i];

        if (urand(rng) < cfg.crossover_rate) {
            const double alpha = 1.20 * urand(rng) - 0.10; // [-0.1, 1.1]
            gene = alpha * p1.genes[i] + (1.0 - alpha) * p2.genes[i];
        }

        if (urand(rng) < cfg.mutation_rate) {
            const double range = upper[i] - lower[i];
            gene += nrand(rng) * range * cur_mut_scale;
        }

        child.genes[i] = clamp_value(gene, lower[i], upper[i]);
    }

    return child;
}

BandResult solve_one_stage_ga(const std::string& stage_name,
                              int stage_index,
                              double fixed_C1,
                              const FixedInductors& Lfix,
                              const std::array<double, 8>& lower,
                              const std::array<double, 8>& upper,
                              const std::vector<BandSpec>& pass_bands,
                              const BandSpec& full_scan_band,
                              const std::vector<std::array<double, 8>>& seed_bank,
                              const GAConfig& cfg,
                              std::mt19937& rng) {
    std::cout << "\n==================================================\n";
    std::cout << "[Stage " << stage_index << "] optimize " << stage_name << " C2~C9\n";
    std::cout << "Passbands: ";
    for (size_t i = 0; i < pass_bands.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << pass_bands[i].name << "(" << pass_bands[i].f_start_hz / 1e9
                  << "~" << pass_bands[i].f_stop_hz / 1e9 << " GHz)";
    }
    std::cout << "\n";
    std::cout << "Full scan range: " << full_scan_band.f_start_hz / 1e9 << "~"
              << full_scan_band.f_stop_hz / 1e9 << " GHz\n";
    std::cout << "Pass constraints: RL(S11)>=15 dB, IL(S21)<=0.3 dB\n";
    std::cout << "Stop constraints: IL(S21)>=10 dB, S11<=0.3 dB\n";

    std::vector<Candidate> population;
    population.reserve(cfg.population_size);

    for (const auto& seed : seed_bank) {
        population.push_back(make_seeded_candidate(seed, rng, lower, upper, 0.00));
        population.push_back(make_seeded_candidate(seed, rng, lower, upper, 0.03));
        population.push_back(make_seeded_candidate(seed, rng, lower, upper, 0.08));
    }

    while (static_cast<int>(population.size()) < cfg.population_size) {
        population.push_back(make_random_candidate(rng, lower, upper));
    }

    evaluate_population(population, fixed_C1, Lfix, pass_bands, full_scan_band);

    Candidate global_best = *std::min_element(
        population.begin(), population.end(),
        [](const Candidate& a, const Candidate& b) { return a.fitness < b.fitness; });

    for (int gen = 0; gen < cfg.generations; ++gen) {
        std::sort(population.begin(), population.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.fitness < b.fitness;
                  });

        if (population.front().fitness < global_best.fitness) {
            global_best = population.front();
        }

        if (gen % 50 == 0 || gen == cfg.generations - 1) {
            BandResult snap = summarize_result(population.front(), fixed_C1, Lfix,
                                               pass_bands, full_scan_band);
            std::cout << "Gen " << std::setw(4) << gen
                      << " | fitness=" << std::scientific << population.front().fitness
                      << " | pass RL(min)=" << std::fixed << std::setprecision(3)
                      << snap.min_pass_return_loss_db << " dB"
                      << " | pass IL(max)=" << snap.max_pass_insertion_loss_db << " dB"
                      << " | stop IL(min)=" << snap.min_stop_insertion_loss_db << " dB"
                      << " | stop S11(max)=" << snap.max_stop_s11_db << " dB"
                      << " | status=" << (snap.pass_constraints ? "PASS" : "SEARCH")
                      << "\n";

            if (snap.pass_constraints) {
                global_best = population.front();
                std::cout << "Early stop: constraints satisfied at generation "
                          << gen << "\n";
                break;
            }
        }

        std::vector<Candidate> next_population;
        next_population.reserve(cfg.population_size);

        for (int i = 0; i < cfg.elite_count && i < static_cast<int>(population.size()); ++i) {
            next_population.push_back(population[i]);
        }

        while (static_cast<int>(next_population.size()) < cfg.population_size) {
            const Candidate& p1 = tournament_select(population, cfg.tournament_k, rng);
            const Candidate& p2 = tournament_select(population, cfg.tournament_k, rng);
            next_population.push_back(
                crossover_and_mutate(p1, p2, lower, upper, cfg, gen, rng));
        }

        population = std::move(next_population);
        evaluate_population(population, fixed_C1, Lfix, pass_bands, full_scan_band);
    }

    std::sort(population.begin(), population.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.fitness < b.fitness;
              });

    if (population.front().fitness < global_best.fitness) {
        global_best = population.front();
    }

    BandResult result = summarize_result(global_best, fixed_C1, Lfix,
                                         pass_bands, full_scan_band);

    std::cout << "-------------------- Stage Result --------------------\n";
    std::cout << "pass RL(S11) min = " << std::fixed << std::setprecision(3)
              << result.min_pass_return_loss_db << " dB\n";
    std::cout << "pass IL(S21) max = " << result.max_pass_insertion_loss_db << " dB\n";
    std::cout << "stop IL(S21) min = " << result.min_stop_insertion_loss_db << " dB\n";
    std::cout << "stop S11 max     = " << result.max_stop_s11_db << " dB\n";
    std::cout << "avg |S11| (pass) = " << result.avg_s11_mag << "\n";
    std::cout << "max |S11| (pass) = " << result.max_s11_mag << "\n";
    std::cout << "status           = " << (result.pass_constraints ? "PASS" : "NOT PASS")
              << "\n";

    print_gene_assignment(stage_name, result.best.genes);
    return result;
}

} // namespace

int main() {
    std::cout << "========== GA mode: random C2~C9 for N78+N79 ==========" << std::endl;

    // Fixed inductors from N77.
    const FixedInductors fixedL{
        0.87 * NH,
        1.50 * NH,
        0.87 * NH,
        0.77 * NH,
        0.10 * NH
    };

    // Fixed C1.
    const double fixed_C1 = 0.20 * PF;

    const std::array<double, 8> lower = {
        0.20 * PF, // C2
        1.00 * PF, // C3
        0.20 * PF, // C4
        0.20 * PF, // C5
        1.00 * PF, // C6
        0.20 * PF, // C7
        0.10 * PF, // C8
        0.10 * PF  // C9
    };

    const std::array<double, 8> upper = {
        3.50 * PF, // C2
        8.00 * PF, // C3
        3.50 * PF, // C4
        3.50 * PF, // C5
        8.00 * PF, // C6
        3.50 * PF, // C7
        2.00 * PF, // C8
        2.00 * PF  // C9
    };

    const BandSpec n78{
        "N78",
        3.30e9,
        3.80e9,
        31
    };

    const BandSpec n79{
        "N79",
        4.40e9,
        5.00e9,
        31
    };

    // Global constraint range in your latest request: 0~15 GHz.
    // 181 points => about 83.33 MHz step.
    const BandSpec full_scan_0_15{
        "FULL_0_15",
        0.01e9,
        15.0e9,
        181
    };

    const std::vector<BandSpec> n78_only{n78};
    const std::vector<BandSpec> n79_only{n79};

    const GAConfig ga_cfg{
        320,
        1200,
        14,
        4,
        0.90,
        0.28,
        0.10,
        20260318
    };

    std::mt19937 rng(ga_cfg.random_seed);

    std::cout << "\nFixed parameters:\n";
    std::cout << "C1 = " << fixed_C1 / PF << " pF\n";
    std::cout << "L1 = " << fixedL.L1 / NH << " nH\n";
    std::cout << "L2 = " << fixedL.L2 / NH << " nH\n";
    std::cout << "L3 = " << fixedL.L3 / NH << " nH\n";
    std::cout << "L4 = " << fixedL.L4 / NH << " nH\n";
    std::cout << "L5 = " << fixedL.L5 / NH << " nH\n";

    auto run_random_stage = [&](const std::string& stage_name,
                                const std::vector<BandSpec>& pass_bands) {
        const int max_random_restarts = 12;
        BandResult best_result;

        for (int attempt = 1; attempt <= max_random_restarts; ++attempt) {
            std::cout << "\n################ " << stage_name
                      << " | Random attempt " << attempt << "/"
                      << max_random_restarts << " ################\n";

            BandResult current = solve_one_stage_ga(
                stage_name,
                attempt,
                fixed_C1,
                fixedL,
                lower,
                upper,
                pass_bands,
                full_scan_0_15,
                {},
                ga_cfg,
                rng
            );

            if (current.best.fitness < best_result.best.fitness) {
                best_result = current;
            }

            if (current.pass_constraints) {
                best_result = current;
                std::cout << "Feasible solution found for "
                          << stage_name << " at random attempt "
                          << attempt << "\n";
                break;
            }
        }

        return best_result;
    };

    BandResult result_n78 = run_random_stage("N78_random", n78_only);
    BandResult result_n79 = run_random_stage("N79_random", n79_only);

    std::cout << "\n==================================================\n";
    std::cout << "Final Summary\n";
    std::cout << "N78 pass: " << (result_n78.pass_constraints ? "YES" : "NO") << "\n";
    std::cout << "N79 pass: " << (result_n79.pass_constraints ? "YES" : "NO") << "\n";

    std::cout << "\nN78 metrics:\n";
    std::cout << "pass RL(S11) min = " << std::fixed << std::setprecision(3)
              << result_n78.min_pass_return_loss_db << " dB\n";
    std::cout << "pass IL(S21) max = " << result_n78.max_pass_insertion_loss_db << " dB\n";
    std::cout << "stop IL(S21) min = " << result_n78.min_stop_insertion_loss_db << " dB\n";
    std::cout << "stop S11 max     = " << result_n78.max_stop_s11_db << " dB\n";

    std::cout << "\nN79 metrics:\n";
    std::cout << "pass RL(S11) min = " << result_n79.min_pass_return_loss_db << " dB\n";
    std::cout << "pass IL(S21) max = " << result_n79.max_pass_insertion_loss_db << " dB\n";
    std::cout << "stop IL(S21) min = " << result_n79.min_stop_insertion_loss_db << " dB\n";
    std::cout << "stop S11 max     = " << result_n79.max_stop_s11_db << " dB\n";

    std::cout << "\nN78 directly usable values:\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << "C" << (i + 2) << "_N78 = "
                  << std::fixed << std::setprecision(6)
                  << result_n78.best.genes[i] / PF << " pF\n";
    }

    std::cout << "\nN79 directly usable values:\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << "C" << (i + 2) << "_N79 = "
                  << std::fixed << std::setprecision(6)
                  << result_n79.best.genes[i] / PF << " pF\n";
    }

    return 0;
}

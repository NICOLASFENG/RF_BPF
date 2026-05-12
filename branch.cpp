#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double TINY_PF = 1.0e-12;
constexpr double HUGE_PENALTY = 1.0e12;

struct CapacitorModel {
    std::string model = "";
    int multiplier = 1;
    double density_ff_per_um2 = 2.1;
};

struct MOSSwitchParams {
    std::string model = "n33_ckt_rf";
    double channel_width_um = 5.0;
    double channel_length_nm = 350.0;
    int fingers = 40;
    int multiplier = 1;
    bool bodytie = true;
    int stack_devices_per_branch = 2;

    double total_width_per_transistor_um() const {
        return channel_width_um * static_cast<double>(fingers) *
               static_cast<double>(multiplier);
    }
};

struct BranchSwitchSet {
    MOSSwitchParams n77;
    MOSSwitchParams n78;
};

struct ResistorParams {
    std::string model = "rhppo_ckt_rf";
    double resistance_ohm = 0.0;
    double segW_um = 2.0;
    double segL_um = 20.0;
    int segments = 1;
    int multiplier = 1;
    std::string connection = "Series";
    std::string note = "";
};

struct RFModelParams {
    double ron_ref_ohm = 42.5;
    double w_ref_um = 200.0;
    double body_effect_ron_factor = 1.20;
    double coff_base_pf = 0.0;
    double coff_per_um_pf = 0.00052;
    double coff_mim_factor = 0.0;
    double con_extra_base_pf = 0.0;
    double con_extra_per_um_pf = 0.0;
    double cfix_parasitic_pf = 0.0;
    double always_parasitic_pf = 0.0;
    double mim_q = 50.0;
    double mim_esr_ohm = std::numeric_limits<double>::quiet_NaN();
    double min_branch_q = 5.0;
    double warn_branch_q = 10.0;
    double max_ron_stack_ohm = 20.0;
    double warn_ron_stack_ohm = 8.0;
    double weight_off = 0.05;
    double weight_area = 1.0e-6;
    double weight_ron = 0.001;
};

struct Targets {
    double n77_pf = 0.0;
    double n78_pf = 0.0;
    double n79_pf = 0.0;

    double max_pf() const {
        return std::max({n77_pf, n78_pf, n79_pf});
    }

    double min_pf() const {
        return std::min({n77_pf, n78_pf, n79_pf});
    }
};

struct BranchRFValues {
    double c_eff_pf = 0.0;
    double c_series_eff_pf = 0.0;
    double c_on_extra_pf = 0.0;
    double g_eff_s = 0.0;
    double q_eff = std::numeric_limits<double>::infinity();
    double x_cap_ohm = std::numeric_limits<double>::infinity();
    double q_approx = std::numeric_limits<double>::infinity();
    double esr_mim_ohm = 0.0;
    double ron_stack_ohm = 0.0;
};

struct PredictionPoint {
    double freq_ghz = 0.0;
    double ceff_n77_pf = 0.0;
    double ceff_n78_pf = 0.0;
    double ceff_n79_pf = 0.0;
};

struct ObjectiveBreakdown {
    double objective = 0.0;
    double cap_error = 0.0;
    double off_penalty = 0.0;
    double area_penalty = 0.0;
    double ron_penalty = 0.0;
    double hard_penalty = 0.0;
    double total_mim_area_um2 = 0.0;
    double off_total_pf = 0.0;
    double ron_stack_ohm = 0.0;
    double ron_stack_n77_ohm = 0.0;
    double ron_stack_n78_ohm = 0.0;
    double min_branch_q_exact = std::numeric_limits<double>::infinity();
    double q_penalty = 0.0;
};

struct CapLayout {
    std::string model = "";
    int multiplier = 1;
    double capacitance_pf = 0.0;
    double capacitance_ff = 0.0;
    double width_um = 0.0;
    double length_um = 0.0;
    double area_um2 = 0.0;
    std::string status = "OMIT";
};

struct CommandLine {
    bool check_pcell = false;
    bool pcell_self_test = false;
    bool verify_filter = false;
    bool sweep_parasitics = false;
    bool strict = false;
    std::string name;
    std::string type = "generic";
    bool type_explicit = false;
    double cn77 = 0.0;
    double cn78 = 0.0;
    double cn79 = 0.0;
    double min_branch_pf = 0.10;
    std::string parasitic_model = "conservative_scalar";
    double density_ff_per_um2 = 2.1;
    bool optimize_mos = false;
    std::vector<int> mos_mult_list = {1, 2, 4};
    std::vector<double> freqs_ghz = {3.3, 3.8, 4.2, 4.6, 5.0};
    bool json = false;
    bool csv = false;
    std::string calibration_json;
    int seed = 20260508;
    int maxiter = 220;
    int fallback_samples = 8000;
    std::string cap_model = "mim2_rf_2mask";
    double cap_width_um = std::numeric_limits<double>::quiet_NaN();
    double cap_length_um = std::numeric_limits<double>::quiet_NaN();
    double target_pf = std::numeric_limits<double>::quiet_NaN();
    double extracted_ceff_pf = std::numeric_limits<double>::quiet_NaN();
    std::string mos_model = "n33_ckt_rf";
    double mos_width_um = 5.0;
    double mos_length_nm = 350.0;
    int mos_fingers = 40;
    int mos_multiplier = 1;
    int mos_stack_devices_per_branch = 2;
    bool bodytie = true;
    double filter_l1_nh = std::numeric_limits<double>::quiet_NaN();
    double filter_l2_nh = std::numeric_limits<double>::quiet_NaN();
    double filter_l3_nh = std::numeric_limits<double>::quiet_NaN();
    double filter_l4_nh = std::numeric_limits<double>::quiet_NaN();
    double filter_c1_pf = std::numeric_limits<double>::quiet_NaN();
    double filter_c3_pf = std::numeric_limits<double>::quiet_NaN();
    double filter_c4_pf = std::numeric_limits<double>::quiet_NaN();
    double filter_c6_pf = std::numeric_limits<double>::quiet_NaN();
    double filter_c7_pf = std::numeric_limits<double>::quiet_NaN();
    double filter_c9_pf = std::numeric_limits<double>::quiet_NaN();
    std::map<std::string, std::string> scalar_options;
};

struct OptimizationResult {
    std::string cell;
    std::string cell_type;
    Targets targets;
    BranchSwitchSet switches;
    std::array<double, 3> drawn_pf{0.0, 0.0, 0.0};
    std::vector<PredictionPoint> predictions;
    std::map<std::string, double> worst_error_pct;
    std::map<std::string, CapLayout> layouts;
    ResistorParams gate_resistor;
    ResistorParams midpoint_bias_resistor;
    RFModelParams model_params;
    ObjectiveBreakdown objective;
    std::vector<std::string> warnings;
    std::string optimizer_method;
    std::string verdict = "FAILED";
    bool calibrated_from_json = false;
    double min_branch_pf = 0.0;
};

std::string lower_copy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return text;
}

std::string upper_copy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    return text;
}

bool supplied(double value) {
    return std::isfinite(value);
}

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> parts;
    std::string item;
    std::istringstream stream(text);
    while (std::getline(stream, item, delimiter)) {
        if (!item.empty()) {
            parts.push_back(item);
        }
    }
    return parts;
}

double parse_double(const std::string& text, const std::string& name) {
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0' || !std::isfinite(value)) {
        throw std::runtime_error("Invalid numeric value for " + name + ": " + text);
    }
    return value;
}

int parse_int(const std::string& text, const std::string& name) {
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        throw std::runtime_error("Invalid integer value for " + name + ": " + text);
    }
    return static_cast<int>(value);
}

bool parse_bool(const std::string& text, const std::string& name) {
    const std::string value = lower_copy(text);
    if (value == "true" || value == "1" || value == "yes" || value == "y") {
        return true;
    }
    if (value == "false" || value == "0" || value == "no" || value == "n") {
        return false;
    }
    throw std::runtime_error("Invalid boolean value for " + name + ": " + text);
}

std::vector<double> parse_freqs_ghz(const std::string& text) {
    std::vector<double> values;
    for (const auto& part : split(text, ',')) {
        values.push_back(parse_double(part, "--freqs-ghz"));
    }
    if (values.empty()) {
        throw std::runtime_error("--freqs-ghz must contain at least one frequency");
    }
    for (double value : values) {
        if (value <= 0.0) {
            throw std::runtime_error("--freqs-ghz values must be positive");
        }
    }
    return values;
}

std::vector<int> parse_mos_mult_list(const std::string& text) {
    std::vector<int> values;
    for (const auto& part : split(text, ',')) {
        values.push_back(parse_int(part, "--mos-mult-list"));
    }
    if (values.empty()) {
        throw std::runtime_error("--mos-mult-list must contain at least one multiplier");
    }
    for (int value : values) {
        if (value <= 0) {
            throw std::runtime_error("--mos-mult-list values must be positive");
        }
    }
    return values;
}

void print_help() {
    std::cout
        << "usage: branch                                             # prompt for CN77/CN78/CN79\n"
        << "       branch --name NAME --cn77 PF --cn78 PF --cn79 PF [options]\n"
        << "       branch CN77_PF CN78_PF CN79_PF\n"
        << "       branch NAME CN77_PF CN78_PF CN79_PF\n"
        << "       branch --check-pcell --cap-width-um UM --cap-length-um UM [options]\n"
        << "       branch --verify-filter --l1-nh NH --l2-nh NH --l3-nh NH --l4-nh NH \\\n"
        << "              --c1-pf PF --c3-pf PF --c4-pf PF --c6-pf PF --c7-pf PF --c9-pf PF\n\n"
        << "Deterministic N79-base pre-layout sizing for an SMIC N130 RF switched-capacitor cell.\n\n"
        << "Quick examples:\n"
        << "  branch                         # then enter: C3 5 3.32 0.85\n"
        << "  branch 5 3.32 0.85\n"
        << "  branch C1 5 3.32 0.85\n"
        << "  branch --name C1 --cn77 5 --cn78 3.32 --cn79 0.85\n\n"
        << "Sizing mode inputs:\n"
        << "  no arguments reads from stdin: [NAME] CN77 CN78 CN79\n"
        << "  --name NAME\n"
        << "  --cn77 PF\n"
        << "  --cn78 PF\n"
        << "  --cn79 PF\n\n"
        << "PCell sanity-check mode:\n"
        << "  --check-pcell                       check manually entered Cadence PCell values\n"
        << "  --pcell-self-test                   run built-in screenshot consistency example\n"
        << "  --strict                            return non-zero exit on HARD_FAIL in --check-pcell paper audit\n"
        << "  --cap-model NAME                    default: mim2_rf_2mask\n"
        << "  --cap-width-um VALUE\n"
        << "  --cap-length-um VALUE\n"
        << "  --target-pf VALUE                   optional ideal target capacitance\n"
        << "  --extracted-ceff-pf VALUE           optional PDK/PEX extracted value\n"
        << "  --mos-model NAME                    default: n33_ckt_rf\n"
        << "  --mos-width-um VALUE                default: 5\n"
        << "  --mos-length-nm VALUE               default: 350\n"
        << "  --mos-fingers INT                   default: 40\n"
        << "  --mos-multiplier INT                default: 1\n"
        << "  --mos-stack-devices INT             default: 2\n"
        << "  --bodytie true|false                default: true\n\n"
        << "Filter transmission-zero verification mode:\n"
        << "  --verify-filter                     compute fTZ1..fTZ4 from Section II formulas\n"
        << "  --l1-nh VALUE --l2-nh VALUE --l3-nh VALUE --l4-nh VALUE\n"
        << "  --c1-pf VALUE --c3-pf VALUE --c4-pf VALUE --c6-pf VALUE --c7-pf VALUE --c9-pf VALUE\n\n"
        << "Options:\n"
        << "  --min-branch-pf PF                   default: 0.10\n"
        << "  --parasitic-model NAME               conservative_scalar|no_midpoint|no_on_extra|simple\n"
        << "  --sweep-parasitics                   compare parasitic assumptions for this one cell\n"
        << "  --density-ff-per-um2 VALUE           default: 2.1\n"
        << "  --freqs-ghz 3.3,3.8,4.2,4.6,5.0      RF sample points\n"
        << "  --calibration-json FILE              optional JSON coefficient file\n"
        << "  --json                               print JSON only\n"
        << "  --csv                                print CSV only\n"
        << "  --ron-ref-ohm VALUE\n"
        << "  --w-ref-um VALUE\n"
        << "  --body-effect-ron-factor VALUE      default: 1.20 when bodytie=false\n"
        << "  --coff-base-pf VALUE\n"
        << "  --coff-per-um-pf VALUE\n"
        << "  --coff-mim-factor VALUE\n"
        << "  --con-extra-base-pf VALUE\n"
        << "  --con-extra-per-um-pf VALUE\n"
        << "  --cfix-parasitic-pf VALUE\n"
        << "  --always-parasitic-pf VALUE\n"
        << "  --mim-q VALUE                       optional MIM RF Q estimate\n"
        << "  --mim-esr-ohm VALUE                 optional MIM ESR override\n"
        << "  --kr-stack-ohm-um VALUE             pre-layout Ron_stack*W, default: 8500\n"
        << "  --coff-stack-ff-per-um VALUE        two-NMOS off stack C/W, default: 0.52\n"
        << "  --c-mid-ff-per-um VALUE             midpoint diffusion C/W, default: 1.5\n"
        << "  --bias-resistance-ohm VALUE         midpoint bias resistor, default: 500k\n"
        << "  --mim-bias-v VALUE                  MIM voltage-coefficient bias, default: 0\n"
        << "  --mim-temp-c VALUE                  MIM temperature, default: 25\n"
        << "  --mim-vc1-ppm-per-v VALUE           MIM first-order VC, default: -61.2\n"
        << "  --mim-vc2-ppm-per-v2 VALUE          MIM second-order VC, default: 26.6\n"
        << "  --mim-tc1-ppm-per-c VALUE           MIM first-order TC, default: 32.6\n"
        << "  --min-branch-q VALUE                default: 5\n"
        << "  --warn-branch-q VALUE               default: 10\n"
        << "  --max-ron-stack-ohm VALUE           default: 20\n"
        << "  --warn-ron-stack-ohm VALUE          default: 8\n"
        << "  --weight-off VALUE\n"
        << "  --weight-area VALUE\n"
        << "  --weight-ron VALUE\n"
        << "  --seed INT                           default: 20260508\n"
        << "  --maxiter INT                        optimizer generations, default: 220\n"
        << "  --fallback-samples INT               random samples, default: 8000\n";
}

bool is_flag(const std::string& option) {
    return option == "--optimize-mos" || option == "--json" ||
           option == "--csv" || option == "--check-pcell" ||
           option == "--pcell-self-test" || option == "--verify-filter" ||
           option == "--sweep-parasitics" || option == "--strict" ||
           option == "--no-bodytie" || option == "--help" || option == "-h";
}

void read_sizing_targets_from_cin(CommandLine& args) {
    std::cerr
        << "Enter target capacitances in pF as [NAME] CN77 CN78 CN79\n"
        << "Example: C3 5 3.32 0.85, or just: 5 3.32 0.85\n"
        << "> ";

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty()) {
            break;
        }
    }
    if (line.empty()) {
        throw std::runtime_error("Expected stdin input: [NAME] CN77 CN78 CN79");
    }

    std::istringstream input(line);
    std::vector<std::string> fields;
    std::string field;
    while (input >> field) {
        fields.push_back(field);
    }

    std::size_t offset = 0;
    if (fields.size() == 4) {
        if (args.name.empty()) {
            args.name = fields[0];
        } else if (args.name != fields[0]) {
            throw std::runtime_error("Cell name supplied twice with different values");
        }
        offset = 1;
    } else if (fields.size() == 3) {
        if (args.name.empty()) {
            args.name = "CELL";
        }
    } else {
        throw std::runtime_error("Expected stdin input: [NAME] CN77 CN78 CN79");
    }

    args.cn77 = parse_double(fields[offset], "CN77 stdin value");
    args.cn78 = parse_double(fields[offset + 1], "CN78 stdin value");
    args.cn79 = parse_double(fields[offset + 2], "CN79 stdin value");
    if (args.cn77 <= 0.0 || args.cn78 <= 0.0 || args.cn79 <= 0.0) {
        throw std::runtime_error("CN77, CN78, and CN79 stdin values must be positive pF values");
    }
}

CommandLine parse_command_line(int argc, char** argv) {
    if (argc == 1) {
        CommandLine args;
        read_sizing_targets_from_cin(args);
        return args;
    }

    CommandLine args;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string token = argv[i];
        if (token == "--help" || token == "-h") {
            print_help();
            std::exit(0);
        }
        if (token.rfind("--", 0) != 0) {
            positional.push_back(token);
            continue;
        }

        std::string key = token;
        std::string value;
        const std::size_t equal_pos = token.find('=');
        if (equal_pos != std::string::npos) {
            key = token.substr(0, equal_pos);
            value = token.substr(equal_pos + 1);
        } else if (!is_flag(key)) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + key);
            }
            value = argv[++i];
        }

        if (key == "--optimize-mos") {
            args.optimize_mos = true;
        } else if (key == "--json") {
            args.json = true;
        } else if (key == "--csv") {
            args.csv = true;
        } else if (key == "--check-pcell") {
            args.check_pcell = true;
        } else if (key == "--pcell-self-test") {
            args.pcell_self_test = true;
            args.check_pcell = true;
        } else if (key == "--verify-filter") {
            args.verify_filter = true;
        } else if (key == "--sweep-parasitics") {
            args.sweep_parasitics = true;
        } else if (key == "--strict") {
            args.strict = true;
        } else if (key == "--no-bodytie") {
            args.bodytie = false;
        } else if (key == "--name") {
            args.name = value;
        } else if (key == "--cn77") {
            args.cn77 = parse_double(value, key);
        } else if (key == "--cn78") {
            args.cn78 = parse_double(value, key);
        } else if (key == "--cn79") {
            args.cn79 = parse_double(value, key);
        } else if (key == "--type") {
            args.type = lower_copy(value);
            args.type_explicit = true;
            if (args.type != "generic" && args.type != "c2" && args.type != "fixed") {
                throw std::runtime_error("--type must be generic, c2, or fixed");
            }
        } else if (key == "--min-branch-pf") {
            args.min_branch_pf = parse_double(value, key);
        } else if (key == "--parasitic-model") {
            args.parasitic_model = lower_copy(value);
            if (args.parasitic_model != "conservative_scalar" &&
                args.parasitic_model != "no_midpoint" &&
                args.parasitic_model != "no_on_extra" &&
                args.parasitic_model != "simple") {
                throw std::runtime_error("--parasitic-model must be conservative_scalar, no_midpoint, no_on_extra, or simple");
            }
        } else if (key == "--density-ff-per-um2") {
            args.density_ff_per_um2 = parse_double(value, key);
        } else if (key == "--mos-mult-list") {
            args.mos_mult_list = parse_mos_mult_list(value);
        } else if (key == "--freqs-ghz") {
            args.freqs_ghz = parse_freqs_ghz(value);
        } else if (key == "--calibration-json") {
            args.calibration_json = value;
        } else if (key == "--seed") {
            args.seed = parse_int(value, key);
        } else if (key == "--maxiter") {
            args.maxiter = parse_int(value, key);
        } else if (key == "--fallback-samples") {
            args.fallback_samples = parse_int(value, key);
        } else if (key == "--cap-model") {
            args.cap_model = value;
        } else if (key == "--cap-width-um") {
            args.cap_width_um = parse_double(value, key);
        } else if (key == "--cap-length-um") {
            args.cap_length_um = parse_double(value, key);
        } else if (key == "--target-pf") {
            args.target_pf = parse_double(value, key);
        } else if (key == "--extracted-ceff-pf") {
            args.extracted_ceff_pf = parse_double(value, key);
        } else if (key == "--mos-model") {
            args.mos_model = value;
        } else if (key == "--mos-width-um") {
            args.mos_width_um = parse_double(value, key);
        } else if (key == "--mos-length-nm") {
            args.mos_length_nm = parse_double(value, key);
        } else if (key == "--mos-fingers") {
            args.mos_fingers = parse_int(value, key);
        } else if (key == "--mos-multiplier") {
            args.mos_multiplier = parse_int(value, key);
        } else if (key == "--mos-stack-devices") {
            args.mos_stack_devices_per_branch = parse_int(value, key);
        } else if (key == "--bodytie") {
            args.bodytie = parse_bool(value, key);
        } else if (key == "--l1-nh") {
            args.filter_l1_nh = parse_double(value, key);
        } else if (key == "--l2-nh") {
            args.filter_l2_nh = parse_double(value, key);
        } else if (key == "--l3-nh") {
            args.filter_l3_nh = parse_double(value, key);
        } else if (key == "--l4-nh") {
            args.filter_l4_nh = parse_double(value, key);
        } else if (key == "--c1-pf") {
            args.filter_c1_pf = parse_double(value, key);
        } else if (key == "--c3-pf") {
            args.filter_c3_pf = parse_double(value, key);
        } else if (key == "--c4-pf") {
            args.filter_c4_pf = parse_double(value, key);
        } else if (key == "--c6-pf") {
            args.filter_c6_pf = parse_double(value, key);
        } else if (key == "--c7-pf") {
            args.filter_c7_pf = parse_double(value, key);
        } else if (key == "--c9-pf") {
            args.filter_c9_pf = parse_double(value, key);
        } else if (
            key == "--ron-ref-ohm" || key == "--w-ref-um" ||
            key == "--body-effect-ron-factor" ||
            key == "--coff-base-pf" || key == "--coff-per-um-pf" ||
            key == "--coff-mim-factor" || key == "--con-extra-base-pf" ||
            key == "--con-extra-per-um-pf" || key == "--cfix-parasitic-pf" ||
            key == "--always-parasitic-pf" || key == "--weight-off" ||
            key == "--weight-area" || key == "--weight-ron" ||
            key == "--mim-q" || key == "--mim-esr-ohm" ||
            key == "--min-branch-q" || key == "--warn-branch-q" ||
            key == "--max-ron-stack-ohm" || key == "--warn-ron-stack-ohm" ||
            key == "--kr-stack-ohm-um" || key == "--coff-stack-ff-per-um" ||
            key == "--mim-r-route-fixed-ohm" ||
            key == "--mim-r-route-per-unit-ohm" ||
            key == "--mim-r-via-per-unit-ohm" ||
            key == "--mim-c-fringe-per-unit-pf" ||
            key == "--mim-c-route-per-unit-pf" ||
            key == "--mim-c-coupling-per-gap-pf" ||
            key == "--c-common-route-pf" ||
            key == "--c-on-extra-ff-per-um" ||
            key == "--c-off-extra-direct-ff-per-um" ||
            key == "--c-mid-ff-per-um" ||
            key == "--c-bias-res-par-pf" ||
            key == "--mid-factor-on" || key == "--mid-factor-off" ||
            key == "--bias-resistance-ohm" ||
            key == "--mim-bias-v" || key == "--mim-temp-c" ||
            key == "--mim-vc1-ppm-per-v" ||
            key == "--mim-vc2-ppm-per-v2" ||
            key == "--mim-tc1-ppm-per-c" ||
            key == "--eps-rel" || key == "--eps-abs-pf" ||
            key == "--mim-max-w-um" || key == "--mim-max-l-um" ||
            key == "--mim-mult-max" || key == "--mos-mult-max") {
            args.scalar_options[key.substr(2)] = value;
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }

    if (!positional.empty()) {
        if (args.check_pcell || args.verify_filter) {
            throw std::runtime_error("Positional capacitance values cannot be mixed with this mode");
        }
        if (args.cn77 != 0.0 || args.cn78 != 0.0 || args.cn79 != 0.0) {
            throw std::runtime_error("Use either positional capacitances or --cn77/--cn78/--cn79, not both");
        }
        if (positional.size() == 3) {
            if (args.name.empty()) {
                args.name = "CELL";
            }
            args.cn77 = parse_double(positional[0], "CN77 positional value");
            args.cn78 = parse_double(positional[1], "CN78 positional value");
            args.cn79 = parse_double(positional[2], "CN79 positional value");
        } else if (positional.size() == 4) {
            if (args.name.empty()) {
                args.name = positional[0];
            } else if (args.name != positional[0]) {
                throw std::runtime_error("Cell name supplied twice with different values");
            }
            args.cn77 = parse_double(positional[1], "CN77 positional value");
            args.cn78 = parse_double(positional[2], "CN78 positional value");
            args.cn79 = parse_double(positional[3], "CN79 positional value");
        } else {
            throw std::runtime_error(
                "Expected positional form: branch CN77 CN78 CN79 or branch NAME CN77 CN78 CN79");
        }
    }

    if (args.verify_filter) {
        const std::array<std::pair<std::string, double>, 10> required = {{
            {"--l1-nh", args.filter_l1_nh},
            {"--l2-nh", args.filter_l2_nh},
            {"--l3-nh", args.filter_l3_nh},
            {"--l4-nh", args.filter_l4_nh},
            {"--c1-pf", args.filter_c1_pf},
            {"--c3-pf", args.filter_c3_pf},
            {"--c4-pf", args.filter_c4_pf},
            {"--c6-pf", args.filter_c6_pf},
            {"--c7-pf", args.filter_c7_pf},
            {"--c9-pf", args.filter_c9_pf},
        }};
        for (const auto& item : required) {
            if (!supplied(item.second) || item.second <= 0.0) {
                throw std::runtime_error(item.first + " is required and must be positive in --verify-filter mode");
            }
        }
        if (args.json && args.csv) {
            throw std::runtime_error("Use only one output mode: --json or --csv");
        }
        return args;
    }

    if (!args.check_pcell &&
        args.cn77 == 0.0 && args.cn78 == 0.0 && args.cn79 == 0.0) {
        read_sizing_targets_from_cin(args);
    }

    if (!args.check_pcell) {
        if (args.name.empty()) {
            throw std::runtime_error("--name is required");
        }
        if (args.cn77 <= 0.0 || args.cn78 <= 0.0 || args.cn79 <= 0.0) {
            throw std::runtime_error("--cn77, --cn78, and --cn79 must be positive pF values");
        }
    } else if (!args.pcell_self_test) {
        if (!supplied(args.cap_width_um) || args.cap_width_um <= 0.0) {
            throw std::runtime_error("--cap-width-um is required and must be positive in --check-pcell mode");
        }
        if (!supplied(args.cap_length_um) || args.cap_length_um <= 0.0) {
            throw std::runtime_error("--cap-length-um is required and must be positive in --check-pcell mode");
        }
    }
    if (args.min_branch_pf < 0.0) {
        throw std::runtime_error("--min-branch-pf must be non-negative");
    }
    if (args.density_ff_per_um2 <= 0.0) {
        throw std::runtime_error("--density-ff-per-um2 must be positive");
    }
    if (args.maxiter <= 0) {
        throw std::runtime_error("--maxiter must be positive");
    }
    if (args.fallback_samples < 10) {
        throw std::runtime_error("--fallback-samples must be at least 10");
    }
    if (args.mos_width_um <= 0.0) {
        throw std::runtime_error("--mos-width-um must be positive");
    }
    if (args.mos_length_nm <= 0.0) {
        throw std::runtime_error("--mos-length-nm must be positive");
    }
    if (args.mos_fingers <= 0) {
        throw std::runtime_error("--mos-fingers must be positive");
    }
    if (args.mos_multiplier <= 0) {
        throw std::runtime_error("--mos-multiplier must be positive");
    }
    if (args.mos_stack_devices_per_branch <= 0) {
        throw std::runtime_error("--mos-stack-devices must be positive");
    }
    if (args.json && args.csv) {
        throw std::runtime_error("Use only one output mode: --json or --csv");
    }
    if (!args.type_explicit && upper_copy(args.name) == "C5") {
        args.type = "fixed";
    }
    return args;
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open calibration JSON: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

bool assign_param(RFModelParams& params, const std::string& key, double value) {
    if (key == "ron_ref_ohm") params.ron_ref_ohm = value;
    else if (key == "w_ref_um") params.w_ref_um = value;
    else if (key == "body_effect_ron_factor") params.body_effect_ron_factor = value;
    else if (key == "coff_base_pf") params.coff_base_pf = value;
    else if (key == "coff_per_um_pf") params.coff_per_um_pf = value;
    else if (key == "coff_mim_factor") params.coff_mim_factor = value;
    else if (key == "con_extra_base_pf") params.con_extra_base_pf = value;
    else if (key == "con_extra_per_um_pf") params.con_extra_per_um_pf = value;
    else if (key == "cfix_parasitic_pf") params.cfix_parasitic_pf = value;
    else if (key == "always_parasitic_pf") params.always_parasitic_pf = value;
    else if (key == "mim_q") params.mim_q = value;
    else if (key == "mim_esr_ohm") params.mim_esr_ohm = value;
    else if (key == "min_branch_q") params.min_branch_q = value;
    else if (key == "warn_branch_q") params.warn_branch_q = value;
    else if (key == "max_ron_stack_ohm") params.max_ron_stack_ohm = value;
    else if (key == "warn_ron_stack_ohm") params.warn_ron_stack_ohm = value;
    else if (key == "weight_off") params.weight_off = value;
    else if (key == "weight_area") params.weight_area = value;
    else if (key == "weight_ron") params.weight_ron = value;
    else return false;
    return true;
}

std::string dash_to_underscore(std::string key) {
    std::replace(key.begin(), key.end(), '-', '_');
    return key;
}

bool is_prelayout_scalar_key(const std::string& key) {
    static const std::set<std::string> keys = {
        "kr_stack_ohm_um",
        "coff_stack_ff_per_um",
        "mim_r_route_fixed_ohm",
        "mim_r_route_per_unit_ohm",
        "mim_r_via_per_unit_ohm",
        "mim_c_fringe_per_unit_pf",
        "mim_c_route_per_unit_pf",
        "mim_c_coupling_per_gap_pf",
        "c_common_route_pf",
        "c_on_extra_ff_per_um",
        "c_off_extra_direct_ff_per_um",
        "c_mid_ff_per_um",
        "c_bias_res_par_pf",
        "mid_factor_on",
        "mid_factor_off",
        "bias_resistance_ohm",
        "mim_bias_v",
        "mim_temp_c",
        "mim_vc1_ppm_per_v",
        "mim_vc2_ppm_per_v2",
        "mim_tc1_ppm_per_c",
        "eps_rel",
        "eps_abs_pf",
        "min_branch_q",
        "warn_branch_q",
        "max_ron_stack_ohm",
        "warn_ron_stack_ohm",
        "mim_max_w_um",
        "mim_max_l_um",
        "mim_mult_max",
        "mos_mult_max",
    };
    return keys.find(key) != keys.end();
}

bool load_calibration_json(const std::string& path, RFModelParams& params) {
    if (path.empty()) {
        return false;
    }
    const std::string text = read_text_file(path);
    const std::regex pair_regex(
        "\"([A-Za-z0-9_\\-]+)\"\\s*:\\s*([-+]?((\\d+\\.?\\d*)|(\\.\\d+))([eE][-+]?\\d+)?)");
    bool assigned = false;
    for (std::sregex_iterator it(text.begin(), text.end(), pair_regex), end; it != end; ++it) {
        const std::string key = dash_to_underscore((*it)[1].str());
        const double value = parse_double((*it)[2].str(), key);
        assigned = assign_param(params, key, value) || assigned;
    }
    return assigned;
}

void apply_cli_overrides(const CommandLine& args, RFModelParams& params) {
    for (const auto& item : args.scalar_options) {
        const std::string key = dash_to_underscore(item.first);
        const double value = parse_double(item.second, "--" + item.first);
        if (!assign_param(params, key, value)) {
            if (!is_prelayout_scalar_key(key)) {
                throw std::runtime_error("Internal option mapping error: " + item.first);
            }
        }
    }
}

void validate_model_params(const RFModelParams& params) {
    if (params.ron_ref_ohm <= 0.0) {
        throw std::runtime_error("ron_ref_ohm must be positive");
    }
    if (params.w_ref_um <= 0.0) {
        throw std::runtime_error("w_ref_um must be positive");
    }
    if (params.body_effect_ron_factor <= 0.0) {
        throw std::runtime_error("body_effect_ron_factor must be positive");
    }
    if (supplied(params.mim_q) && params.mim_q <= 0.0) {
        throw std::runtime_error("mim_q must be positive when supplied");
    }
    if (supplied(params.mim_esr_ohm) && params.mim_esr_ohm < 0.0) {
        throw std::runtime_error("mim_esr_ohm must be non-negative when supplied");
    }
    if (params.min_branch_q <= 0.0 || params.warn_branch_q <= 0.0) {
        throw std::runtime_error("branch Q thresholds must be positive");
    }
    if (params.warn_branch_q < params.min_branch_q) {
        throw std::runtime_error("warn_branch_q must be greater than or equal to min_branch_q");
    }
    if (params.max_ron_stack_ohm <= 0.0 || params.warn_ron_stack_ohm <= 0.0) {
        throw std::runtime_error("Ron thresholds must be positive");
    }
    if (params.max_ron_stack_ohm < params.warn_ron_stack_ohm) {
        throw std::runtime_error("max_ron_stack_ohm must be greater than or equal to warn_ron_stack_ohm");
    }
}

MOSSwitchParams make_mos_from_command_line(const CommandLine& args, int multiplier) {
    MOSSwitchParams mos;
    mos.model = args.mos_model;
    mos.channel_width_um = args.mos_width_um;
    mos.channel_length_nm = args.mos_length_nm;
    mos.fingers = args.mos_fingers;
    mos.multiplier = multiplier;
    mos.bodytie = args.bodytie;
    mos.stack_devices_per_branch = args.mos_stack_devices_per_branch;
    return mos;
}

double clamp(double value, double lo, double hi) {
    return std::min(std::max(value, lo), hi);
}

bool branch_is_active(double c_drawn_pf, double min_branch_pf) {
    return c_drawn_pf >= min_branch_pf;
}

std::array<double, 3> unpack_x_array(const std::array<double, 3>& x, const std::string& cell_type) {
    if (cell_type == "fixed") {
        return {x[0], 0.0, 0.0};
    }
    return x;
}

double fixed_cap_eff_pf(double cfix_drawn_pf, const RFModelParams& params) {
    return cfix_drawn_pf + params.cfix_parasitic_pf;
}

bool has_mim_loss_input(const RFModelParams& params) {
    return supplied(params.mim_q) || supplied(params.mim_esr_ohm);
}

double mim_esr_ohm(double c_drawn_pf, double freq_ghz, const RFModelParams& params) {
    if (supplied(params.mim_esr_ohm)) {
        return params.mim_esr_ohm;
    }
    if (!supplied(params.mim_q)) {
        return 0.0;
    }
    const double omega = 2.0 * PI * freq_ghz * 1.0e9;
    const double c_f = c_drawn_pf * 1.0e-12;
    if (omega <= 0.0 || c_f <= 0.0) {
        return 0.0;
    }
    return 1.0 / (omega * c_f * params.mim_q);
}

double ron_stack_ohm(const MOSSwitchParams& mos, const RFModelParams& params) {
    const double width_um = std::max(mos.total_width_per_transistor_um(), 1.0e-9);
    const double stack_factor =
        static_cast<double>(std::max(mos.stack_devices_per_branch, 1)) / 2.0;
    const double body_factor = mos.bodytie ? 1.0 : params.body_effect_ron_factor;
    return params.ron_ref_ohm * (params.w_ref_um / width_um) *
           stack_factor * body_factor;
}

BranchRFValues switched_branch_on_eff(
    double c_drawn_pf,
    double freq_ghz,
    const MOSSwitchParams& mos,
    const RFModelParams& params,
    double min_branch_pf) {
    if (!branch_is_active(c_drawn_pf, min_branch_pf)) {
        return {};
    }

    const double omega = 2.0 * PI * freq_ghz * 1.0e9;
    const double c_f = c_drawn_pf * 1.0e-12;
    if (c_f <= 0.0) {
        return {};
    }

    // RF equation:
    // A selected switched MIM is behind finite two-NMOS-stack resistance and
    // optional MIM ESR. Z = Ron_stack + ESR_mim + 1/(j*w*Cdrawn), Y = 1/Z.
    // The effective capacitance is extracted from susceptance: Ceff = imag(Y)/w.
    const std::complex<double> j(0.0, 1.0);
    const double ron = ron_stack_ohm(mos, params);
    const double esr = mim_esr_ohm(c_drawn_pf, freq_ghz, params);
    const double x_cap_ohm = 1.0 / (omega * c_f);
    const std::complex<double> z =
        std::complex<double>(ron + esr, 0.0) +
        1.0 / (j * omega * c_f);
    const std::complex<double> y = 1.0 / z;

    const double c_series_eff_pf = (std::imag(y) / omega) * 1.0e12;
    const double c_on_extra_pf =
        params.con_extra_base_pf +
        params.con_extra_per_um_pf * mos.total_width_per_transistor_um();
    const double q_eff =
        std::real(y) > 0.0 ? std::abs(std::imag(y) / std::real(y))
                           : std::numeric_limits<double>::infinity();
    const double q_approx =
        ron > 0.0 ? x_cap_ohm / ron : std::numeric_limits<double>::infinity();
    return {c_series_eff_pf + c_on_extra_pf, c_series_eff_pf, c_on_extra_pf,
            std::real(y), q_eff, x_cap_ohm, q_approx, esr, ron};
}

double switched_branch_off_eff_pf(
    double c_drawn_pf,
    const MOSSwitchParams& mos,
    const RFModelParams& params,
    double min_branch_pf) {
    if (!branch_is_active(c_drawn_pf, min_branch_pf)) {
        return 0.0;
    }

    const double c_sw_off_pf =
        params.coff_per_um_pf * mos.total_width_per_transistor_um();
    const double c_series_pf =
        (c_drawn_pf > 1.0e-9 && c_sw_off_pf > 1.0e-9)
            ? c_drawn_pf * c_sw_off_pf / (c_drawn_pf + c_sw_off_pf)
            : 0.0;
    return params.coff_base_pf + c_series_pf +
           params.coff_mim_factor * c_drawn_pf;
}

std::vector<PredictionPoint> predict_effective_caps(
    const std::array<double, 3>& x_in,
    const std::string& cell_type,
    const std::vector<double>& freqs_ghz,
    const BranchSwitchSet& switches,
    const RFModelParams& params,
    double min_branch_pf) {
    const auto x = unpack_x_array(x_in, cell_type);
    const double cfix_eff = fixed_cap_eff_pf(x[0], params);
    const double c77_off = switched_branch_off_eff_pf(x[1], switches.n77, params, min_branch_pf);
    const double c78_off = switched_branch_off_eff_pf(x[2], switches.n78, params, min_branch_pf);

    std::vector<PredictionPoint> points;
    for (double freq_ghz : freqs_ghz) {
        const double c77_on = switched_branch_on_eff(
                                  x[1], freq_ghz, switches.n77, params, min_branch_pf)
                                  .c_eff_pf;
        const double c78_on = switched_branch_on_eff(
                                  x[2], freq_ghz, switches.n78, params, min_branch_pf)
                                  .c_eff_pf;

        PredictionPoint point;
        point.freq_ghz = freq_ghz;
        point.ceff_n79_pf = cfix_eff + c77_off + c78_off + params.always_parasitic_pf;
        point.ceff_n77_pf = cfix_eff + c77_on + c78_off + params.always_parasitic_pf;
        point.ceff_n78_pf = cfix_eff + c77_off + c78_on + params.always_parasitic_pf;
        points.push_back(point);
    }
    return points;
}

std::vector<std::pair<double, double>> variable_bounds(
    const Targets& targets,
    const std::string& cell_type) {
    const double cfix_upper = targets.max_pf() + 0.5;
    const double cdelta_upper = targets.max_pf() - targets.min_pf() + 1.0;
    if (cell_type == "fixed") {
        return {{0.001, std::max(0.001, cfix_upper)}};
    }
    return {
        {0.001, std::max(0.001, cfix_upper)},
        {0.0, std::max(0.0, cdelta_upper)},
        {0.0, std::max(0.0, cdelta_upper)},
    };
}

std::vector<std::vector<double>> initial_guesses(
    const Targets& targets,
    const std::string& cell_type,
    const std::vector<std::pair<double, double>>& bounds,
    const BranchSwitchSet& switches,
    const RFModelParams& params,
    double min_branch_pf) {
    std::vector<std::vector<double>> guesses;
    if (cell_type == "fixed") {
        const double mean_target = (targets.n77_pf + targets.n78_pf + targets.n79_pf) / 3.0;
        guesses.push_back({
            clamp(mean_target - params.cfix_parasitic_pf - params.always_parasitic_pf,
                  bounds[0].first, bounds[0].second)});
        return guesses;
    }

    const double c77_off_min =
        switched_branch_off_eff_pf(min_branch_pf, switches.n77, params, min_branch_pf);
    const double c78_off_min =
        switched_branch_off_eff_pf(min_branch_pf, switches.n78, params, min_branch_pf);
    const double cfix_guess =
        targets.n79_pf - params.cfix_parasitic_pf -
        params.always_parasitic_pf - c77_off_min - c78_off_min;
    guesses.push_back({
        clamp(cfix_guess, bounds[0].first, bounds[0].second),
        clamp(std::max(0.0, targets.n77_pf - targets.n79_pf), bounds[1].first, bounds[1].second),
        clamp(std::max(0.0, targets.n78_pf - targets.n79_pf), bounds[2].first, bounds[2].second),
    });
    guesses.push_back({
        clamp(targets.n79_pf - params.cfix_parasitic_pf - params.always_parasitic_pf,
              bounds[0].first, bounds[0].second),
        clamp(std::max(0.0, targets.n77_pf - targets.n79_pf), bounds[1].first, bounds[1].second),
        clamp(std::max(0.0, targets.n78_pf - targets.n79_pf), bounds[2].first, bounds[2].second),
    });
    guesses.push_back({bounds[0].first, 0.0, 0.0});
    guesses.push_back({bounds[0].second, 0.0, 0.0});
    return guesses;
}

ObjectiveBreakdown objective_breakdown(
    const std::array<double, 3>& x_in,
    const std::string& cell_type,
    const Targets& targets,
    const std::vector<double>& freqs_ghz,
    const BranchSwitchSet& switches,
    const RFModelParams& params,
    double min_branch_pf,
    double density_ff_per_um2) {
    const auto x = unpack_x_array(x_in, cell_type);
    double hard_penalty = 0.0;

    if (!std::isfinite(x[0]) || !std::isfinite(x[1]) || !std::isfinite(x[2])) {
        hard_penalty += HUGE_PENALTY;
    }
    if (x[0] <= 0.0) {
        hard_penalty += HUGE_PENALTY + 1.0e9 * std::abs(x[0]);
    }
    if (x[1] < 0.0 || x[2] < 0.0) {
        hard_penalty += HUGE_PENALTY + 1.0e9 * (std::abs(std::min(x[1], 0.0)) +
                                                std::abs(std::min(x[2], 0.0)));
    }

    if (cell_type != "fixed") {
        for (double c_pf : {x[1], x[2]}) {
            if (TINY_PF < c_pf && c_pf < min_branch_pf) {
                hard_penalty += 10.0 + 100.0 * std::pow(min_branch_pf - c_pf, 2.0);
            }
            if (c_pf > 1.0) {
                hard_penalty += 1.0e-5 * std::pow(c_pf - 1.0, 2.0);
            }
            if (c_pf > 3.0) {
                hard_penalty += 1.0e-4 * std::pow(c_pf - 3.0, 2.0);
            }
        }
    }

    const auto predictions = predict_effective_caps(
        x, cell_type, freqs_ghz, switches, params, min_branch_pf);
    double sum_sq = 0.0;
    int count = 0;
    for (const auto& point : predictions) {
        const std::array<std::pair<double, double>, 3> pairs = {{
            {point.ceff_n77_pf, targets.n77_pf},
            {point.ceff_n78_pf, targets.n78_pf},
            {point.ceff_n79_pf, targets.n79_pf},
        }};
        for (const auto& pair : pairs) {
            const double denom = std::max(pair.second, 0.1);
            const double err = (pair.first - pair.second) / denom;
            sum_sq += err * err;
            ++count;
        }
    }
    const double cap_error = sum_sq / std::max(count, 1);

    const double c77_off = switched_branch_off_eff_pf(x[1], switches.n77, params, min_branch_pf);
    const double c78_off = switched_branch_off_eff_pf(x[2], switches.n78, params, min_branch_pf);
    const double off_total_pf = c77_off + c78_off;
    const double off_penalty = params.weight_off * off_total_pf * off_total_pf;

    const double active_c77 = branch_is_active(x[1], min_branch_pf) ? x[1] : 0.0;
    const double active_c78 = branch_is_active(x[2], min_branch_pf) ? x[2] : 0.0;
    double total_mim_area_um2 = std::max(x[0], 0.0) * 1000.0 / density_ff_per_um2;
    total_mim_area_um2 += active_c77 * 1000.0 / density_ff_per_um2;
    total_mim_area_um2 += active_c78 * 1000.0 / density_ff_per_um2;
    const double area_penalty = params.weight_area * total_mim_area_um2;

    const double ron_n77 = ron_stack_ohm(switches.n77, params);
    const double ron_n78 = ron_stack_ohm(switches.n78, params);
    const double ron = std::max(ron_n77, ron_n78);
    double ron_penalty = 0.0;
    double q_penalty = 0.0;
    double min_q_exact = std::numeric_limits<double>::infinity();
    if (cell_type != "fixed") {
        const std::array<std::tuple<double, const MOSSwitchParams*, double>, 2> branches = {{
            std::make_tuple(x[1], &switches.n77, c77_off),
            std::make_tuple(x[2], &switches.n78, c78_off),
        }};
        for (const auto& branch : branches) {
            const double c_pf = std::get<0>(branch);
            const MOSSwitchParams& branch_mos = *std::get<1>(branch);
            if (!branch_is_active(c_pf, min_branch_pf)) {
                continue;
            }
            const double branch_ron = ron_stack_ohm(branch_mos, params);
            ron_penalty += params.weight_ron * branch_ron;
            if (branch_ron > params.max_ron_stack_ohm) {
                q_penalty += 10.0 + 0.25 *
                    std::pow(branch_ron - params.max_ron_stack_ohm, 2.0);
            } else if (branch_ron > params.warn_ron_stack_ohm) {
                q_penalty += 0.02 *
                    std::pow(branch_ron - params.warn_ron_stack_ohm, 2.0);
            }
            for (double freq_ghz : freqs_ghz) {
                const BranchRFValues rf = switched_branch_on_eff(
                    c_pf, freq_ghz, branch_mos, params, min_branch_pf);
                min_q_exact = std::min(min_q_exact, rf.q_eff);
                if (rf.q_eff < params.min_branch_q) {
                    q_penalty += 10.0 +
                        std::pow(params.min_branch_q - rf.q_eff, 2.0);
                } else if (rf.q_eff < params.warn_branch_q) {
                    q_penalty += 0.10 *
                        std::pow(params.warn_branch_q - rf.q_eff, 2.0);
                }
            }
        }
    }
    const double objective =
        cap_error + off_penalty + area_penalty + ron_penalty + q_penalty + hard_penalty;

    return {objective, cap_error, off_penalty, area_penalty, ron_penalty,
            hard_penalty, total_mim_area_um2, off_total_pf, ron,
            ron_n77, ron_n78, min_q_exact, q_penalty};
}

using ObjectiveFn = std::function<double(const std::vector<double>&)>;

std::array<double, 3> vec_to_x3(const std::vector<double>& v, const std::string& cell_type) {
    if (cell_type == "fixed") {
        return {v.at(0), 0.0, 0.0};
    }
    return {v.at(0), v.at(1), v.at(2)};
}

std::vector<double> coordinate_polish(
    std::vector<double> best,
    double& best_score,
    const ObjectiveFn& objective,
    const std::vector<std::pair<double, double>>& bounds) {
    std::vector<double> steps;
    for (const auto& bound : bounds) {
        steps.push_back((bound.second - bound.first) / 5.0);
    }

    for (int iter = 0; iter < 180; ++iter) {
        bool improved = false;
        for (std::size_t dim = 0; dim < best.size(); ++dim) {
            if (steps[dim] <= 1.0e-10) {
                continue;
            }
            for (double sign : {-1.0, 1.0}) {
                std::vector<double> trial = best;
                trial[dim] = clamp(trial[dim] + sign * steps[dim],
                                   bounds[dim].first, bounds[dim].second);
                const double score = objective(trial);
                if (score < best_score) {
                    best = trial;
                    best_score = score;
                    improved = true;
                }
            }
        }
        if (!improved) {
            for (double& step : steps) {
                step *= 0.55;
            }
        }
    }
    return best;
}

std::pair<std::vector<double>, double> optimize_continuous(
    const ObjectiveFn& objective,
    const std::vector<std::pair<double, double>>& bounds,
    const std::vector<std::vector<double>>& guesses,
    std::uint32_t seed,
    int generations,
    int fallback_samples,
    double min_branch_pf) {
    const int dims = static_cast<int>(bounds.size());
    const int pop_size = std::max(36, 18 * dims);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    std::vector<std::vector<double>> population(pop_size, std::vector<double>(dims));
    for (int i = 0; i < pop_size; ++i) {
        for (int d = 0; d < dims; ++d) {
            const auto& b = bounds[d];
            population[i][d] = b.first + uniform01(rng) * (b.second - b.first);
            if (d > 0 && uniform01(rng) < 0.16) {
                population[i][d] = 0.0;
            } else if (d > 0 && uniform01(rng) < 0.08) {
                population[i][d] = clamp(min_branch_pf, b.first, b.second);
            }
        }
    }
    for (std::size_t i = 0; i < guesses.size() && i < population.size(); ++i) {
        population[i] = guesses[i];
    }

    std::vector<double> scores(pop_size);
    for (int i = 0; i < pop_size; ++i) {
        scores[i] = objective(population[i]);
    }

    const double F = 0.72;
    const double CR = 0.88;
    std::uniform_int_distribution<int> pick(0, pop_size - 1);
    for (int gen = 0; gen < generations; ++gen) {
        for (int i = 0; i < pop_size; ++i) {
            int a = pick(rng);
            int b = pick(rng);
            int c = pick(rng);
            while (a == i) a = pick(rng);
            while (b == i || b == a) b = pick(rng);
            while (c == i || c == a || c == b) c = pick(rng);

            std::vector<double> trial = population[i];
            std::uniform_int_distribution<int> dim_pick(0, dims - 1);
            const int force_dim = dim_pick(rng);
            for (int d = 0; d < dims; ++d) {
                if (uniform01(rng) < CR || d == force_dim) {
                    double value = population[a][d] + F * (population[b][d] - population[c][d]);
                    if (value < bounds[d].first || value > bounds[d].second) {
                        value = bounds[d].first + uniform01(rng) *
                                  (bounds[d].second - bounds[d].first);
                    }
                    trial[d] = clamp(value, bounds[d].first, bounds[d].second);
                }
            }

            if (dims > 1 && uniform01(rng) < 0.04) {
                const int branch_dim = 1 + static_cast<int>(uniform01(rng) * (dims - 1));
                trial[branch_dim] = 0.0;
            }

            const double trial_score = objective(trial);
            if (trial_score < scores[i]) {
                population[i] = trial;
                scores[i] = trial_score;
            }
        }
    }

    int best_index = static_cast<int>(
        std::min_element(scores.begin(), scores.end()) - scores.begin());
    std::vector<double> best = population[best_index];
    double best_score = scores[best_index];

    for (int s = 0; s < fallback_samples; ++s) {
        std::vector<double> trial(dims);
        for (int d = 0; d < dims; ++d) {
            trial[d] = bounds[d].first + uniform01(rng) *
                       (bounds[d].second - bounds[d].first);
            if (d > 0 && uniform01(rng) < 0.20) {
                trial[d] = 0.0;
            } else if (d > 0 && uniform01(rng) < 0.10) {
                trial[d] = clamp(min_branch_pf, bounds[d].first, bounds[d].second);
            }
        }
        const double score = objective(trial);
        if (score < best_score) {
            best = trial;
            best_score = score;
        }
    }

    for (const auto& guess : guesses) {
        const double score = objective(guess);
        if (score < best_score) {
            best = guess;
            best_score = score;
        }
    }

    best = coordinate_polish(best, best_score, objective, bounds);
    return {best, best_score};
}

std::vector<double> snap_tiny_branches(
    std::vector<double> x,
    const std::string& cell_type,
    const std::vector<std::pair<double, double>>& bounds,
    const ObjectiveFn& objective,
    double min_branch_pf) {
    if (cell_type == "fixed") {
        return x;
    }
    double best_score = objective(x);
    for (int idx : {1, 2}) {
        if (TINY_PF < x[idx] && x[idx] < min_branch_pf) {
            for (double replacement : {0.0, clamp(min_branch_pf, bounds[idx].first, bounds[idx].second)}) {
                std::vector<double> trial = x;
                trial[idx] = replacement;
                const double score = objective(trial);
                if (score <= best_score) {
                    x = trial;
                    best_score = score;
                }
            }
        }
    }
    return x;
}

struct CandidateSolution {
    std::vector<double> x;
    BranchSwitchSet switches;
    ObjectiveBreakdown breakdown;
    std::string method;
};

CandidateSolution optimize_for_switches(
    const std::string& cell_type,
    const Targets& targets,
    const std::vector<double>& freqs_ghz,
    const BranchSwitchSet& switches,
    const RFModelParams& params,
    double min_branch_pf,
    double density_ff_per_um2,
    std::uint32_t seed,
    int maxiter,
    int fallback_samples) {
    const auto bounds = variable_bounds(targets, cell_type);
    const auto guesses = initial_guesses(targets, cell_type, bounds, switches, params, min_branch_pf);
    const ObjectiveFn objective = [&](const std::vector<double>& v) {
        return objective_breakdown(vec_to_x3(v, cell_type), cell_type, targets,
                                   freqs_ghz, switches, params, min_branch_pf,
                                   density_ff_per_um2)
            .objective;
    };
    auto opt = optimize_continuous(objective, bounds, guesses, seed, maxiter,
                                   fallback_samples, min_branch_pf);
    std::vector<double> x = snap_tiny_branches(opt.first, cell_type, bounds,
                                               objective, min_branch_pf);
    ObjectiveBreakdown breakdown = objective_breakdown(
        vec_to_x3(x, cell_type), cell_type, targets, freqs_ghz, switches, params,
        min_branch_pf, density_ff_per_um2);
    return {x, switches, breakdown,
            "cxx_differential_evolution+random_search+coordinate_polish+branch_specific_mos"};
}

[[maybe_unused]] CandidateSolution choose_best_solution(
    const std::string& cell_type,
    const Targets& targets,
    const std::vector<double>& freqs_ghz,
    const MOSSwitchParams& base_mos,
    const RFModelParams& params,
    double min_branch_pf,
    double density_ff_per_um2,
    const std::vector<int>& mos_multipliers,
    int seed,
    int maxiter,
    int fallback_samples) {
    bool have_best = false;
    CandidateSolution best;
    std::size_t index = 0;
    for (int n77_mult : mos_multipliers) {
        const std::vector<int> n78_multipliers =
            cell_type == "fixed" ? std::vector<int>{base_mos.multiplier}
                                 : mos_multipliers;
        for (int n78_mult : n78_multipliers) {
            BranchSwitchSet switches;
            switches.n77 = base_mos;
            switches.n78 = base_mos;
            switches.n77.multiplier = n77_mult;
            switches.n78.multiplier = n78_mult;
            CandidateSolution candidate = optimize_for_switches(
                cell_type, targets, freqs_ghz, switches, params, min_branch_pf,
                density_ff_per_um2, static_cast<std::uint32_t>(seed + 1009 * index),
                maxiter, fallback_samples);
            if (!have_best || candidate.breakdown.objective < best.breakdown.objective) {
                best = candidate;
                have_best = true;
            }
            ++index;
        }
    }
    return best;
}

std::array<double, 3> cap_square_size_um(double c_pf, double density_ff_per_um2) {
    if (c_pf <= 0.0) {
        return {0.0, 0.0, 0.0};
    }
    const double c_ff = c_pf * 1000.0;
    const double area_um2 = c_ff / density_ff_per_um2;
    const double side_um = std::sqrt(area_um2);
    return {side_um, side_um, area_um2};
}

CapLayout build_cap_layout(
    double c_pf,
    const CapacitorModel& model,
    double min_branch_pf,
    bool is_fixed) {
    const bool active = is_fixed ? c_pf > 0.0 : branch_is_active(c_pf, min_branch_pf);
    const double layout_c_pf = active ? c_pf : 0.0;
    const auto size = cap_square_size_um(layout_c_pf, model.density_ff_per_um2);
    return {model.model, model.multiplier, layout_c_pf, layout_c_pf * 1000.0,
            size[0], size[1], size[2], active ? "ACTIVE" : "OMIT"};
}

std::map<std::string, double> worst_errors_pct(
    const std::vector<PredictionPoint>& predictions,
    const Targets& targets) {
    std::map<std::string, double> worst{{"N77", 0.0}, {"N78", 0.0}, {"N79", 0.0}};
    for (const auto& point : predictions) {
        const std::array<std::pair<std::string, std::pair<double, double>>, 3> pairs = {{
            {"N77", {point.ceff_n77_pf, targets.n77_pf}},
            {"N78", {point.ceff_n78_pf, targets.n78_pf}},
            {"N79", {point.ceff_n79_pf, targets.n79_pf}},
        }};
        for (const auto& item : pairs) {
            const double denom = std::max(std::abs(item.second.second), 0.1);
            const double err = std::abs(item.second.first - item.second.second) / denom * 100.0;
            worst[item.first] = std::max(worst[item.first], err);
        }
    }
    return worst;
}

void push_unique(std::vector<std::string>& warnings, const std::string& warning) {
    if (std::find(warnings.begin(), warnings.end(), warning) == warnings.end()) {
        warnings.push_back(warning);
    }
}

struct BranchHealth {
    std::string name;
    bool active = false;
    double c_drawn_pf = 0.0;
    double total_width_um = 0.0;
    double ron_stack_ohm = 0.0;
    double coff_branch_pf = 0.0;
    double min_q_exact = std::numeric_limits<double>::infinity();
    double min_q_approx = std::numeric_limits<double>::infinity();
    double min_ceff_on_pf = std::numeric_limits<double>::infinity();
};

BranchHealth summarize_branch_health(
    const std::string& name,
    double c_drawn_pf,
    const std::vector<double>& freqs_ghz,
    const MOSSwitchParams& mos,
    const RFModelParams& params,
    double min_branch_pf) {
    BranchHealth health;
    health.name = name;
    health.active = branch_is_active(c_drawn_pf, min_branch_pf);
    health.c_drawn_pf = c_drawn_pf;
    health.total_width_um = mos.total_width_per_transistor_um();
    health.ron_stack_ohm = ron_stack_ohm(mos, params);
    health.coff_branch_pf = switched_branch_off_eff_pf(c_drawn_pf, mos, params, min_branch_pf);
    if (!health.active) {
        return health;
    }
    for (double freq_ghz : freqs_ghz) {
        const BranchRFValues rf = switched_branch_on_eff(
            c_drawn_pf, freq_ghz, mos, params, min_branch_pf);
        health.min_q_exact = std::min(health.min_q_exact, rf.q_eff);
        health.min_q_approx = std::min(health.min_q_approx, rf.q_approx);
        health.min_ceff_on_pf = std::min(health.min_ceff_on_pf, rf.c_series_eff_pf);
    }
    return health;
}

std::string determine_verdict(
    const std::string& cell_type,
    const Targets& targets,
    const std::array<double, 3>& drawn_pf,
    const std::map<std::string, double>& worst,
    const ObjectiveBreakdown& objective,
    const BranchSwitchSet& switches,
    const std::vector<double>& freqs_ghz,
    const RFModelParams& params,
    double min_branch_pf,
    bool calibrated_from_json) {
    const bool cap_error_bad =
        worst.at("N77") > 2.0 || worst.at("N78") > 2.0 || worst.at("N79") > 2.0;
    if (drawn_pf[0] <= 0.0 || objective.hard_penalty >= HUGE_PENALTY * 0.5 ||
        !std::isfinite(objective.objective) || cap_error_bad) {
        return "FAILED";
    }

    if (cell_type == "fixed") {
        return (calibrated_from_json && has_mim_loss_input(params))
                   ? "FIRST_PASS_OK"
                   : "FIRST_PASS_ONLY_NEEDS_PDK_VERIFICATION";
    }

    const std::array<BranchHealth, 2> branches = {{
        summarize_branch_health("N77", drawn_pf[1], freqs_ghz, switches.n77, params, min_branch_pf),
        summarize_branch_health("N78", drawn_pf[2], freqs_ghz, switches.n78, params, min_branch_pf),
    }};

    bool low_q_fail = false;
    bool low_q_warn = false;
    bool ron_fail = false;
    bool ron_warn = false;
    bool large_branch_poor_tradeoff = false;
    for (const auto& branch : branches) {
        if (!branch.active) {
            continue;
        }
        low_q_fail = low_q_fail || branch.min_q_exact < params.min_branch_q;
        low_q_warn = low_q_warn || branch.min_q_exact < params.warn_branch_q;
        ron_fail = ron_fail || branch.ron_stack_ohm > params.max_ron_stack_ohm;
        ron_warn = ron_warn || branch.ron_stack_ohm > params.warn_ron_stack_ohm;
        if (branch.c_drawn_pf > 1.0 &&
            (branch.min_q_exact < params.warn_branch_q ||
             branch.ron_stack_ohm > params.warn_ron_stack_ohm ||
             branch.coff_branch_pf / std::max(targets.n79_pf, 0.1) > 0.20)) {
            large_branch_poor_tradeoff = true;
        }
    }

    const double off_fraction = objective.off_total_pf / std::max(std::abs(targets.n79_pf), 0.1);
    if (large_branch_poor_tradeoff && (low_q_fail || ron_fail || off_fraction > 0.20)) {
        return "NEEDS_SEGMENTATION";
    }
    if (low_q_fail || low_q_warn || ron_fail || ron_warn || off_fraction > 0.20) {
        return "PHYSICALLY_SUSPICIOUS";
    }
    if (!calibrated_from_json || !has_mim_loss_input(params)) {
        return "FIRST_PASS_ONLY_NEEDS_PDK_VERIFICATION";
    }
    return "FIRST_PASS_OK";
}

std::vector<std::string> build_warnings(
    const std::string& cell_type,
    const Targets& targets,
    const std::array<double, 3>& drawn_pf,
    const std::map<std::string, double>& worst,
    const ObjectiveBreakdown& objective,
    const BranchSwitchSet& switches,
    const std::vector<double>& freqs_ghz,
    const RFModelParams& params,
    double min_branch_pf,
    bool calibrated_from_json,
    const std::string& auto_type_note) {
    std::vector<std::string> warnings;
    push_unique(warnings, "This is a first-pass analytical sizing result.");
    push_unique(warnings, "Final values must be corrected using RF extraction.");
    push_unique(warnings, "Extract Ceff_on, Ceff_off, Ron_eff, Coff, and branch Q from 3.3 to 5.0 GHz.");
    push_unique(warnings, "If the extracted off-state residual capacitance is large, rerun this program with calibrated parasitic parameters.");
    if (!auto_type_note.empty()) {
        push_unique(warnings, auto_type_note);
    }
    if (!calibrated_from_json) {
        push_unique(warnings, "No calibration JSON supplied; default/CLI parasitic coefficients are SMIC N130 first-order estimates and must be replaced by extracted RF PDK data for serious design.");
    }
    if (!has_mim_loss_input(params)) {
        push_unique(warnings, "MIM Q/ESR is not supplied. The capacitor PCell density is checked, but RF loss is not fully verified. Extract MIM Q/ESR from the PDK RF model.");
    } else if (supplied(params.mim_q) && supplied(params.mim_esr_ohm)) {
        push_unique(warnings, "Both mim_q and mim_esr_ohm are supplied; mim_esr_ohm is used directly in the on-state RF branch model.");
    }
    if (worst.at("N77") > 2.0 || worst.at("N78") > 2.0 || worst.at("N79") > 2.0) {
        push_unique(warnings, "Optimizer could not match all three mode targets within 2%.");
    }
    if (drawn_pf[0] <= 0.005) {
        push_unique(warnings, "C_FIX_drawn is extremely small; verify the fixed MIM is physically meaningful.");
    }
    if (cell_type != "fixed") {
        const std::array<std::pair<std::string, double>, 2> branches = {{
            {"N77", drawn_pf[1]},
            {"N78", drawn_pf[2]},
        }};
        for (const auto& branch : branches) {
            std::ostringstream min_msg;
            if (branch.second < min_branch_pf) {
                min_msg << branch.first << " switched branch is below min_branch_pf="
                        << std::setprecision(4) << min_branch_pf
                        << " pF and is marked OMIT.";
                push_unique(warnings, min_msg.str());
            }
            if (branch.second > 1.0) {
                push_unique(warnings, branch.first + " switched branch is larger than 1.0 pF. Consider segmentation.");
            }
            if (branch.second > 3.0) {
                push_unique(warnings, branch.first + " switched branch is larger than 3.0 pF. Very large switched branch. Single-branch implementation is likely physically poor; segmentation is recommended.");
            }
        }
        const std::array<BranchHealth, 2> healths = {{
            summarize_branch_health("N77", drawn_pf[1], freqs_ghz, switches.n77, params, min_branch_pf),
            summarize_branch_health("N78", drawn_pf[2], freqs_ghz, switches.n78, params, min_branch_pf),
        }};
        for (const auto& health : healths) {
            if (!health.active) {
                continue;
            }
            if (health.min_q_exact < params.warn_branch_q) {
                push_unique(warnings, "Branch Q is low. Insertion loss may be high. Verify with PDK RF S-parameter simulation.");
                std::ostringstream msg;
                msg << health.name << " minimum exact branch Q is "
                    << std::fixed << std::setprecision(3) << health.min_q_exact
                    << " across the requested RF sample points.";
                push_unique(warnings, msg.str());
            }
            if (health.min_q_exact < params.min_branch_q) {
                std::ostringstream msg;
                msg << health.name << " exact branch Q is below min_branch_q="
                    << std::fixed << std::setprecision(3) << params.min_branch_q
                    << "; capacitance matching alone is not a valid RF solution.";
                push_unique(warnings, msg.str());
            }
            if (health.ron_stack_ohm > params.max_ron_stack_ohm) {
                push_unique(warnings, "Ron_stack is too high for a pF-level RF switched-capacitor branch.");
            } else if (health.ron_stack_ohm > params.warn_ron_stack_ohm) {
                push_unique(warnings, "Ron_stack may be acceptable only after full S-parameter verification.");
            }
        }
    }

    const double cn79_ref = std::max(std::abs(targets.n79_pf), 0.1);
    const double off_fraction = objective.off_total_pf / cn79_ref;
    if (off_fraction > 0.20) {
        std::ostringstream msg;
        msg << "Disabled-branch off capacitance is " << std::fixed << std::setprecision(1)
            << (off_fraction * 100.0)
            << "% of CN79; default mode may be strongly contaminated.";
        push_unique(warnings, msg.str());
    }
    if (objective.ron_stack_ohm > params.warn_ron_stack_ohm ||
        switches.n77.total_width_per_transistor_um() < params.w_ref_um ||
        switches.n78.total_width_per_transistor_um() < params.w_ref_um) {
        std::ostringstream msg;
        msg << "Selected MOS width may be too small for low Ron: worst Ron_stack="
            << std::setprecision(4) << objective.ron_stack_ohm << " Ohm.";
        push_unique(warnings, msg.str());
    }
    if (switches.n77.total_width_per_transistor_um() > 800.0 ||
        switches.n78.total_width_per_transistor_um() > 800.0 ||
        (off_fraction > 0.30 &&
         (switches.n77.total_width_per_transistor_um() >= 800.0 ||
          switches.n78.total_width_per_transistor_um() >= 800.0))) {
        push_unique(warnings, "Selected MOS width may be too large for clean off-state isolation.");
    }
    const RFModelParams defaults;
    if (!calibrated_from_json &&
        params.coff_per_um_pf == defaults.coff_per_um_pf &&
        params.coff_base_pf == defaults.coff_base_pf &&
        params.coff_mim_factor == defaults.coff_mim_factor) {
        push_unique(warnings, "Using SMIC N130 first-order C_off estimate; replace it with extracted n33_ckt_rf off-capacitance data before final sizing.");
    }
    return warnings;
}

[[maybe_unused]] OptimizationResult make_result(
    const CommandLine& args,
    const Targets& targets,
    const CandidateSolution& candidate,
    const RFModelParams& params,
    bool calibrated_from_json,
    const std::string& auto_type_note) {
    OptimizationResult result;
    result.cell = args.name;
    result.cell_type = args.type;
    result.targets = targets;
    result.switches = candidate.switches;
    result.drawn_pf = unpack_x_array(vec_to_x3(candidate.x, args.type), args.type);
    result.predictions = predict_effective_caps(
        result.drawn_pf, args.type, args.freqs_ghz, result.switches, params,
        args.min_branch_pf);
    result.worst_error_pct = worst_errors_pct(result.predictions, targets);

    const CapacitorModel fixed_model{"mim2_rf_2mask", 1, args.density_ff_per_um2};
    const CapacitorModel delta_model{"mim2_rf_2mask", 1, args.density_ff_per_um2};
    result.layouts["FIX"] = build_cap_layout(
        result.drawn_pf[0], fixed_model, args.min_branch_pf, true);
    result.layouts["N77_DELTA"] = build_cap_layout(
        result.drawn_pf[1], delta_model, args.min_branch_pf, false);
    result.layouts["N78_DELTA"] = build_cap_layout(
        result.drawn_pf[2], delta_model, args.min_branch_pf, false);

    result.gate_resistor = {"rhppo_ckt_rf", 20133.0, 2.0, 20.0, 2, 1, "Series", ""};
    result.midpoint_bias_resistor = {
        "rhppo_ckt_rf", 989624.0, 2.0, 100.0, 20, 1, "Series",
        "Connect between the two-NMOS midpoint and ground/DC bias."};
    result.model_params = params;
    result.objective = objective_breakdown(
        result.drawn_pf, args.type, targets, args.freqs_ghz, result.switches, params,
        args.min_branch_pf, args.density_ff_per_um2);
    result.optimizer_method = candidate.method;
    result.calibrated_from_json = calibrated_from_json;
    result.min_branch_pf = args.min_branch_pf;
    result.verdict = determine_verdict(
        args.type, targets, result.drawn_pf, result.worst_error_pct,
        result.objective, result.switches, args.freqs_ghz, params,
        args.min_branch_pf, calibrated_from_json);
    result.warnings = build_warnings(
        args.type, targets, result.drawn_pf, result.worst_error_pct,
        result.objective, result.switches, args.freqs_ghz, params, args.min_branch_pf,
        calibrated_from_json, auto_type_note);
    return result;
}

std::string fmt_maybe_int(double value) {
    std::ostringstream out;
    if (std::abs(value - std::round(value)) < 1.0e-12) {
        out << static_cast<long long>(std::llround(value));
    } else {
        out << value;
    }
    return out.str();
}

void print_cap_layout(const std::string& name, const CapLayout& layout) {
    std::cout << "  " << name << ":\n";
    std::cout << "    model       = " << layout.model << "\n";
    std::cout << "    multiplier  = " << layout.multiplier << "\n";
    std::cout << "    capacitance = " << std::fixed << std::setprecision(6)
              << layout.capacitance_pf << " pF = " << std::setprecision(3)
              << layout.capacitance_ff << " fF\n";
    std::cout << "    width       = " << std::setprecision(4)
              << layout.width_um << " um\n";
    std::cout << "    length      = " << std::setprecision(4)
              << layout.length_um << " um\n";
    std::cout << "    status      = " << layout.status << "\n";
}

void print_resistor(const ResistorParams& res) {
    std::cout << "  model       = " << res.model << "\n";
    if (res.resistance_ohm >= 1000.0) {
        std::cout << "  resistance  = " << std::fixed << std::setprecision(3)
                  << (res.resistance_ohm / 1000.0) << " kOhm\n";
    } else {
        std::cout << "  resistance  = " << std::fixed << std::setprecision(3)
                  << res.resistance_ohm << " Ohm\n";
    }
    std::cout << "  segW        = " << fmt_maybe_int(res.segW_um) << " um\n";
    std::cout << "  segL        = " << fmt_maybe_int(res.segL_um) << " um\n";
    std::cout << "  segments    = " << res.segments << "\n";
    std::cout << "  multiplier  = " << res.multiplier << "\n";
    std::cout << "  connection  = " << res.connection << "\n";
}

std::vector<double> freqs_from_predictions(const std::vector<PredictionPoint>& predictions) {
    std::vector<double> freqs;
    for (const auto& point : predictions) {
        freqs.push_back(point.freq_ghz);
    }
    return freqs;
}

void print_mos_fields(const MOSSwitchParams& mos, const std::string& indent) {
    std::cout << indent << "model                         = " << mos.model << "\n";
    std::cout << indent << "channel_width_um              = "
              << fmt_maybe_int(mos.channel_width_um) << "\n";
    std::cout << indent << "channel_length_nm             = "
              << fmt_maybe_int(mos.channel_length_nm) << "\n";
    std::cout << indent << "fingers                       = " << mos.fingers << "\n";
    std::cout << indent << "multiplier                    = " << mos.multiplier << "\n";
    std::cout << indent << "total_width_per_transistor_um = "
              << fmt_maybe_int(mos.total_width_per_transistor_um()) << "\n";
    std::cout << indent << "bodytie                       = "
              << (mos.bodytie ? "true" : "false") << "\n";
    std::cout << indent << "devices_per_branch            = "
              << mos.stack_devices_per_branch << " series NMOS\n";
}

void print_branch_rf_details(
    const std::string& name,
    double c_drawn_pf,
    const MOSSwitchParams& mos,
    const std::vector<double>& freqs_ghz,
    const RFModelParams& params,
    double min_branch_pf) {
    const bool active = branch_is_active(c_drawn_pf, min_branch_pf);
    std::cout << "  " << name << " branch:\n";
    std::cout << "    status                       = "
              << (active ? "ACTIVE" : "OMIT") << "\n";
    std::cout << "    C_drawn_pF                   = "
              << std::fixed << std::setprecision(6) << c_drawn_pf << "\n";
    if (!active) {
        std::cout << "    MOS PCell                    = not instantiated by default for omitted branch\n";
        std::cout << "    RF Q                         = not calculated for omitted branch\n";
        return;
    }

    std::cout << "    MOS PCell parameters:\n";
    print_mos_fields(mos, "      ");
    std::cout << "    Ron_stack_ohm                = "
              << std::fixed << std::setprecision(4)
              << ron_stack_ohm(mos, params) << "\n";
    std::cout << "    Coff_branch_pF               = "
              << std::setprecision(6)
              << switched_branch_off_eff_pf(c_drawn_pf, mos, params, min_branch_pf)
              << "\n";
    std::cout << "    RF on-state table:\n";
    std::cout << "      Freq_GHz   Xc_ohm   ESR_mim_ohm   Q_approx   Q_exact   Ceff_on_pF   G_on_S\n";
    for (double freq_ghz : freqs_ghz) {
        const BranchRFValues rf = switched_branch_on_eff(
            c_drawn_pf, freq_ghz, mos, params, min_branch_pf);
        std::cout << "      " << std::setw(7) << std::setprecision(2) << freq_ghz
                  << "   " << std::setw(7) << std::setprecision(3) << rf.x_cap_ohm
                  << "   " << std::setw(11) << std::setprecision(5) << rf.esr_mim_ohm
                  << "   " << std::setw(8) << std::setprecision(3) << rf.q_approx
                  << "   " << std::setw(7) << std::setprecision(3) << rf.q_eff
                  << "   " << std::setw(10) << std::setprecision(6) << rf.c_series_eff_pf
                  << "   " << std::setw(8) << std::setprecision(6) << rf.g_eff_s
                  << "\n";
    }
}

[[maybe_unused]] void print_report(const OptimizationResult& result) {
    std::cout << "============================================================\n";
    std::cout << "PARASITIC-AWARE RF SWITCHED-CAP SIZING REPORT\n";
    std::cout << "Cell: " << result.cell << "\n";
    std::cout << "Type: " << result.cell_type << "\n";
    std::cout << "Default/off mode: N79\n";
    std::cout << "============================================================\n\n";

    std::cout << "Input targets:\n";
    std::cout << "  CN77 = " << std::fixed << std::setprecision(4)
              << result.targets.n77_pf << " pF\n";
    std::cout << "  CN78 = " << std::fixed << std::setprecision(4)
              << result.targets.n78_pf << " pF\n";
    std::cout << "  CN79 = " << std::fixed << std::setprecision(4)
              << result.targets.n79_pf << " pF\n\n";

    std::cout << "Selected branch MOS switches:\n";
    std::cout << "  N77 status = "
              << (branch_is_active(result.drawn_pf[1], result.min_branch_pf) ? "ACTIVE" : "OMIT")
              << "\n";
    if (branch_is_active(result.drawn_pf[1], result.min_branch_pf)) {
        print_mos_fields(result.switches.n77, "    ");
    } else {
        std::cout << "    MOS PCell = not instantiated by default for omitted branch\n";
    }
    std::cout << "  N78 status = "
              << (branch_is_active(result.drawn_pf[2], result.min_branch_pf) ? "ACTIVE" : "OMIT")
              << "\n";
    if (branch_is_active(result.drawn_pf[2], result.min_branch_pf)) {
        print_mos_fields(result.switches.n78, "    ");
    } else {
        std::cout << "    MOS PCell = not instantiated by default for omitted branch\n";
    }
    std::cout << "\n";

    std::cout << "Optimized drawn capacitances:\n";
    std::cout << "  " << result.cell << "_FIX_drawn        = " << std::fixed
              << std::setprecision(6) << result.drawn_pf[0] << " pF\n";
    std::cout << "  " << result.cell << "_N77_DELTA_drawn  = "
              << result.drawn_pf[1] << " pF\n";
    std::cout << "  " << result.cell << "_N78_DELTA_drawn  = "
              << result.drawn_pf[2] << " pF\n\n";

    std::cout << "Value definitions:\n";
    std::cout << "  PCell nominal drawn capacitance is the MIM geometry value above.\n";
    std::cout << "  On-state effective RF capacitance is Ceff_on = imag(Y_on)/omega and is frequency dependent.\n";
    std::cout << "  Off-state residual capacitance is Coff_branch from the MOS/MIM parasitic model.\n";
    std::cout << "  Target ideal capacitances are CN77/CN78/CN79 from the input.\n";
    std::cout << "  Extracted PDK/PEX values are not available in this run unless supplied through calibration/simulation data.\n\n";

    std::cout << "Predicted effective capacitance by mode:\n";
    std::cout << "  Frequency    Ceff_N77    Ceff_N78    Ceff_N79\n";
    for (const auto& point : result.predictions) {
        std::cout << "  " << std::setw(4) << std::setprecision(1) << point.freq_ghz
                  << " GHz    " << std::setw(8) << std::setprecision(4)
                  << point.ceff_n77_pf << " pF  " << std::setw(8)
                  << point.ceff_n78_pf << " pF  " << std::setw(8)
                  << point.ceff_n79_pf << " pF\n";
    }
    std::cout << "\nMode errors:\n";
    std::cout << "  N77 worst error = " << std::setprecision(4)
              << result.worst_error_pct.at("N77") << " %\n";
    std::cout << "  N78 worst error = " << result.worst_error_pct.at("N78") << " %\n";
    std::cout << "  N79 worst error = " << result.worst_error_pct.at("N79") << " %\n\n";

    std::cout << "Objective summary:\n";
    std::cout << "  verdict               = " << result.verdict << "\n";
    std::cout << "  optimizer_method      = " << result.optimizer_method << "\n";
    std::cout << "  objective             = " << std::setprecision(8)
              << result.objective.objective << "\n";
    std::cout << "  cap_error             = " << result.objective.cap_error << "\n";
    std::cout << "  off_penalty           = " << result.objective.off_penalty << "\n";
    std::cout << "  area_penalty          = " << result.objective.area_penalty << "\n";
    std::cout << "  ron_penalty           = " << result.objective.ron_penalty << "\n";
    std::cout << "  q_penalty             = " << result.objective.q_penalty << "\n";
    std::cout << "  worst_Ron_stack       = " << std::fixed << std::setprecision(4)
              << result.objective.ron_stack_ohm << " Ohm\n";
    std::cout << "  N77_Ron_stack         = " << std::setprecision(4)
              << result.objective.ron_stack_n77_ohm << " Ohm\n";
    std::cout << "  N78_Ron_stack         = " << std::setprecision(4)
              << result.objective.ron_stack_n78_ohm << " Ohm\n";
    std::cout << "  min_branch_Q_exact    = " << std::setprecision(4)
              << result.objective.min_branch_q_exact << "\n";
    std::cout << "  off_total             = " << std::setprecision(6)
              << result.objective.off_total_pf << " pF\n\n";

    std::cout << "Capacitor layout suggestions:\n";
    print_cap_layout(result.cell + "_FIX", result.layouts.at("FIX"));
    std::cout << "\n";
    print_cap_layout(result.cell + "_N77_DELTA", result.layouts.at("N77_DELTA"));
    std::cout << "\n";
    print_cap_layout(result.cell + "_N78_DELTA", result.layouts.at("N78_DELTA"));
    std::cout << "\n";

    std::cout << "Branch-specific RF switch health:\n";
    const std::vector<double> branch_freqs = freqs_from_predictions(result.predictions);
    print_branch_rf_details("N77", result.drawn_pf[1], result.switches.n77,
                            branch_freqs, result.model_params, result.min_branch_pf);
    std::cout << "\n";
    print_branch_rf_details("N78", result.drawn_pf[2], result.switches.n78,
                            branch_freqs, result.model_params, result.min_branch_pf);
    std::cout << "\n";

    std::cout << "Gate resistor parameters:\n";
    print_resistor(result.gate_resistor);
    std::cout << "\n";

    std::cout << "Midpoint bias resistor parameters:\n";
    print_resistor(result.midpoint_bias_resistor);
    std::cout << "  note        = " << result.midpoint_bias_resistor.note << "\n\n";

    std::cout << "Model coefficients:\n";
    std::cout << std::defaultfloat << std::setprecision(8);
    std::cout << "  ron_ref_ohm              = " << result.model_params.ron_ref_ohm << "\n";
    std::cout << "  w_ref_um                 = " << result.model_params.w_ref_um << "\n";
    std::cout << "  body_effect_ron_factor   = " << result.model_params.body_effect_ron_factor << "\n";
    std::cout << "  coff_base_pf             = " << result.model_params.coff_base_pf << "\n";
    std::cout << "  coff_per_um_pf           = " << result.model_params.coff_per_um_pf << "\n";
    std::cout << "  coff_mim_factor          = " << result.model_params.coff_mim_factor << "\n";
    std::cout << "  con_extra_base_pf        = " << result.model_params.con_extra_base_pf << "\n";
    std::cout << "  con_extra_per_um_pf      = " << result.model_params.con_extra_per_um_pf << "\n";
    std::cout << "  cfix_parasitic_pf        = " << result.model_params.cfix_parasitic_pf << "\n";
    std::cout << "  always_parasitic_pf      = " << result.model_params.always_parasitic_pf << "\n";
    if (supplied(result.model_params.mim_q)) {
        std::cout << "  mim_q                    = " << result.model_params.mim_q << "\n";
    } else {
        std::cout << "  mim_q                    = not supplied\n";
    }
    if (supplied(result.model_params.mim_esr_ohm)) {
        std::cout << "  mim_esr_ohm              = " << result.model_params.mim_esr_ohm << "\n";
    } else {
        std::cout << "  mim_esr_ohm              = not supplied\n";
    }
    std::cout << "  min_branch_q             = " << result.model_params.min_branch_q << "\n";
    std::cout << "  warn_branch_q            = " << result.model_params.warn_branch_q << "\n";
    std::cout << "  max_ron_stack_ohm        = " << result.model_params.max_ron_stack_ohm << "\n";
    std::cout << "  warn_ron_stack_ohm       = " << result.model_params.warn_ron_stack_ohm << "\n";
    std::cout << "  weight_off               = " << result.model_params.weight_off << "\n";
    std::cout << "  weight_area              = " << result.model_params.weight_area << "\n";
    std::cout << "  weight_ron               = " << result.model_params.weight_ron << "\n\n";

    std::cout << "Warnings:\n";
    for (const auto& warning : result.warnings) {
        std::cout << "  - " << warning << "\n";
    }
    std::cout << "\n============================================================\n";
    std::cout << "Final schematic values must be verified by extracted Ceff_on/Ceff_off simulation. "
                 "This script only provides parasitic-aware first-pass drawn sizing.\n";
}

std::string json_escape(const std::string& text) {
    std::ostringstream out;
    for (char ch : text) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

std::string json_number(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream out;
    out << std::setprecision(12) << value;
    return out.str();
}

std::string freq_key(double freq_ghz) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6) << freq_ghz;
    std::string text = out.str();
    while (!text.empty() && text.back() == '0') text.pop_back();
    if (!text.empty() && text.back() == '.') text.pop_back();
    if (text.find('.') == std::string::npos) text += ".0";
    return text + "GHz";
}

[[maybe_unused]] void print_json(const OptimizationResult& result) {
    std::cout << std::setprecision(12);
    std::cout << "{\n";
    std::cout << "  \"cell\": \"" << json_escape(result.cell) << "\",\n";
    std::cout << "  \"type\": \"" << json_escape(result.cell_type) << "\",\n";
    std::cout << "  \"verdict\": \"" << json_escape(result.verdict) << "\",\n";
    std::cout << "  \"targets_pf\": {\"N77\": " << result.targets.n77_pf
              << ", \"N78\": " << result.targets.n78_pf
              << ", \"N79\": " << result.targets.n79_pf << "},\n";
    std::cout << "  \"selected_mos\": {\n";
    const std::array<std::pair<std::string, MOSSwitchParams>, 2> branch_mos = {{
        {"N77", result.switches.n77},
        {"N78", result.switches.n78},
    }};
    for (std::size_t i = 0; i < branch_mos.size(); ++i) {
        const auto& item = branch_mos[i];
        const auto& mos = item.second;
        std::cout << "    \"" << item.first << "\": {"
                  << "\"model\": \"" << json_escape(mos.model) << "\", "
                  << "\"channel_width_um\": " << mos.channel_width_um << ", "
                  << "\"channel_length_nm\": " << mos.channel_length_nm << ", "
                  << "\"fingers\": " << mos.fingers << ", "
                  << "\"multiplier\": " << mos.multiplier << ", "
                  << "\"total_width_per_transistor_um\": "
                  << mos.total_width_per_transistor_um() << ", "
                  << "\"bodytie\": " << (mos.bodytie ? "true" : "false") << ", "
                  << "\"stack_devices_per_branch\": " << mos.stack_devices_per_branch
                  << "}";
        std::cout << (i + 1 == branch_mos.size() ? "\n" : ",\n");
    }
    std::cout << "  },\n";
    std::cout << "  \"optimized_drawn_pf\": {\"FIX\": " << result.drawn_pf[0]
              << ", \"N77_DELTA\": " << result.drawn_pf[1]
              << ", \"N78_DELTA\": " << result.drawn_pf[2] << "},\n";
    std::cout << "  \"predicted_effective_pf\": {\n";
    for (std::size_t i = 0; i < result.predictions.size(); ++i) {
        const auto& p = result.predictions[i];
        std::cout << "    \"" << freq_key(p.freq_ghz) << "\": {\"N77\": "
                  << p.ceff_n77_pf << ", \"N78\": " << p.ceff_n78_pf
                  << ", \"N79\": " << p.ceff_n79_pf << "}";
        std::cout << (i + 1 == result.predictions.size() ? "\n" : ",\n");
    }
    std::cout << "  },\n";
    std::cout << "  \"capacitor_layout\": {\n";
    const std::array<std::string, 3> layout_keys = {"FIX", "N77_DELTA", "N78_DELTA"};
    for (std::size_t i = 0; i < layout_keys.size(); ++i) {
        const auto& key = layout_keys[i];
        const auto& l = result.layouts.at(key);
        std::cout << "    \"" << key << "\": {\"model\": \"" << l.model
                  << "\", \"multiplier\": " << l.multiplier
                  << ", \"capacitance_pf\": " << l.capacitance_pf
                  << ", \"capacitance_ff\": " << l.capacitance_ff
                  << ", \"width_um\": " << l.width_um
                  << ", \"length_um\": " << l.length_um
                  << ", \"area_um2\": " << l.area_um2
                  << ", \"status\": \"" << l.status << "\"}";
        std::cout << (i + 1 == layout_keys.size() ? "\n" : ",\n");
    }
    std::cout << "  },\n";
    std::cout << "  \"gate_resistor\": {\"model\": \"" << result.gate_resistor.model
              << "\", \"resistance_ohm\": " << result.gate_resistor.resistance_ohm
              << ", \"segW_um\": " << result.gate_resistor.segW_um
              << ", \"segL_um\": " << result.gate_resistor.segL_um
              << ", \"segments\": " << result.gate_resistor.segments
              << ", \"multiplier\": " << result.gate_resistor.multiplier
              << ", \"connection\": \"" << result.gate_resistor.connection << "\"},\n";
    std::cout << "  \"midpoint_bias_resistor\": {\"model\": \"" << result.midpoint_bias_resistor.model
              << "\", \"resistance_ohm\": " << result.midpoint_bias_resistor.resistance_ohm
              << ", \"segW_um\": " << result.midpoint_bias_resistor.segW_um
              << ", \"segL_um\": " << result.midpoint_bias_resistor.segL_um
              << ", \"segments\": " << result.midpoint_bias_resistor.segments
              << ", \"multiplier\": " << result.midpoint_bias_resistor.multiplier
              << ", \"connection\": \"" << result.midpoint_bias_resistor.connection
              << "\", \"note\": \"" << json_escape(result.midpoint_bias_resistor.note) << "\"},\n";
    std::cout << "  \"warnings\": [";
    for (std::size_t i = 0; i < result.warnings.size(); ++i) {
        std::cout << "\"" << json_escape(result.warnings[i]) << "\"";
        if (i + 1 != result.warnings.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "}\n";
}

std::string csv_escape(const std::string& text) {
    if (text.find_first_of(",\"\n\r") == std::string::npos) {
        return text;
    }
    std::string escaped = "\"";
    for (char ch : text) {
        if (ch == '"') escaped += "\"\"";
        else escaped += ch;
    }
    escaped += "\"";
    return escaped;
}

std::string join_warnings(const std::vector<std::string>& warnings) {
    std::ostringstream out;
    for (std::size_t i = 0; i < warnings.size(); ++i) {
        if (i != 0) out << " | ";
        out << warnings[i];
    }
    return out.str();
}

[[maybe_unused]] void print_csv(const OptimizationResult& result) {
    std::cout << "cell,type,verdict,CN77_pf,CN78_pf,CN79_pf,"
                 "N77_MOS_multiplier,N77_MOS_total_width_um,"
                 "N78_MOS_multiplier,N78_MOS_total_width_um,"
                 "C_FIX_drawn_pf,C_N77_DELTA_drawn_pf,"
                 "C_N78_DELTA_drawn_pf,C_FIX_W_um,C_FIX_L_um,"
                 "C_N77_DELTA_W_um,C_N77_DELTA_L_um,C_N78_DELTA_W_um,"
                 "C_N78_DELTA_L_um,N77_worst_error_pct,N78_worst_error_pct,"
                 "N79_worst_error_pct,Rgate_ohm,Rmid_ohm,warnings\n";
    std::cout << std::setprecision(12)
              << csv_escape(result.cell) << ","
              << csv_escape(result.cell_type) << ","
              << csv_escape(result.verdict) << ","
              << result.targets.n77_pf << ","
              << result.targets.n78_pf << ","
              << result.targets.n79_pf << ","
              << result.switches.n77.multiplier << ","
              << result.switches.n77.total_width_per_transistor_um() << ","
              << result.switches.n78.multiplier << ","
              << result.switches.n78.total_width_per_transistor_um() << ","
              << result.drawn_pf[0] << ","
              << result.drawn_pf[1] << ","
              << result.drawn_pf[2] << ","
              << result.layouts.at("FIX").width_um << ","
              << result.layouts.at("FIX").length_um << ","
              << result.layouts.at("N77_DELTA").width_um << ","
              << result.layouts.at("N77_DELTA").length_um << ","
              << result.layouts.at("N78_DELTA").width_um << ","
              << result.layouts.at("N78_DELTA").length_um << ","
              << result.worst_error_pct.at("N77") << ","
              << result.worst_error_pct.at("N78") << ","
              << result.worst_error_pct.at("N79") << ","
              << result.gate_resistor.resistance_ohm << ","
              << result.midpoint_bias_resistor.resistance_ohm << ","
              << csv_escape(join_warnings(result.warnings)) << "\n";
}

struct PrelayoutConfig {
    bool conservative_preset = true;
    std::string parasitic_model = "conservative_scalar";
    double freq_hz = 5.0e9;
    std::vector<double> report_freqs_ghz = {3.3, 3.8, 4.2, 4.6, 5.0};
    double eps_rel = 0.01;
    double eps_abs_pf = 0.01;
    double min_branch_q = 5.0;
    double warn_branch_q = 10.0;
    double max_ron_stack_ohm = 20.0;
    double warn_ron_stack_ohm = 8.0;

    double mos_w_unit_um = 5.0;
    double mos_l_nm = 350.0;
    int mos_fingers = 40;
    int mos_stack_devices_per_branch = 2;
    bool bodytie = true;
    double body_effect_ron_factor = 1.20;
    int mos_mult_max = 200;

    double mim_density_pf_per_um2 = 0.0021;
    double mim_max_w_um = 25.0;
    double mim_max_l_um = 25.0;
    int mim_mult_max = 128;

    double kr_stack = 8500.0;
    double kr_stack_paper_ohm_um = 7130.0;
    double coff_per_um_ff = 0.52;
    double coff_device_cdb_06v_ff_per_um = 0.54;
    double coff_device_cgd0_ff_per_um = 0.50;
    double mim_q = 50.0;
    double mim_r_route_fixed_ohm = 0.02;
    double mim_r_route_per_unit_ohm = 0.005;
    double mim_r_via_per_unit_ohm = 0.002;
    double mim_c_fringe_per_unit_pf = 0.003;
    double mim_c_route_per_unit_pf = 0.002;
    double mim_c_coupling_per_gap_pf = 0.001;
    double c_common_route_pf = 0.005;
    double c_on_extra_per_um_ff = 0.50;
    double c_off_extra_direct_per_um_ff = 0.0;
    double c_mid_per_um_ff = 1.5;
    double c_bias_res_par_pf = 0.001;
    double mid_factor_on = 0.1;
    double mid_factor_off = 0.05;
    double bias_resistance_ohm = 500000.0;
    double mim_bias_v = 0.0;
    double mim_temp_c = 25.0;
    double mim_vc1_ppm_per_v = -61.2;
    double mim_vc2_ppm_per_v2 = 26.6;
    double mim_tc1_ppm_per_c = 32.6;

    double mim_vt_factor() const {
        return 1.0 +
               mim_vc1_ppm_per_v * 1.0e-6 * mim_bias_v +
               mim_vc2_ppm_per_v2 * 1.0e-6 * mim_bias_v * mim_bias_v +
               mim_tc1_ppm_per_c * 1.0e-6 * (mim_temp_c - 25.0);
    }

    double mim_density_effective_pf_per_um2() const {
        return mim_density_pf_per_um2 * mim_vt_factor();
    }

    double mim_unit_max_pf() const {
        return mim_density_effective_pf_per_um2() * mim_max_w_um * mim_max_l_um;
    }
};

struct PrelayoutBranch {
    std::string branch;
    double delta_target_pf = 0.0;
    bool omitted = false;
    bool solve_failed = false;
    std::string failure_reason = "";
    int mos_multiplier = 0;
    double mos_w_unit_um = 0.0;
    double mos_l_nm = 0.0;
    int mos_fingers = 0;
    int mos_stack_devices_per_branch = 2;
    bool bodytie = true;
    double mos_total_width_um = 0.0;
    double ron_stack_ohm = 0.0;
    int mim_multiplier = 0;
    double mim_unit_area_cap_pf = 0.0;
    double mim_total_area_cap_pf = 0.0;
    double mim_parasitic_cap_pf = 0.0;
    double mim_rf_cap_pf = 0.0;
    double mim_unit_w_um = 0.0;
    double mim_unit_l_um = 0.0;
    double r_mim_esr_ohm = 0.0;
    double r_mim_route_ohm = 0.0;
    double r_branch_on_ohm = 0.0;
    double c_sw_off_stack_pf = 0.0;
    double c_on_extra_pf = 0.0;
    double c_off_extra_direct_pf = 0.0;
    double c_mid_pf = 0.0;
    double c_mid_diffusion_pf = 0.0;
    double c_on_eff_pf = 0.0;
    double g_on_s = 0.0;
    double q_on = std::numeric_limits<double>::infinity();
    double min_q = std::numeric_limits<double>::infinity();
    double min_q_freq_ghz = 0.0;
    double c_off_branch_pf = 0.0;
    double finite_r_loss_pf = 0.0;
    double finite_r_rel_loss = 0.0;
    double omega_r_c = 0.0;
    double ron_max_rel_ohm = std::numeric_limits<double>::infinity();
    double ron_max_abs_ohm = std::numeric_limits<double>::infinity();
    double w_req_stack_only_um = 0.0;
    double r_bias_min_ohm = 0.0;
    double equation_error_pf = 0.0;
};

struct PrelayoutFixed {
    int mim_multiplier = 0;
    double mim_unit_area_cap_pf = 0.0;
    double mim_total_area_cap_pf = 0.0;
    double mim_parasitic_cap_pf = 0.0;
    double mim_rf_cap_pf = 0.0;
    double mim_unit_w_um = 0.0;
    double mim_unit_l_um = 0.0;
};

struct PrelayoutResult {
    std::string cell;
    Targets targets;
    PrelayoutConfig cfg;
    PrelayoutBranch n77;
    PrelayoutBranch n78;
    PrelayoutFixed fix;
    bool fixed_cap_valid = false;
    std::string verdict = "FAILED";
    std::vector<std::string> warnings;
    double c_fix_target_rf_pf = 0.0;
    double max_mode_error_pf = std::numeric_limits<double>::infinity();
    double c_n79_pf = 0.0;
    double c_n77_pf = 0.0;
    double c_n78_pf = 0.0;
};

double scalar_option_or(
    const CommandLine& args,
    const std::string& key,
    double fallback) {
    for (const auto& item : args.scalar_options) {
        if (dash_to_underscore(item.first) == key) {
            return parse_double(item.second, "--" + item.first);
        }
    }
    return fallback;
}

int scalar_int_option_or(
    const CommandLine& args,
    const std::string& key,
    int fallback) {
    for (const auto& item : args.scalar_options) {
        if (dash_to_underscore(item.first) == key) {
            return parse_int(item.second, "--" + item.first);
        }
    }
    return fallback;
}

PrelayoutConfig make_prelayout_config_from_args(const CommandLine& args) {
    PrelayoutConfig cfg;
    cfg.parasitic_model = args.parasitic_model;
    cfg.mos_w_unit_um = args.mos_width_um;
    cfg.mos_l_nm = args.mos_length_nm;
    cfg.mos_fingers = args.mos_fingers;
    cfg.mos_stack_devices_per_branch = args.mos_stack_devices_per_branch;
    cfg.bodytie = args.bodytie;
    cfg.body_effect_ron_factor =
        scalar_option_or(args, "body_effect_ron_factor", cfg.body_effect_ron_factor);
    cfg.mim_density_pf_per_um2 = args.density_ff_per_um2 * 1.0e-3;
    cfg.kr_stack = scalar_option_or(args, "kr_stack_ohm_um", cfg.kr_stack);
    cfg.coff_per_um_ff = scalar_option_or(args, "coff_stack_ff_per_um", cfg.coff_per_um_ff);
    cfg.mim_q = scalar_option_or(args, "mim_q", cfg.mim_q);
    cfg.mim_r_route_fixed_ohm =
        scalar_option_or(args, "mim_r_route_fixed_ohm", cfg.mim_r_route_fixed_ohm);
    cfg.mim_r_route_per_unit_ohm =
        scalar_option_or(args, "mim_r_route_per_unit_ohm", cfg.mim_r_route_per_unit_ohm);
    cfg.mim_r_via_per_unit_ohm =
        scalar_option_or(args, "mim_r_via_per_unit_ohm", cfg.mim_r_via_per_unit_ohm);
    cfg.mim_c_fringe_per_unit_pf =
        scalar_option_or(args, "mim_c_fringe_per_unit_pf", cfg.mim_c_fringe_per_unit_pf);
    cfg.mim_c_route_per_unit_pf =
        scalar_option_or(args, "mim_c_route_per_unit_pf", cfg.mim_c_route_per_unit_pf);
    cfg.mim_c_coupling_per_gap_pf =
        scalar_option_or(args, "mim_c_coupling_per_gap_pf", cfg.mim_c_coupling_per_gap_pf);
    cfg.c_common_route_pf = scalar_option_or(args, "c_common_route_pf", cfg.c_common_route_pf);
    cfg.c_on_extra_per_um_ff =
        scalar_option_or(args, "c_on_extra_ff_per_um", cfg.c_on_extra_per_um_ff);
    cfg.c_off_extra_direct_per_um_ff =
        scalar_option_or(args, "c_off_extra_direct_ff_per_um", cfg.c_off_extra_direct_per_um_ff);
    cfg.c_mid_per_um_ff = scalar_option_or(args, "c_mid_ff_per_um", cfg.c_mid_per_um_ff);
    cfg.c_bias_res_par_pf =
        scalar_option_or(args, "c_bias_res_par_pf", cfg.c_bias_res_par_pf);
    cfg.mid_factor_on = scalar_option_or(args, "mid_factor_on", cfg.mid_factor_on);
    cfg.mid_factor_off = scalar_option_or(args, "mid_factor_off", cfg.mid_factor_off);
    cfg.bias_resistance_ohm =
        scalar_option_or(args, "bias_resistance_ohm", cfg.bias_resistance_ohm);
    cfg.mim_bias_v = scalar_option_or(args, "mim_bias_v", cfg.mim_bias_v);
    cfg.mim_temp_c = scalar_option_or(args, "mim_temp_c", cfg.mim_temp_c);
    cfg.mim_vc1_ppm_per_v =
        scalar_option_or(args, "mim_vc1_ppm_per_v", cfg.mim_vc1_ppm_per_v);
    cfg.mim_vc2_ppm_per_v2 =
        scalar_option_or(args, "mim_vc2_ppm_per_v2", cfg.mim_vc2_ppm_per_v2);
    cfg.mim_tc1_ppm_per_c =
        scalar_option_or(args, "mim_tc1_ppm_per_c", cfg.mim_tc1_ppm_per_c);
    cfg.eps_rel = scalar_option_or(args, "eps_rel", cfg.eps_rel);
    cfg.eps_abs_pf = scalar_option_or(args, "eps_abs_pf", cfg.eps_abs_pf);
    cfg.min_branch_q = scalar_option_or(args, "min_branch_q", cfg.min_branch_q);
    cfg.warn_branch_q = scalar_option_or(args, "warn_branch_q", cfg.warn_branch_q);
    cfg.max_ron_stack_ohm =
        scalar_option_or(args, "max_ron_stack_ohm", cfg.max_ron_stack_ohm);
    cfg.warn_ron_stack_ohm =
        scalar_option_or(args, "warn_ron_stack_ohm", cfg.warn_ron_stack_ohm);
    cfg.mim_max_w_um = scalar_option_or(args, "mim_max_w_um", cfg.mim_max_w_um);
    cfg.mim_max_l_um = scalar_option_or(args, "mim_max_l_um", cfg.mim_max_l_um);
    cfg.mim_mult_max = scalar_int_option_or(args, "mim_mult_max", cfg.mim_mult_max);
    cfg.mos_mult_max = scalar_int_option_or(args, "mos_mult_max", cfg.mos_mult_max);
    if (cfg.mos_w_unit_um <= 0.0 || cfg.mos_l_nm <= 0.0 ||
        cfg.mos_fingers <= 0 || cfg.mos_stack_devices_per_branch <= 0) {
        throw std::runtime_error("MOS width, length, fingers, and stack devices must be positive");
    }
    if (cfg.kr_stack <= 0.0 || cfg.coff_per_um_ff < 0.0) {
        throw std::runtime_error("KR stack must be positive and Coff per um must be non-negative");
    }
    if (cfg.mim_density_pf_per_um2 <= 0.0 || cfg.mim_density_effective_pf_per_um2() <= 0.0) {
        throw std::runtime_error("MIM density and voltage/temperature corrected density must be positive");
    }
    if (cfg.mim_max_w_um <= 0.0 || cfg.mim_max_l_um <= 0.0 ||
        cfg.mim_mult_max <= 0 || cfg.mos_mult_max <= 0) {
        throw std::runtime_error("MIM/MOS layout maxima must be positive");
    }
    if (cfg.eps_rel <= 0.0 || cfg.eps_abs_pf <= 0.0) {
        throw std::runtime_error("Finite-R loss limits must be positive");
    }
    if (cfg.min_branch_q <= 0.0 || cfg.warn_branch_q <= 0.0 ||
        cfg.max_ron_stack_ohm <= 0.0 || cfg.warn_ron_stack_ohm <= 0.0) {
        throw std::runtime_error("Ron/Q warning thresholds must be positive");
    }
    if (cfg.warn_branch_q < cfg.min_branch_q ||
        cfg.max_ron_stack_ohm < cfg.warn_ron_stack_ohm) {
        throw std::runtime_error("Ron/Q warning thresholds are inconsistent");
    }
    if (cfg.body_effect_ron_factor <= 0.0) {
        throw std::runtime_error("body-effect Ron factor must be positive");
    }
    if (cfg.mim_q <= 0.0 && !std::isinf(cfg.mim_q)) {
        throw std::runtime_error("MIM Q must be positive or infinity");
    }
    if (!args.freqs_ghz.empty()) {
        cfg.report_freqs_ghz = args.freqs_ghz;
        cfg.freq_hz = *std::max_element(args.freqs_ghz.begin(), args.freqs_ghz.end()) * 1.0e9;
    }

    if (cfg.parasitic_model == "no_midpoint") {
        cfg.mid_factor_on = 0.0;
        cfg.mid_factor_off = 0.0;
        cfg.c_bias_res_par_pf = 0.0;
    } else if (cfg.parasitic_model == "no_on_extra") {
        cfg.c_on_extra_per_um_ff = 0.0;
    } else if (cfg.parasitic_model == "simple") {
        cfg.c_on_extra_per_um_ff = 0.0;
        cfg.c_off_extra_direct_per_um_ff = 0.0;
        cfg.c_mid_per_um_ff = 0.0;
        cfg.c_bias_res_par_pf = 0.0;
        cfg.mid_factor_on = 0.0;
        cfg.mid_factor_off = 0.0;
        cfg.mim_r_route_fixed_ohm = 0.0;
        cfg.mim_r_route_per_unit_ohm = 0.0;
        cfg.mim_r_via_per_unit_ohm = 0.0;
        cfg.mim_c_fringe_per_unit_pf = 0.0;
        cfg.mim_c_route_per_unit_pf = 0.0;
        cfg.mim_c_coupling_per_gap_pf = 0.0;
        cfg.c_common_route_pf = 0.0;
    }
    return cfg;
}

double prelayout_omega(const PrelayoutConfig& cfg) {
    return 2.0 * PI * cfg.freq_hz;
}

double prelayout_series_cap_pf(double c1_pf, double c2_pf) {
    if (c1_pf <= 0.0 || c2_pf <= 0.0) {
        return 0.0;
    }
    return c1_pf * c2_pf / (c1_pf + c2_pf);
}

double prelayout_mos_total_width_um(const PrelayoutConfig& cfg, int mos_m) {
    return cfg.mos_w_unit_um * static_cast<double>(cfg.mos_fingers) *
           static_cast<double>(mos_m);
}

double prelayout_kr_stack_effective_ohm_um(const PrelayoutConfig& cfg) {
    const double stack_factor =
        static_cast<double>(std::max(cfg.mos_stack_devices_per_branch, 1)) / 2.0;
    const double body_factor = cfg.bodytie ? 1.0 : cfg.body_effect_ron_factor;
    return cfg.kr_stack * stack_factor * body_factor;
}

double prelayout_ron_stack_ohm(const PrelayoutConfig& cfg, double wsw_um) {
    return prelayout_kr_stack_effective_ohm_um(cfg) / wsw_um;
}

double prelayout_c_sw_off_pf(const PrelayoutConfig& cfg, double wsw_um) {
    return cfg.coff_per_um_ff * wsw_um * 1.0e-3;
}

double prelayout_c_on_extra_pf(const PrelayoutConfig& cfg, double wsw_um) {
    return cfg.c_on_extra_per_um_ff * wsw_um * 1.0e-3;
}

double prelayout_c_off_extra_direct_pf(const PrelayoutConfig& cfg, double wsw_um) {
    return cfg.c_off_extra_direct_per_um_ff * wsw_um * 1.0e-3;
}

double prelayout_c_mid_diffusion_pf(const PrelayoutConfig& cfg, double wsw_um) {
    return cfg.c_mid_per_um_ff * wsw_um * 1.0e-3;
}

double prelayout_c_mid_pf(const PrelayoutConfig& cfg, double wsw_um) {
    return prelayout_c_mid_diffusion_pf(cfg, wsw_um) + cfg.c_bias_res_par_pf;
}

double prelayout_mim_parasitic_cap_pf(const PrelayoutConfig& cfg, int mim_m) {
    if (mim_m <= 0) {
        return 0.0;
    }
    return cfg.mim_c_fringe_per_unit_pf * mim_m +
           cfg.mim_c_route_per_unit_pf * mim_m +
           cfg.mim_c_coupling_per_gap_pf * std::max(mim_m - 1, 0);
}

double prelayout_mim_rf_cap_pf(
    const PrelayoutConfig& cfg,
    double c_unit_area_pf,
    int mim_m) {
    return static_cast<double>(mim_m) * c_unit_area_pf +
           prelayout_mim_parasitic_cap_pf(cfg, mim_m);
}

double prelayout_mim_unit_square_size_um(
    const PrelayoutConfig& cfg,
    double c_unit_area_pf) {
    if (c_unit_area_pf <= 0.0) {
        return 0.0;
    }
    return std::sqrt(c_unit_area_pf / cfg.mim_density_effective_pf_per_um2());
}

double prelayout_loss_x_limit(double rel_loss_limit) {
    if (rel_loss_limit <= 0.0) {
        return 0.0;
    }
    if (rel_loss_limit >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(rel_loss_limit / (1.0 - rel_loss_limit));
}

double prelayout_ron_max_rel_ohm(const PrelayoutConfig& cfg, double delta_pf) {
    if (delta_pf <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return prelayout_loss_x_limit(cfg.eps_rel) /
           (prelayout_omega(cfg) * delta_pf * 1.0e-12);
}

double prelayout_ron_max_abs_ohm(const PrelayoutConfig& cfg, double delta_pf) {
    if (delta_pf <= 0.0 || cfg.eps_abs_pf >= delta_pf) {
        return std::numeric_limits<double>::infinity();
    }
    const double x_abs =
        std::sqrt(cfg.eps_abs_pf / std::max(delta_pf - cfg.eps_abs_pf, 1.0e-30));
    return x_abs / (prelayout_omega(cfg) * delta_pf * 1.0e-12);
}

double prelayout_w_req_stack_only_um(const PrelayoutConfig& cfg, double delta_pf) {
    const double ron_allow =
        std::min(prelayout_ron_max_rel_ohm(cfg, delta_pf),
                 prelayout_ron_max_abs_ohm(cfg, delta_pf));
    if (!std::isfinite(ron_allow) || ron_allow <= 0.0) {
        return 0.0;
    }
    return prelayout_kr_stack_effective_ohm_um(cfg) / ron_allow;
}

double prelayout_bias_r_min_ohm(const PrelayoutConfig& cfg, double wsw_um) {
    const double c_mid_f = prelayout_c_mid_diffusion_pf(cfg, wsw_um) * 1.0e-12;
    if (c_mid_f <= 0.0) {
        return 0.0;
    }
    return 100.0 / (prelayout_omega(cfg) * c_mid_f);
}

double prelayout_r_mim_esr_ohm_at_hz(
    const PrelayoutConfig& cfg,
    double c_mim_rf_pf,
    double freq_hz) {
    if (c_mim_rf_pf <= 0.0) {
        return 0.0;
    }
    if (std::isinf(cfg.mim_q) || cfg.mim_q <= 0.0) {
        return 0.0;
    }
    const double omega = 2.0 * PI * freq_hz;
    return 1.0 / (omega * c_mim_rf_pf * 1.0e-12 * cfg.mim_q);
}

double prelayout_r_mim_esr_ohm(const PrelayoutConfig& cfg, double c_mim_rf_pf) {
    return prelayout_r_mim_esr_ohm_at_hz(cfg, c_mim_rf_pf, cfg.freq_hz);
}

double prelayout_r_mim_route_ohm(const PrelayoutConfig& cfg, int mim_m) {
    if (mim_m <= 0) {
        return 0.0;
    }
    return cfg.mim_r_route_fixed_ohm +
           cfg.mim_r_route_per_unit_ohm * static_cast<double>(mim_m) +
           cfg.mim_r_via_per_unit_ohm / static_cast<double>(mim_m);
}

double prelayout_r_branch_on_ohm(
    const PrelayoutConfig& cfg,
    double wsw_um,
    double c_mim_rf_pf,
    int mim_m) {
    return prelayout_ron_stack_ohm(cfg, wsw_um) +
           prelayout_r_mim_esr_ohm(cfg, c_mim_rf_pf) +
           prelayout_r_mim_route_ohm(cfg, mim_m);
}

double prelayout_r_branch_on_ohm_at_hz(
    const PrelayoutConfig& cfg,
    double wsw_um,
    double c_mim_rf_pf,
    int mim_m,
    double freq_hz) {
    return prelayout_ron_stack_ohm(cfg, wsw_um) +
           prelayout_r_mim_esr_ohm_at_hz(cfg, c_mim_rf_pf, freq_hz) +
           prelayout_r_mim_route_ohm(cfg, mim_m);
}

double prelayout_c_on_eff_pf_at_hz(
    const PrelayoutConfig& cfg,
    double c_mim_rf_pf,
    double wsw_um,
    int mim_m,
    double freq_hz) {
    const double r = prelayout_r_branch_on_ohm_at_hz(
        cfg, wsw_um, c_mim_rf_pf, mim_m, freq_hz);
    const double omega = 2.0 * PI * freq_hz;
    const double x = omega * r * c_mim_rf_pf * 1.0e-12;
    return c_mim_rf_pf / (1.0 + x * x) +
           prelayout_c_on_extra_pf(cfg, wsw_um) +
           cfg.mid_factor_on * prelayout_c_mid_pf(cfg, wsw_um);
}

double prelayout_c_on_eff_pf(
    const PrelayoutConfig& cfg,
    double c_mim_rf_pf,
    double wsw_um,
    int mim_m) {
    return prelayout_c_on_eff_pf_at_hz(cfg, c_mim_rf_pf, wsw_um, mim_m, cfg.freq_hz);
}

double prelayout_g_on_s_at_hz(
    const PrelayoutConfig& cfg,
    double c_mim_rf_pf,
    double wsw_um,
    int mim_m,
    double freq_hz) {
    if (c_mim_rf_pf <= 0.0) {
        return 0.0;
    }
    const double r = prelayout_r_branch_on_ohm_at_hz(
        cfg, wsw_um, c_mim_rf_pf, mim_m, freq_hz);
    const double omega = 2.0 * PI * freq_hz;
    const double c_f = c_mim_rf_pf * 1.0e-12;
    const double x = omega * r * c_f;
    return omega * omega * r * c_f * c_f / (1.0 + x * x);
}

double prelayout_g_on_s(
    const PrelayoutConfig& cfg,
    double c_mim_rf_pf,
    double wsw_um,
    int mim_m) {
    return prelayout_g_on_s_at_hz(cfg, c_mim_rf_pf, wsw_um, mim_m, cfg.freq_hz);
}

double prelayout_q_on_at_hz(
    const PrelayoutConfig& cfg,
    double c_mim_rf_pf,
    double wsw_um,
    int mim_m,
    double freq_hz) {
    const double g = prelayout_g_on_s_at_hz(cfg, c_mim_rf_pf, wsw_um, mim_m, freq_hz);
    if (g <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const double c_on_eff_f =
        prelayout_c_on_eff_pf_at_hz(cfg, c_mim_rf_pf, wsw_um, mim_m, freq_hz) *
        1.0e-12;
    return (2.0 * PI * freq_hz * c_on_eff_f) / g;
}

double prelayout_c_off_branch_pf(
    const PrelayoutConfig& cfg,
    double c_mim_rf_pf,
    double wsw_um) {
    return prelayout_series_cap_pf(c_mim_rf_pf, prelayout_c_sw_off_pf(cfg, wsw_um)) +
           prelayout_c_off_extra_direct_pf(cfg, wsw_um) +
           cfg.mid_factor_off * prelayout_c_mid_pf(cfg, wsw_um);
}

double prelayout_finite_r_loss_pf(
    const PrelayoutConfig& cfg,
    double c_mim_rf_pf,
    double wsw_um,
    int mim_m) {
    const double r = prelayout_r_branch_on_ohm(cfg, wsw_um, c_mim_rf_pf, mim_m);
    const double x = prelayout_omega(cfg) * r * c_mim_rf_pf * 1.0e-12;
    return c_mim_rf_pf - c_mim_rf_pf / (1.0 + x * x);
}

bool prelayout_solve_unit_area_cap_for_branch(
    const PrelayoutConfig& cfg,
    double delta_pf,
    int mos_m,
    int mim_m,
    double& c_unit_pf) {
    const double wsw_um = prelayout_mos_total_width_um(cfg, mos_m);
    const auto f = [&](double unit_pf) {
        const double c_rf = prelayout_mim_rf_cap_pf(cfg, unit_pf, mim_m);
        return prelayout_c_on_eff_pf(cfg, c_rf, wsw_um, mim_m) -
               prelayout_c_off_branch_pf(cfg, c_rf, wsw_um) - delta_pf;
    };

    double lo = 0.0;
    double hi = cfg.mim_unit_max_pf();
    if (f(lo) > 0.0 || f(hi) < 0.0) {
        return false;
    }

    for (int i = 0; i < 80; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (f(mid) < 0.0) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    c_unit_pf = 0.5 * (lo + hi);
    if (prelayout_mim_unit_square_size_um(cfg, c_unit_pf) > cfg.mim_max_w_um + 1.0e-9) {
        return false;
    }
    return true;
}

bool prelayout_loss_ok(
    const PrelayoutConfig& cfg,
    double c_mim_rf_pf,
    double wsw_um,
    int mim_m,
    double& loss_pf,
    double& rel_loss) {
    loss_pf = prelayout_finite_r_loss_pf(cfg, c_mim_rf_pf, wsw_um, mim_m);
    rel_loss = loss_pf / std::max(c_mim_rf_pf, 1.0e-30);
    return loss_pf <= cfg.eps_abs_pf + 1.0e-12 &&
           rel_loss <= cfg.eps_rel + 1.0e-12;
}

[[maybe_unused]] int prelayout_estimate_mos_start(
    const PrelayoutConfig& cfg,
    double delta_pf,
    int mim_m) {
    const double c_approx_pf =
        std::max(delta_pf + prelayout_mim_parasitic_cap_pf(cfg, mim_m), 1.0e-9);
    const double c_f = c_approx_pf * 1.0e-12;
    const double eps_abs_f = cfg.eps_abs_pf * 1.0e-12;

    const double x_rel_max =
        std::sqrt(cfg.eps_rel / std::max(1.0 - cfg.eps_rel, 1.0e-30));
    const double r_rel_max = x_rel_max / (prelayout_omega(cfg) * c_f);

    double r_abs_max = std::numeric_limits<double>::infinity();
    if (c_f > eps_abs_f) {
        const double x_abs_max =
            std::sqrt(eps_abs_f / std::max(c_f - eps_abs_f, 1.0e-30));
        r_abs_max = x_abs_max / (prelayout_omega(cfg) * c_f);
    }

    const double r_allow = std::min(r_rel_max, r_abs_max);
    const double r_non_mos =
        prelayout_r_mim_route_ohm(cfg, mim_m) +
        prelayout_r_mim_esr_ohm(cfg, c_approx_pf);
    const double r_allow_mos = std::max(r_allow - r_non_mos, 1.0e-6);
    const double w_req = prelayout_kr_stack_effective_ohm_um(cfg) / r_allow_mos;
    const int m_req = static_cast<int>(
        std::ceil(w_req / (cfg.mos_w_unit_um * static_cast<double>(cfg.mos_fingers))));
    return std::max(1, std::min(m_req, cfg.mos_mult_max));
}

bool prelayout_branch_candidate(
    const PrelayoutConfig& cfg,
    const std::string& branch_name,
    double delta_pf,
    int mos_m,
    int mim_m,
    PrelayoutBranch& out) {
    double c_unit = 0.0;
    if (!prelayout_solve_unit_area_cap_for_branch(cfg, delta_pf, mos_m, mim_m, c_unit)) {
        return false;
    }

    const double wsw_um = prelayout_mos_total_width_um(cfg, mos_m);
    const double c_rf = prelayout_mim_rf_cap_pf(cfg, c_unit, mim_m);
    double loss_pf = 0.0;
    double rel_loss = 0.0;
    prelayout_loss_ok(cfg, c_rf, wsw_um, mim_m, loss_pf, rel_loss);

    const double c_on = prelayout_c_on_eff_pf(cfg, c_rf, wsw_um, mim_m);
    const double c_off = prelayout_c_off_branch_pf(cfg, c_rf, wsw_um);
    out.branch = branch_name;
    out.delta_target_pf = delta_pf;
    out.omitted = false;
    out.mos_multiplier = mos_m;
    out.mos_w_unit_um = cfg.mos_w_unit_um;
    out.mos_l_nm = cfg.mos_l_nm;
    out.mos_fingers = cfg.mos_fingers;
    out.mos_stack_devices_per_branch = cfg.mos_stack_devices_per_branch;
    out.bodytie = cfg.bodytie;
    out.mos_total_width_um = wsw_um;
    out.ron_stack_ohm = prelayout_ron_stack_ohm(cfg, wsw_um);
    out.mim_multiplier = mim_m;
    out.mim_unit_area_cap_pf = c_unit;
    out.mim_total_area_cap_pf = c_unit * static_cast<double>(mim_m);
    out.mim_parasitic_cap_pf = prelayout_mim_parasitic_cap_pf(cfg, mim_m);
    out.mim_rf_cap_pf = c_rf;
    out.mim_unit_w_um = prelayout_mim_unit_square_size_um(cfg, c_unit);
    out.mim_unit_l_um = out.mim_unit_w_um;
    out.r_mim_esr_ohm = prelayout_r_mim_esr_ohm(cfg, c_rf);
    out.r_mim_route_ohm = prelayout_r_mim_route_ohm(cfg, mim_m);
    out.r_branch_on_ohm = prelayout_r_branch_on_ohm(cfg, wsw_um, c_rf, mim_m);
    out.c_sw_off_stack_pf = prelayout_c_sw_off_pf(cfg, wsw_um);
    out.c_on_extra_pf = prelayout_c_on_extra_pf(cfg, wsw_um);
    out.c_off_extra_direct_pf = prelayout_c_off_extra_direct_pf(cfg, wsw_um);
    out.c_mid_pf = prelayout_c_mid_pf(cfg, wsw_um);
    out.c_mid_diffusion_pf = prelayout_c_mid_diffusion_pf(cfg, wsw_um);
    out.c_on_eff_pf = c_on;
    out.g_on_s = prelayout_g_on_s(cfg, c_rf, wsw_um, mim_m);
    out.q_on = prelayout_q_on_at_hz(cfg, c_rf, wsw_um, mim_m, cfg.freq_hz);
    out.min_q = std::numeric_limits<double>::infinity();
    out.min_q_freq_ghz = 0.0;
    for (double freq_ghz : cfg.report_freqs_ghz) {
        const double q = prelayout_q_on_at_hz(cfg, c_rf, wsw_um, mim_m, freq_ghz * 1.0e9);
        if (q < out.min_q) {
            out.min_q = q;
            out.min_q_freq_ghz = freq_ghz;
        }
    }
    out.c_off_branch_pf = c_off;
    out.finite_r_loss_pf = loss_pf;
    out.finite_r_rel_loss = rel_loss;
    out.omega_r_c = prelayout_omega(cfg) * out.r_branch_on_ohm * c_rf * 1.0e-12;
    out.ron_max_rel_ohm = prelayout_ron_max_rel_ohm(cfg, delta_pf);
    out.ron_max_abs_ohm = prelayout_ron_max_abs_ohm(cfg, delta_pf);
    out.w_req_stack_only_um = prelayout_w_req_stack_only_um(cfg, delta_pf);
    out.r_bias_min_ohm = prelayout_bias_r_min_ohm(cfg, wsw_um);
    out.equation_error_pf = std::abs((c_on - c_off) - delta_pf);
    return true;
}

PrelayoutBranch prelayout_omitted_branch(
    const std::string& branch_name,
    double delta_pf) {
    PrelayoutBranch out;
    out.branch = branch_name;
    out.delta_target_pf = delta_pf;
    out.omitted = true;
    return out;
}

PrelayoutBranch prelayout_failed_branch(
    const std::string& branch_name,
    double delta_pf,
    const std::string& reason) {
    PrelayoutBranch out;
    out.branch = branch_name;
    out.delta_target_pf = delta_pf;
    out.solve_failed = true;
    out.failure_reason = reason;
    return out;
}

double prelayout_branch_candidate_score(
    const PrelayoutConfig& cfg,
    const PrelayoutBranch& b) {
    double score = 0.0;
    if (b.finite_r_loss_pf > cfg.eps_abs_pf) {
        score += 1000.0 * (b.finite_r_loss_pf - cfg.eps_abs_pf) / cfg.eps_abs_pf;
    }
    if (b.finite_r_rel_loss > cfg.eps_rel) {
        score += 1000.0 * (b.finite_r_rel_loss - cfg.eps_rel) / cfg.eps_rel;
    }
    if (b.min_q < cfg.min_branch_q) {
        score += 500.0 * (cfg.min_branch_q - b.min_q);
    } else if (b.min_q < cfg.warn_branch_q) {
        score += 20.0 * (cfg.warn_branch_q - b.min_q);
    }
    if (b.ron_stack_ohm > cfg.max_ron_stack_ohm) {
        score += 100.0 * (b.ron_stack_ohm - cfg.max_ron_stack_ohm);
    } else if (b.ron_stack_ohm > cfg.warn_ron_stack_ohm) {
        score += 5.0 * (b.ron_stack_ohm - cfg.warn_ron_stack_ohm);
    }
    if (b.mos_total_width_um > 5000.0) {
        score += 0.002 * (b.mos_total_width_um - 5000.0);
    }
    score += 1.0e-4 * b.mos_total_width_um;
    score += 1.0e-3 * static_cast<double>(b.mim_multiplier);
    score += 1.0e6 * b.equation_error_pf;
    return score;
}

PrelayoutBranch prelayout_solve_branch(
    const PrelayoutConfig& cfg,
    const std::string& branch_name,
    double delta_pf) {
    std::vector<PrelayoutBranch> candidates;
    const int min_mim_m = std::max(
        1,
        static_cast<int>(std::ceil(
            std::max(delta_pf - prelayout_mim_parasitic_cap_pf(cfg, 1), 0.0) /
            cfg.mim_unit_max_pf())));

    for (int mim_m = min_mim_m; mim_m <= cfg.mim_mult_max; ++mim_m) {
        for (int mos_m = 1; mos_m <= cfg.mos_mult_max; ++mos_m) {
            PrelayoutBranch cand;
            if (prelayout_branch_candidate(cfg, branch_name, delta_pf, mos_m, mim_m, cand)) {
                candidates.push_back(cand);
            }
        }
    }

    if (candidates.empty()) {
        return prelayout_failed_branch(
            branch_name,
            delta_pf,
            "No valid solution. Increase MOS_MULT_MAX/MIM_MULT_MAX, relax EPS limits, or segment the branch.");
    }

    std::sort(candidates.begin(), candidates.end(),
              [&](const PrelayoutBranch& a, const PrelayoutBranch& b) {
                  const double score_a = prelayout_branch_candidate_score(cfg, a);
                  const double score_b = prelayout_branch_candidate_score(cfg, b);
                  if (std::abs(score_a - score_b) > 1.0e-12) {
                      return score_a < score_b;
                  }
                  if (a.mos_multiplier != b.mos_multiplier) {
                      return a.mos_multiplier < b.mos_multiplier;
                  }
                  return a.mim_multiplier < b.mim_multiplier;
              });
    return candidates.front();
}

PrelayoutFixed prelayout_solve_fixed_mim(
    const PrelayoutConfig& cfg,
    double target_rf_cap_pf) {
    for (int mim_m = 1; mim_m <= cfg.mim_mult_max; ++mim_m) {
        const double parasitic = prelayout_mim_parasitic_cap_pf(cfg, mim_m);
        const double area_total = target_rf_cap_pf - parasitic;
        if (area_total < -1.0e-12) {
            continue;
        }
        const double c_unit = area_total / static_cast<double>(mim_m);
        if (0.0 <= c_unit && c_unit <= cfg.mim_unit_max_pf()) {
            const double w = prelayout_mim_unit_square_size_um(cfg, c_unit);
            if (w <= cfg.mim_max_w_um + 1.0e-9) {
                PrelayoutFixed fix;
                fix.mim_multiplier = mim_m;
                fix.mim_unit_area_cap_pf = c_unit;
                fix.mim_total_area_cap_pf = c_unit * static_cast<double>(mim_m);
                fix.mim_parasitic_cap_pf = parasitic;
                fix.mim_rf_cap_pf = prelayout_mim_rf_cap_pf(cfg, c_unit, mim_m);
                fix.mim_unit_w_um = w;
                fix.mim_unit_l_um = w;
                return fix;
            }
        }
    }
    throw std::runtime_error(
        "No valid fixed MIM solution. Disabled branch residual may exceed N79 budget or MIM PCell bounds.");
}

PrelayoutResult solve_prelayout_result(
    const CommandLine& args,
    const Targets& targets,
    const PrelayoutConfig& cfg) {
    if (targets.n77_pf + 1.0e-12 < targets.n79_pf ||
        targets.n78_pf + 1.0e-12 < targets.n79_pf) {
        throw std::runtime_error(
            "This single-cell solver assumes N79 is the minimum/base capacitance.");
    }

    PrelayoutResult result;
    result.cell = args.name.empty() ? "C3" : args.name;
    result.targets = targets;
    result.cfg = cfg;
    const double delta_n77 = targets.n77_pf - targets.n79_pf;
    const double delta_n78 = targets.n78_pf - targets.n79_pf;
    result.n77 = delta_n77 < args.min_branch_pf - 1.0e-12
                     ? prelayout_omitted_branch(result.cell + "_N77", delta_n77)
                     : prelayout_solve_branch(cfg, result.cell + "_N77", delta_n77);
    result.n78 = delta_n78 < args.min_branch_pf - 1.0e-12
                     ? prelayout_omitted_branch(result.cell + "_N78", delta_n78)
                     : prelayout_solve_branch(cfg, result.cell + "_N78", delta_n78);
    if (result.n77.omitted && delta_n77 > 1.0e-12) {
        std::ostringstream msg;
        msg << "N77 branch omitted because delta=" << std::fixed << std::setprecision(6)
            << delta_n77 << " pF is at or below min_branch_pf=" << args.min_branch_pf
            << " pF; absorb it into tolerance and verify in S-parameters.";
        push_unique(result.warnings, msg.str());
    }
    if (result.n78.omitted && delta_n78 > 1.0e-12) {
        std::ostringstream msg;
        msg << "N78 branch omitted because delta=" << std::fixed << std::setprecision(6)
            << delta_n78 << " pF is at or below min_branch_pf=" << args.min_branch_pf
            << " pF; absorb it into tolerance and verify in S-parameters.";
        push_unique(result.warnings, msg.str());
    }
    if (result.n77.solve_failed) {
        push_unique(result.warnings, "N77 branch sizing failed: " + result.n77.failure_reason);
    }
    if (result.n78.solve_failed) {
        push_unique(result.warnings, "N78 branch sizing failed: " + result.n78.failure_reason);
    }
    result.c_fix_target_rf_pf =
        targets.n79_pf - result.n77.c_off_branch_pf -
        result.n78.c_off_branch_pf - cfg.c_common_route_pf;
    if (result.c_fix_target_rf_pf >= -1.0e-12) {
        try {
            result.fix = prelayout_solve_fixed_mim(cfg, std::max(result.c_fix_target_rf_pf, 0.0));
            result.fixed_cap_valid = true;
        } catch (const std::exception& exc) {
            result.fixed_cap_valid = false;
            push_unique(result.warnings, exc.what());
        }
    } else {
        std::ostringstream msg;
        msg << "No non-negative C_FIX is possible: disabled branch residuals exceed CN79 by "
            << std::fixed << std::setprecision(6) << -result.c_fix_target_rf_pf
            << " pF. This matches the paper warning for large C3/C6 branches and requires segmentation, reduced switch width, or architecture-level correction.";
        push_unique(result.warnings, msg.str());
    }

    result.c_n79_pf =
        result.fix.mim_rf_cap_pf + result.n77.c_off_branch_pf +
        result.n78.c_off_branch_pf + cfg.c_common_route_pf;
    result.c_n77_pf =
        result.fix.mim_rf_cap_pf + result.n77.c_on_eff_pf +
        result.n78.c_off_branch_pf + cfg.c_common_route_pf;
    result.c_n78_pf =
        result.fix.mim_rf_cap_pf + result.n77.c_off_branch_pf +
        result.n78.c_on_eff_pf + cfg.c_common_route_pf;
    double max_mode_error = 0.0;
    for (double freq_ghz : cfg.report_freqs_ghz) {
        const double freq_hz = freq_ghz * 1.0e9;
        const double n77_on =
            result.n77.solve_failed || result.n77.omitted
                ? 0.0
                : prelayout_c_on_eff_pf_at_hz(
                      cfg, result.n77.mim_rf_cap_pf, result.n77.mos_total_width_um,
                      result.n77.mim_multiplier, freq_hz);
        const double n78_on =
            result.n78.solve_failed || result.n78.omitted
                ? 0.0
                : prelayout_c_on_eff_pf_at_hz(
                      cfg, result.n78.mim_rf_cap_pf, result.n78.mos_total_width_um,
                      result.n78.mim_multiplier, freq_hz);
        const double ceff_n79 =
            result.fix.mim_rf_cap_pf + result.n77.c_off_branch_pf +
            result.n78.c_off_branch_pf + cfg.c_common_route_pf;
        const double ceff_n77 =
            result.fix.mim_rf_cap_pf + n77_on +
            result.n78.c_off_branch_pf + cfg.c_common_route_pf;
        const double ceff_n78 =
            result.fix.mim_rf_cap_pf + result.n77.c_off_branch_pf +
            n78_on + cfg.c_common_route_pf;
        max_mode_error = std::max({
            max_mode_error,
            std::abs(ceff_n79 - targets.n79_pf),
            std::abs(ceff_n77 - targets.n77_pf),
            std::abs(ceff_n78 - targets.n78_pf),
        });
    }
    result.max_mode_error_pf = max_mode_error;
    const bool exact_mode_match = max_mode_error <= 0.001;
    const bool tiny_omission_match =
        (result.n77.omitted || result.n78.omitted) &&
        max_mode_error <= args.min_branch_pf + 1.0e-9;
    const bool branch_solve_failed = result.n77.solve_failed || result.n78.solve_failed;
    (void)exact_mode_match;
    (void)tiny_omission_match;
    if (result.n77.c_off_branch_pf + result.n78.c_off_branch_pf >
        targets.n79_pf - cfg.c_common_route_pf) {
        push_unique(result.warnings,
                    "C77_off + C78_off is larger than the available N79 fixed-capacitance budget.");
    }
    const double n79_budget = std::max(targets.n79_pf - cfg.c_common_route_pf, 1.0e-12);
    const double off_fraction =
        (result.n77.c_off_branch_pf + result.n78.c_off_branch_pf) / n79_budget;
    if (off_fraction > 0.5 && result.fixed_cap_valid) {
        std::ostringstream msg;
        msg << "N79 base capacitance is parasitic-dominated: disabled branch residuals are "
            << std::fixed << std::setprecision(1) << 100.0 * off_fraction
            << "% of the N79 fixed-capacitance budget.";
        push_unique(result.warnings, msg.str());
    }
    if (result.n77.w_req_stack_only_um > 5000.0 ||
        result.n78.w_req_stack_only_um > 5000.0) {
        push_unique(result.warnings,
                    "Finite-R 1%/10fF criteria require multi-millimeter total switch width in at least one branch.");
    }
    if (result.n77.mos_total_width_um > 5000.0 ||
        result.n78.mos_total_width_um > 5000.0) {
        push_unique(result.warnings,
                    "Selected switch width exceeds 5000 um in at least one branch; segmentation is recommended before layout.");
    }
    if (result.n77.c_mid_diffusion_pf > 0.2 * std::max(result.n77.delta_target_pf, 0.1) ||
        result.n78.c_mid_diffusion_pf > 0.2 * std::max(result.n78.delta_target_pf, 0.1)) {
        push_unique(result.warnings,
                    "Switch midpoint diffusion capacitance is large relative to the target delta; include the full stack network in Spectre, not only the scalar capacitance correction.");
    }
    if (cfg.bias_resistance_ohm < result.n77.r_bias_min_ohm ||
        cfg.bias_resistance_ohm < result.n78.r_bias_min_ohm) {
        push_unique(result.warnings,
                    "Midpoint bias resistor is below the 100/(omega*Cmid) RF-open estimate.");
    }

    const auto add_branch_warnings = [&](const PrelayoutBranch& b) {
        if (b.omitted || b.solve_failed) {
            return;
        }
        if (b.delta_target_pf > 1.0) {
            push_unique(result.warnings,
                        b.branch + ": Large switched-MIM increment. A single switched branch may be unrealistic. Consider segmentation, parallel unit branches, or re-optimization after PDK extraction.");
        }
        if (b.delta_target_pf > 2.0) {
            push_unique(result.warnings,
                        b.branch + ": delta exceeds 2 pF; segmentation is strongly recommended.");
        }
        if (b.mos_multiplier > 100) {
            push_unique(result.warnings,
                        b.branch + ": required MOS multiplier is very large.");
        }
        if (b.mos_total_width_um > 5000.0) {
            push_unique(result.warnings,
                        b.branch + ": total NMOS width exceeds 5000 um.");
        }
        if (b.ron_stack_ohm > cfg.warn_ron_stack_ohm) {
            push_unique(result.warnings,
                        b.branch + ": Ron_stack is above the warning threshold.");
        }
        if (b.min_q < cfg.warn_branch_q) {
            std::ostringstream msg;
            msg << b.branch << ": branch Q is below warn_branch_q at "
                << std::fixed << std::setprecision(3) << b.min_q_freq_ghz
                << " GHz.";
            push_unique(result.warnings, msg.str());
        }
        if (b.finite_r_loss_pf > cfg.eps_abs_pf ||
            b.finite_r_rel_loss > cfg.eps_rel) {
            push_unique(result.warnings,
                        b.branch + ": finite-R loss exceeds one of the formula-level warning limits.");
        }
        if (b.c_off_branch_pf > 0.05 * std::max(targets.n79_pf, 1.0e-12)) {
            std::ostringstream msg;
            msg << b.branch << ": off residual exceeds 5% of CN79.";
            push_unique(result.warnings, msg.str());
        }
    };
    add_branch_warnings(result.n77);
    add_branch_warnings(result.n78);

    if (result.c_fix_target_rf_pf <= 0.0) {
        push_unique(result.warnings,
                    "Cfix_required is zero or negative; disabled-branch residual capacitance consumes the N79 budget.");
    } else if (result.c_fix_target_rf_pf < 0.05 * std::max(targets.n79_pf, 1.0e-12)) {
        push_unique(result.warnings,
                    "Cfix_required is close to zero; verify off-state capacitance with PDK extraction.");
    }

    const auto finite_r_bad = [&](const PrelayoutBranch& b) {
        return !b.omitted && !b.solve_failed &&
               b.finite_r_loss_pf > cfg.eps_abs_pf + 1.0e-12 &&
               b.finite_r_rel_loss > cfg.eps_rel + 1.0e-12;
    };
    const auto segmentation_needed = [&](const PrelayoutBranch& b) {
        return !b.omitted && !b.solve_failed &&
               (b.delta_target_pf > 2.0 ||
                b.w_req_stack_only_um > 5000.0 ||
                b.mos_total_width_um > 5000.0 ||
                b.min_q < cfg.min_branch_q);
    };
    const bool finite_r_fail = finite_r_bad(result.n77) || finite_r_bad(result.n78);
    const bool needs_segmentation =
        segmentation_needed(result.n77) || segmentation_needed(result.n78);

    if (branch_solve_failed) {
        result.verdict = "FAIL_BRANCH_EQUATION";
    } else if (result.c_fix_target_rf_pf <= 0.0 || !result.fixed_cap_valid) {
        result.verdict = "FAIL_NEGATIVE_CFIX";
    } else if (needs_segmentation) {
        result.verdict = "NEEDS_SEGMENTATION";
    } else if (finite_r_fail) {
        result.verdict = "FAIL_FINITE_R";
    } else if (result.max_mode_error_pf <= 0.001 && result.warnings.empty()) {
        result.verdict = "PASS_FORMULA_ONLY";
    } else if (result.max_mode_error_pf <= 0.010) {
        result.verdict = result.warnings.empty() ? "PASS_FORMULA_ONLY" : "PASS_WITH_WARNINGS";
    } else {
        result.verdict = "FAIL_BRANCH_EQUATION";
    }
    return result;
}

std::string int_vector_text(const std::vector<int>& values) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << values[i];
    }
    out << "]";
    return out.str();
}

std::vector<int> distribute_mos_multiplier(int mos_multiplier, int slices) {
    std::vector<int> values;
    if (slices <= 0) {
        return values;
    }
    const int base = mos_multiplier / slices;
    const int rem = mos_multiplier % slices;
    for (int i = 0; i < slices; ++i) {
        values.push_back(base + (i < rem ? 1 : 0));
    }
    return values;
}

[[maybe_unused]] void print_prelayout_branch(const PrelayoutBranch& b, const PrelayoutConfig& cfg) {
    std::cout << "\n" << std::string(88, '-') << "\n";
    std::cout << "BRANCH: " << b.branch << "\n";
    std::cout << std::string(88, '-') << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Delta target                         = " << b.delta_target_pf << " pF\n";
    if (b.omitted) {
        std::cout << "Status                               = OMITTED\n";
        std::cout << "Reason                               = delta is zero or below practical switched-MIM floor\n";
        return;
    }
    if (b.solve_failed) {
        std::cout << "Status                               = FAILED\n";
        std::cout << "Reason                               = " << b.failure_reason << "\n";
        return;
    }
    std::cout << "\nMOS PCell:\n";
    std::cout << "  model                              = n33_ckt_rf\n";
    std::cout << "  W                                  = " << std::setprecision(3)
              << b.mos_w_unit_um << " um\n";
    std::cout << "  L                                  = " << std::setprecision(0)
              << b.mos_l_nm << " nm\n";
    std::cout << "  fingers                            = " << b.mos_fingers << "\n";
    std::cout << "  devices per branch                 = "
              << b.mos_stack_devices_per_branch << "\n";
    std::cout << "  multiplier                         = " << b.mos_multiplier << "\n";
    std::cout << "  bodytie                            = "
              << (b.bodytie ? "true" : "false") << "\n";
    std::cout << "  total width per NMOS               = " << std::setprecision(3)
              << b.mos_total_width_um << " um\n";
    std::cout << "  Ron_stack                          = " << std::setprecision(6)
              << b.ron_stack_ohm << " ohm\n";
    std::cout << "\nMIM2 PCell:\n";
    std::cout << "  W                                  = " << std::setprecision(3)
              << b.mim_unit_w_um << " um\n";
    std::cout << "  L                                  = " << b.mim_unit_l_um << " um\n";
    std::cout << "  multiplier                         = " << b.mim_multiplier << "\n";
    std::cout << "  unit area C                         = " << std::setprecision(6)
              << b.mim_unit_area_cap_pf << " pF\n";
    std::cout << "  total area C                        = " << b.mim_total_area_cap_pf
              << " pF\n";
    std::cout << "  parasitic C                         = " << b.mim_parasitic_cap_pf
              << " pF\n";
    std::cout << "  RF C                                = " << b.mim_rf_cap_pf << " pF\n";
    std::cout << "\nRF details:\n";
    std::cout << "  R_mim_esr_at_sizing_freq            = " << b.r_mim_esr_ohm
              << " ohm\n";
    std::cout << "  R_mim_esr_sweep:\n";
    std::cout << "    Freq_GHz                          ESR_ohm\n";
    for (double freq_ghz : cfg.report_freqs_ghz) {
        std::cout << "    " << std::setw(8) << std::setprecision(3) << freq_ghz
                  << "                          "
                  << std::setprecision(6)
                  << prelayout_r_mim_esr_ohm_at_hz(
                         cfg, b.mim_rf_cap_pf, freq_ghz * 1.0e9)
                  << "\n";
    }
    std::cout << "  R_mim_route                         = " << b.r_mim_route_ohm
              << " ohm\n";
    std::cout << "  R_branch_on                         = " << b.r_branch_on_ohm
              << " ohm\n";
    std::cout << "  omega*R*C_MIM                       = " << b.omega_r_c
              << "\n";
    std::cout << "  Ron_max_rel                         = " << b.ron_max_rel_ohm
              << " ohm\n";
    std::cout << "  Ron_max_abs                         = " << b.ron_max_abs_ohm
              << " ohm\n";
    std::cout << "  W_required_stack_only               = " << b.w_req_stack_only_um
              << " um\n";
    std::cout << "  Csw_off_stack                       = " << b.c_sw_off_stack_pf
              << " pF\n";
    std::cout << "  C_on_extra                          = " << b.c_on_extra_pf
              << " pF\n";
    std::cout << "  C_off_extra_direct                  = " << b.c_off_extra_direct_pf
              << " pF\n";
    std::cout << "  Cmid_diffusion                      = " << b.c_mid_diffusion_pf
              << " pF\n";
    std::cout << "  Cmid_effective_model                = " << b.c_mid_pf
              << " pF\n";
    std::cout << "  C_on_eff                            = " << b.c_on_eff_pf << " pF\n";
    std::cout << "  G_on                                = " << b.g_on_s << " S\n";
    std::cout << "  C_off_branch                        = " << b.c_off_branch_pf
              << " pF\n";
    std::cout << "  Rbias_min_100_over_wCmid            = " << b.r_bias_min_ohm
              << " ohm\n";
    std::cout << "  finite-R loss                       = " << b.finite_r_loss_pf
              << " pF\n";
    std::cout << "  finite-R relative loss              = " << std::setprecision(4)
              << (100.0 * b.finite_r_rel_loss) << " %\n";
}

[[maybe_unused]] void print_mode_line(const std::string& name, double value, double target) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << name << " effective                     = " << value << " pF, error=";
    std::cout << std::scientific << std::showpos << std::setprecision(3)
              << (value - target) << std::noshowpos << std::fixed
              << std::setprecision(6) << " pF\n";
}

[[maybe_unused]] void print_prelayout_slice_advice(const PrelayoutResult& result) {
    std::cout << "\n" << std::string(88, '=') << "\n";
    std::cout << "SLICED LAYOUT ADVICE\n";
    std::cout << std::string(88, '=') << "\n";
    const std::array<std::tuple<std::string, const PrelayoutBranch*, std::string>, 2> items = {{
        {result.cell + "_N77", &result.n77, "N77_ENABLE"},
        {result.cell + "_N78", &result.n78, "N78_ENABLE"},
    }};
    for (const auto& item : items) {
        const std::string& name = std::get<0>(item);
        const PrelayoutBranch& b = *std::get<1>(item);
        const std::string& gate = std::get<2>(item);
        std::cout << "\n" << name << ":\n";
        if (b.omitted) {
            std::cout << "  status                             = OMITTED\n";
            std::cout << "  gate                               = " << gate
                      << " unused for this cell\n";
            continue;
        }
        if (b.solve_failed) {
            std::cout << "  status                             = FAILED\n";
            std::cout << "  action                             = segment branch or relax sizing constraints\n";
            continue;
        }
        std::cout << "  MIM slices                         = " << b.mim_multiplier << "\n";
        std::cout << "  each MIM slice                      = " << std::fixed
                  << std::setprecision(3) << b.mim_unit_w_um << " um x "
                  << b.mim_unit_l_um << " um\n";
        std::cout << "  distribute MOS multiplier           = "
                  << int_vector_text(distribute_mos_multiplier(
                         b.mos_multiplier, b.mim_multiplier))
                  << "\n";
        std::cout << "  gate                                = " << gate << "\n";
        std::cout << "  slice topology                      = MIM unit + "
                  << result.cfg.mos_stack_devices_per_branch
                  << "-series NMOS stack\n";
    }
}

void print_prelayout_report(const PrelayoutResult& result) {
    const PrelayoutConfig& cfg = result.cfg;
    const double delta_n77 = result.targets.n77_pf - result.targets.n79_pf;
    const double delta_n78 = result.targets.n78_pf - result.targets.n79_pf;
    const auto branch_status = [](const PrelayoutBranch& b) {
        if (b.omitted) return std::string("OMIT");
        if (b.solve_failed) return std::string("FAILED");
        return std::string("ACTIVE");
    };
    const auto print_branch = [&](const std::string& title, const PrelayoutBranch& b) {
        std::cout << "\n" << title << "\n";
        std::cout << "  status                         = " << branch_status(b) << "\n";
        std::cout << "  branch target delta             = " << b.delta_target_pf << " pF\n";
        if (b.solve_failed) {
            std::cout << "  failure reason                  = " << b.failure_reason << "\n";
            return;
        }
        if (b.omitted) {
            std::cout << "  note                            = below practical switched-MIM floor\n";
            return;
        }
        std::cout << "  drawn MIM capacitance           = " << b.mim_rf_cap_pf << " pF\n";
        std::cout << "  MOS model                       = n33_ckt_rf\n";
        std::cout << "  NMOS W per finger               = " << b.mos_w_unit_um << " um\n";
        std::cout << "  NMOS L                          = " << b.mos_l_nm << " nm\n";
        std::cout << "  fingers                         = " << b.mos_fingers << "\n";
        std::cout << "  multiplier                      = " << b.mos_multiplier << "\n";
        std::cout << "  total width per NMOS            = " << b.mos_total_width_um << " um\n";
        std::cout << "  stack Ron                       = " << b.ron_stack_ohm << " ohm\n";
        std::cout << "  estimated off capacitance       = " << b.c_off_branch_pf << " pF\n";
        std::cout << "  estimated on effective C        = " << b.c_on_eff_pf << " pF @ "
                  << cfg.freq_hz / 1.0e9 << " GHz\n";
        std::cout << "  estimated min Q                 = " << b.min_q << " @ "
                  << b.min_q_freq_ghz << " GHz\n";
        std::cout << "  finite-R loss                   = " << b.finite_r_loss_pf
                  << " pF (" << 100.0 * b.finite_r_rel_loss << "%)\n";
    };

    std::cout << std::string(88, '=') << "\n";
    std::cout << result.cell << " SINGLE-CELL SWITCHED-CAP SIZING\n";
    std::cout << std::string(88, '=') << "\n";
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\nA. INPUT TARGET SUMMARY\n";
    std::cout << "  Cell name                       = " << result.cell << "\n";
    std::cout << "  Parasitic model                 = " << cfg.parasitic_model << "\n";
    std::cout << "  CN77 target                     = " << result.targets.n77_pf << " pF\n";
    std::cout << "  CN78 target                     = " << result.targets.n78_pf << " pF\n";
    std::cout << "  CN79 target                     = " << result.targets.n79_pf << " pF\n";
    std::cout << "  Cfix target before off-correction = " << result.targets.n79_pf << " pF\n";
    std::cout << "  Solved Cfix_eff                 = " << result.c_fix_target_rf_pf << " pF\n";
    std::cout << "  Delta_N77 target                = " << delta_n77 << " pF\n";
    std::cout << "  Delta_N78 target                = " << delta_n78 << " pF\n";
    std::cout << "  Frequency sample points         = ";
    for (std::size_t i = 0; i < cfg.report_freqs_ghz.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << cfg.report_freqs_ghz[i];
    }
    std::cout << " GHz\n";

    std::cout << "\nB. FIXED BRANCH\n";
    std::cout << "  MIM model                       = mim2_rf_2mask\n";
    std::cout << "  status                          = "
              << (result.fixed_cap_valid ? "ACTIVE" : "INVALID") << "\n";
    std::cout << "  drawn capacitance               = " << result.fix.mim_rf_cap_pf << " pF\n";
    std::cout << "  estimated capacitance           = "
              << result.fix.mim_rf_cap_pf * 1000.0 << " fF\n";
    std::cout << "  estimated W/L                   = " << result.fix.mim_unit_w_um
              << " um / " << result.fix.mim_unit_l_um << " um\n";
    std::cout << "  multiplier                      = " << result.fix.mim_multiplier << "\n";

    print_branch("C. N77 SWITCHED BRANCH", result.n77);
    print_branch("D. N78 SWITCHED BRANCH", result.n78);

    std::cout << "\nE. MODE CAPACITANCE PREDICTION\n";
    std::cout << "  freq_GHz     Ceff_N77_pF   Ceff_N78_pF   Ceff_N79_pF"
              << "   error_N77_fF   error_N78_fF   error_N79_fF\n";
    for (double freq_ghz : cfg.report_freqs_ghz) {
        const double freq_hz = freq_ghz * 1.0e9;
        const double c77_on =
            result.n77.solve_failed || result.n77.omitted
                ? 0.0
                : prelayout_c_on_eff_pf_at_hz(
                      cfg, result.n77.mim_rf_cap_pf, result.n77.mos_total_width_um,
                      result.n77.mim_multiplier, freq_hz);
        const double c78_on =
            result.n78.solve_failed || result.n78.omitted
                ? 0.0
                : prelayout_c_on_eff_pf_at_hz(
                      cfg, result.n78.mim_rf_cap_pf, result.n78.mos_total_width_um,
                      result.n78.mim_multiplier, freq_hz);
        const double ceff_n79 =
            result.fix.mim_rf_cap_pf + result.n77.c_off_branch_pf +
            result.n78.c_off_branch_pf + cfg.c_common_route_pf;
        const double ceff_n77 =
            result.fix.mim_rf_cap_pf + c77_on +
            result.n78.c_off_branch_pf + cfg.c_common_route_pf;
        const double ceff_n78 =
            result.fix.mim_rf_cap_pf + result.n77.c_off_branch_pf +
            c78_on + cfg.c_common_route_pf;
        std::cout << "  " << std::setw(8) << freq_ghz
                  << "     " << std::setw(11) << ceff_n77
                  << "   " << std::setw(11) << ceff_n78
                  << "   " << std::setw(11) << ceff_n79
                  << "   " << std::setw(12) << (ceff_n77 - result.targets.n77_pf) * 1000.0
                  << "   " << std::setw(12) << (ceff_n78 - result.targets.n78_pf) * 1000.0
                  << "   " << std::setw(12) << (ceff_n79 - result.targets.n79_pf) * 1000.0
                  << "\n";
    }
    std::cout << "  max abs error                  = " << result.max_mode_error_pf * 1000.0
              << " fF\n";

    std::cout << "\nCADENCE IMPLEMENTATION TABLE\n";
    std::cout << "  instance             device/model        W_um       L_nm     fingers   m       RF_C_pF/status\n";
    std::cout << "  " << result.cell << "_FIX"
              << "             mim2_rf_2mask      " << result.fix.mim_unit_w_um
              << "     " << result.fix.mim_unit_l_um
              << "     -         " << result.fix.mim_multiplier
              << "       " << result.fix.mim_rf_cap_pf << "\n";
    const auto print_impl = [&](const std::string& suffix, const PrelayoutBranch& b) {
        if (b.solve_failed || b.omitted) {
            std::cout << "  " << result.cell << "_" << suffix
                      << "             n33_ckt_rf         -          -        -         -       "
                      << branch_status(b) << "\n";
            return;
        }
        std::cout << "  " << result.cell << "_" << suffix
                  << "             n33_ckt_rf         " << b.mos_w_unit_um
                  << "      " << b.mos_l_nm
                  << "      " << b.mos_fingers
                  << "        " << b.mos_multiplier
                  << "       " << b.mim_rf_cap_pf << "\n";
    };
    print_impl("N77", result.n77);
    print_impl("N78", result.n78);

    std::cout << "\nF. FINAL VERDICT\n";
    std::cout << "  " << result.verdict << "\n";
    if (!result.warnings.empty()) {
        std::cout << "\nWarnings:\n";
        for (const auto& warning : result.warnings) {
            std::cout << "  - " << warning << "\n";
        }
    }

    std::cout << "\nModel honesty note:\n";
    std::cout << "  These parameters are conservative pre-layout estimates for n33_ckt_rf and mim2_rf_2mask.\n";
    std::cout << "  Replace them with SMIC N130 Spectre/PDK sweep data before final sizing.\n";
    std::cout << "  This program does formula-level capacitance matching, finite-R/Q warnings,\n";
    std::cout << "  off-state residual estimation, and first-pass MIM/MOS sizing only.\n";
    std::cout << "  It does not perform S-parameter, S11/S21, inductor-Q, substrate-loss, EM routing,\n";
    std::cout << "  transmission-zero movement, or post-layout extraction verification.\n";
}

void print_prelayout_branch_json(
    const std::string& key,
    const PrelayoutBranch& b,
    bool trailing_comma) {
    std::cout << "    \"" << key << "\": {"
              << "\"omitted\": " << (b.omitted ? "true" : "false")
              << ", \"solve_failed\": " << (b.solve_failed ? "true" : "false")
              << ", \"failure_reason\": \"" << json_escape(b.failure_reason) << "\""
              << ", \"delta_target_pf\": " << json_number(b.delta_target_pf)
              << ", \"mos_multiplier\": " << b.mos_multiplier
              << ", \"mos_stack_devices_per_branch\": "
              << b.mos_stack_devices_per_branch
              << ", \"bodytie\": " << (b.bodytie ? "true" : "false")
              << ", \"mos_total_width_um\": " << json_number(b.mos_total_width_um)
              << ", \"ron_stack_ohm\": " << json_number(b.ron_stack_ohm)
              << ", \"omega_r_c\": " << json_number(b.omega_r_c)
              << ", \"ron_max_rel_ohm\": " << json_number(b.ron_max_rel_ohm)
              << ", \"ron_max_abs_ohm\": " << json_number(b.ron_max_abs_ohm)
              << ", \"w_req_stack_only_um\": " << json_number(b.w_req_stack_only_um)
              << ", \"mim_multiplier\": " << b.mim_multiplier
              << ", \"mim_unit_area_cap_pf\": " << json_number(b.mim_unit_area_cap_pf)
              << ", \"mim_rf_cap_pf\": " << json_number(b.mim_rf_cap_pf)
              << ", \"mim_unit_w_um\": " << json_number(b.mim_unit_w_um)
              << ", \"c_on_eff_pf\": " << json_number(b.c_on_eff_pf)
              << ", \"g_on_s\": " << json_number(b.g_on_s)
              << ", \"q_on\": " << json_number(b.q_on)
              << ", \"min_q\": " << json_number(b.min_q)
              << ", \"min_q_freq_ghz\": " << json_number(b.min_q_freq_ghz)
              << ", \"c_off_branch_pf\": " << json_number(b.c_off_branch_pf)
              << ", \"c_sw_off_stack_pf\": " << json_number(b.c_sw_off_stack_pf)
              << ", \"c_mid_diffusion_pf\": " << json_number(b.c_mid_diffusion_pf)
              << ", \"r_bias_min_ohm\": " << json_number(b.r_bias_min_ohm)
              << ", \"finite_r_loss_pf\": " << json_number(b.finite_r_loss_pf)
              << ", \"finite_r_rel_loss\": " << json_number(b.finite_r_rel_loss)
              << "}" << (trailing_comma ? "," : "") << "\n";
}

void print_prelayout_json(const PrelayoutResult& result) {
    std::cout << std::setprecision(12);
    std::cout << "{\n";
    std::cout << "  \"cell\": \"" << json_escape(result.cell) << "\",\n";
    std::cout << "  \"type\": \"n79_base_prelayout\",\n";
    std::cout << "  \"verdict\": \"" << json_escape(result.verdict) << "\",\n";
    std::cout << "  \"frequency_hz\": " << result.cfg.freq_hz << ",\n";
    std::cout << "  \"process_preset\": {\"foundry\": \"SMIC\", \"node\": \"N130\", "
              << "\"mos_model\": \"n33_ckt_rf\", \"mim_model\": \"mim2_rf_2mask\", "
              << "\"parasitic_model\": \"" << json_escape(result.cfg.parasitic_model) << "\", "
              << "\"kr_stack_ohm_um\": " << result.cfg.kr_stack
              << ", \"kr_stack_effective_ohm_um\": "
              << prelayout_kr_stack_effective_ohm_um(result.cfg)
              << ", \"mos_stack_devices_per_branch\": "
              << result.cfg.mos_stack_devices_per_branch
              << ", \"bodytie\": " << (result.cfg.bodytie ? "true" : "false")
              << ", \"body_effect_ron_factor\": "
              << result.cfg.body_effect_ron_factor
              << ", \"coff_stack_ff_per_um\": " << result.cfg.coff_per_um_ff
              << ", \"c_on_extra_ff_per_um\": " << result.cfg.c_on_extra_per_um_ff
              << ", \"mim_density_effective_pf_per_um2\": "
              << result.cfg.mim_density_effective_pf_per_um2() << "},\n";
    std::cout << "  \"targets_pf\": {\"N77\": " << result.targets.n77_pf
              << ", \"N78\": " << result.targets.n78_pf
              << ", \"N79\": " << result.targets.n79_pf << "},\n";
    std::cout << "  \"branches\": {\n";
    print_prelayout_branch_json("N77", result.n77, true);
    print_prelayout_branch_json("N78", result.n78, false);
    std::cout << "  },\n";
    std::cout << "  \"fixed\": {"
              << "\"valid\": " << (result.fixed_cap_valid ? "true" : "false")
              << ", \"required_rf_cap_pf\": " << result.c_fix_target_rf_pf
              << ", "
              << "\"mim_multiplier\": " << result.fix.mim_multiplier
              << ", \"mim_unit_area_cap_pf\": " << result.fix.mim_unit_area_cap_pf
              << ", \"mim_rf_cap_pf\": " << result.fix.mim_rf_cap_pf
              << ", \"mim_unit_w_um\": " << result.fix.mim_unit_w_um
              << "},\n";
    std::cout << "  \"mode_effective_pf\": {\"N77\": " << result.c_n77_pf
              << ", \"N78\": " << result.c_n78_pf
              << ", \"N79\": " << result.c_n79_pf << "},\n";
    std::cout << "  \"max_mode_error_fF\": "
              << result.max_mode_error_pf * 1000.0 << ",\n";
    std::cout << "  \"warnings\": [";
    for (std::size_t i = 0; i < result.warnings.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << "\"" << json_escape(result.warnings[i]) << "\"";
    }
    std::cout << "]\n";
    std::cout << "}\n";
}

void print_prelayout_csv(const PrelayoutResult& result) {
    std::cout << "cell,verdict,fixed_valid,CN77_pf,CN78_pf,CN79_pf,"
                 "N77_mos_m,N77_mim_m,N77_mim_unit_w_um,N77_rf_cap_pf,"
                 "N78_mos_m,N78_mim_m,N78_mim_unit_w_um,N78_rf_cap_pf,"
                 "FIX_mim_m,FIX_mim_unit_w_um,FIX_rf_cap_pf,"
                 "Ceff_N77_pf,Ceff_N78_pf,Ceff_N79_pf\n";
    std::cout << std::setprecision(12)
              << csv_escape(result.cell) << ","
              << csv_escape(result.verdict) << ","
              << (result.fixed_cap_valid ? "true" : "false") << ","
              << result.targets.n77_pf << ","
              << result.targets.n78_pf << ","
              << result.targets.n79_pf << ","
              << result.n77.mos_multiplier << ","
              << result.n77.mim_multiplier << ","
              << result.n77.mim_unit_w_um << ","
              << result.n77.mim_rf_cap_pf << ","
              << result.n78.mos_multiplier << ","
              << result.n78.mim_multiplier << ","
              << result.n78.mim_unit_w_um << ","
              << result.n78.mim_rf_cap_pf << ","
              << result.fix.mim_multiplier << ","
              << result.fix.mim_unit_w_um << ","
              << result.fix.mim_rf_cap_pf << ","
              << result.c_n77_pf << ","
              << result.c_n78_pf << ","
              << result.c_n79_pf << "\n";
}

bool prelayout_pass_like(const PrelayoutResult& result) {
    return result.verdict == "PASS_FORMULA_ONLY" ||
           result.verdict == "PASS_WITH_WARNINGS";
}

PrelayoutResult solve_prelayout_variant(
    CommandLine args,
    const Targets& targets,
    const std::string& parasitic_model,
    double coff_scale,
    double ron_scale) {
    args.parasitic_model = parasitic_model;
    PrelayoutConfig cfg = make_prelayout_config_from_args(args);
    cfg.coff_per_um_ff *= coff_scale;
    cfg.kr_stack *= ron_scale;
    return solve_prelayout_result(args, targets, cfg);
}

void apply_conservative_crosscheck_verdict(
    const CommandLine& args,
    const Targets& targets,
    PrelayoutResult& result) {
    if (result.cfg.parasitic_model != "conservative_scalar" ||
        prelayout_pass_like(result)) {
        return;
    }
    const std::array<std::string, 3> diagnostic_models = {
        "no_midpoint", "no_on_extra", "simple"};
    for (const auto& model : diagnostic_models) {
        try {
            const PrelayoutResult variant =
                solve_prelayout_variant(args, targets, model, 1.0, 1.0);
            if (prelayout_pass_like(variant)) {
                result.verdict = "PHYSICALLY_SUSPICIOUS_NEEDS_PDK_CALIBRATION";
                push_unique(result.warnings,
                            "Conservative scalar parasitics fail, but a reduced parasitic model passes. Treat this as PDK-calibration-sensitive, not a final hard failure.");
                return;
            }
        } catch (const std::exception&) {
        }
    }
}

void print_parasitic_sweep(
    const CommandLine& args,
    const Targets& targets) {
    struct SweepCase {
        std::string label;
        std::string model;
        double coff_scale;
        double ron_scale;
    };
    const std::array<SweepCase, 8> cases = {{
        {"conservative_scalar", "conservative_scalar", 1.0, 1.0},
        {"no_midpoint", "no_midpoint", 1.0, 1.0},
        {"no_on_extra", "no_on_extra", 1.0, 1.0},
        {"simple", "simple", 1.0, 1.0},
        {"lower Coff estimate", "conservative_scalar", 0.75, 1.0},
        {"higher Coff estimate", "conservative_scalar", 1.25, 1.0},
        {"lower Ron estimate", "conservative_scalar", 1.0, 0.85},
        {"higher Ron estimate", "conservative_scalar", 1.0, 1.25},
    }};

    std::cout << "\n" << std::string(88, '=') << "\n";
    std::cout << "ONE-CELL PARASITIC SENSITIVITY SWEEP\n";
    std::cout << std::string(88, '=') << "\n";
    std::cout << "This sweep is diagnostic only. It is not PDK-accurate and does not replace Spectre/PEX.\n\n";
    std::cout << "  case                    verdict                                      Cfix_pF"
              << "    max_err_fF   off_total_pF   min_Q     W_N77_um   W_N78_um   diagnosis\n";
    for (const auto& item : cases) {
        try {
            PrelayoutResult result =
                solve_prelayout_variant(args, targets, item.model, item.coff_scale, item.ron_scale);
            const double off_total = result.n77.c_off_branch_pf + result.n78.c_off_branch_pf;
            const double min_q = std::min(result.n77.min_q, result.n78.min_q);
            std::string diagnosis = "formula/parasitic balance";
            if (result.c_fix_target_rf_pf <= 0.0) {
                diagnosis = "negative fixed-cap budget";
            } else if (result.n77.solve_failed || result.n78.solve_failed) {
                diagnosis = "branch equation";
            } else if (result.n77.w_req_stack_only_um > 5000.0 ||
                       result.n78.w_req_stack_only_um > 5000.0 ||
                       result.n77.mos_total_width_um > 5000.0 ||
                       result.n78.mos_total_width_um > 5000.0) {
                diagnosis = "MOS width requirement";
            } else if (min_q < result.cfg.warn_branch_q) {
                diagnosis = "finite Ron / Q";
            } else if (off_total > 0.05 * std::max(targets.n79_pf, 1.0e-12)) {
                diagnosis = "off-state residual capacitance";
            } else if (item.model == "no_midpoint") {
                diagnosis = "midpoint capacitance sensitivity";
            } else if (item.model == "no_on_extra") {
                diagnosis = "on-state parasitic sensitivity";
            }
            std::cout << "  " << std::left << std::setw(23) << item.label
                      << " " << std::left << std::setw(44) << result.verdict
                      << std::right << std::setw(9) << std::fixed << std::setprecision(4)
                      << result.c_fix_target_rf_pf
                      << "   " << std::setw(10) << std::setprecision(3)
                      << result.max_mode_error_pf * 1000.0
                      << "   " << std::setw(11) << std::setprecision(4) << off_total
                      << "   " << std::setw(7) << std::setprecision(2) << min_q
                      << "   " << std::setw(8) << std::setprecision(1)
                      << result.n77.mos_total_width_um
                      << "   " << std::setw(8) << std::setprecision(1)
                      << result.n78.mos_total_width_um
                      << "   " << diagnosis << "\n";
        } catch (const std::exception& exc) {
            std::cout << "  " << std::left << std::setw(23) << item.label
                      << " " << std::left << std::setw(44) << "ERROR"
                      << "   " << exc.what() << "\n";
        }
    }
}

bool nearly_equal(double a, double b, double rel_tol = 1.0e-6, double abs_tol = 1.0e-9) {
    return std::abs(a - b) <= std::max(abs_tol, rel_tol * std::max(std::abs(a), std::abs(b)));
}

std::string optional_pf_text(double value) {
    if (!supplied(value)) {
        return "not supplied";
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(6) << value << " pF";
    return out.str();
}

std::string determine_pcell_verdict(
    double c_drawn_pf,
    double target_pf,
    const MOSSwitchParams& mos,
    const std::vector<double>& freqs_ghz,
    const RFModelParams& params,
    bool calibrated_from_json) {
    if (c_drawn_pf <= 0.0 || mos.total_width_per_transistor_um() <= 0.0) {
        return "FAILED";
    }
    if (supplied(target_pf)) {
        const double cap_err_pct =
            std::abs(c_drawn_pf - target_pf) / std::max(std::abs(target_pf), 0.1) * 100.0;
        if (cap_err_pct > 2.0) {
            return "FAILED";
        }
    }

    double min_q = std::numeric_limits<double>::infinity();
    for (double freq_ghz : freqs_ghz) {
        min_q = std::min(min_q, switched_branch_on_eff(
            c_drawn_pf, freq_ghz, mos, params, 0.0).q_eff);
    }
    const double ron = ron_stack_ohm(mos, params);
    const bool low_q_fail = min_q < params.min_branch_q;
    const bool low_q_warn = min_q < params.warn_branch_q;
    const bool ron_fail = ron > params.max_ron_stack_ohm;
    const bool ron_warn = ron > params.warn_ron_stack_ohm;

    if (c_drawn_pf > 1.0 && (low_q_fail || ron_fail)) {
        return "NEEDS_SEGMENTATION";
    }
    if (low_q_fail || low_q_warn || ron_fail || ron_warn) {
        return "PHYSICALLY_SUSPICIOUS";
    }
    if (!calibrated_from_json || !has_mim_loss_input(params)) {
        return "FIRST_PASS_ONLY_NEEDS_PDK_VERIFICATION";
    }
    return "FIRST_PASS_OK";
}

// === Paper-target branch audit (RF_BPF.docx Section III-B / III-E, Table I(b)) ===
//
// For every implemented (cell, branch, delta_pf) pair from Table I(b) and every
// frequency in freqs_ghz, computes:
//   x          = omega * Ron_stack * delta_pf
//   dC_loss_pF = delta_pf * x^2 / (1 + x^2)         // paper's finite-Ron loss
//   branch_Q   = 1 / x                              // when x > 0
// Paper threshold for the on-state first-order approximation is x < 0.1.
// We declare HARD_FAIL when:
//   x >= 1.0,  OR  dC_loss / delta_pf >= 0.10,  OR  branch_Q < min_branch_q.
// We declare WARN when the paper's 0.1 threshold is missed but no hard limit is.

struct PaperBranchTarget {
    const char* cell;
    char branch;        // 'A' or 'B' from RF_BPF.docx C_X-A / C_X-B branches
    double delta_pf;    // ideal target increment from RF_BPF.docx Table I(b)
};

inline const std::vector<PaperBranchTarget>& paper_branch_targets() {
    // Table I(b) of RF_BPF.docx, omitted branches excluded:
    //   C2 has only an N77 increment (= 'A')
    //   C9 has only an N78 increment (treated as 'A' here for naming
    //   consistency with the schematic). We retain the paper's value 0.41 pF.
    static const std::vector<PaperBranchTarget> kTargets = {
        {"C2", 'A', 0.57},
        {"C3", 'A', 4.15},
        {"C3", 'B', 2.47},
        {"C4", 'A', 0.19},
        {"C4", 'B', 0.47},
        {"C6", 'A', 4.12},
        {"C6", 'B', 0.75},
        {"C7", 'A', 0.38},
        {"C7", 'B', 0.89},
        {"C8", 'A', 0.41},
        {"C8", 'B', 0.17},
        {"C9", 'A', 0.41},
    };
    return kTargets;
}

struct PaperAuditRow {
    std::string cell;
    char branch = 'A';
    double freq_ghz = 0.0;
    double delta_pf = 0.0;
    double r_on_ohm = 0.0;
    double x = 0.0;
    double dc_loss_pf = 0.0;
    double dc_loss_frac = 0.0;
    double branch_q = std::numeric_limits<double>::infinity();
    std::string status = "PASS";
};

struct PaperAuditSummary {
    std::vector<PaperAuditRow> rows;
    bool any_hard_fail = false;
    bool any_warn = false;
    double worst_x = 0.0;
    double worst_dc_loss_frac = 0.0;
    double min_q = std::numeric_limits<double>::infinity();
    double r_on_ohm = 0.0;
    double w_total_um = 0.0;
};

PaperAuditSummary audit_mos_against_paper_targets(
    const MOSSwitchParams& mos,
    const RFModelParams& params,
    const std::vector<double>& freqs_ghz) {
    PaperAuditSummary out;
    out.w_total_um = mos.total_width_per_transistor_um();
    if (out.w_total_um <= 0.0) {
        // Caller will treat empty audit + nonzero w_total_um == 0 as HARD_FAIL.
        out.any_hard_fail = true;
        return out;
    }
    out.r_on_ohm = ron_stack_ohm(mos, params);
    if (!std::isfinite(out.r_on_ohm) || out.r_on_ohm <= 0.0) {
        out.any_hard_fail = true;
        return out;
    }

    const auto& targets = paper_branch_targets();
    for (const auto& tgt : targets) {
        for (double f_ghz : freqs_ghz) {
            const double omega = 2.0 * PI * f_ghz * 1.0e9;
            const double dc_f = tgt.delta_pf * 1.0e-12;
            const double x = omega * out.r_on_ohm * dc_f;
            const double dc_loss_f = dc_f * (x * x) / (1.0 + x * x);
            const double dc_loss_pf = dc_loss_f * 1.0e12;
            const double dc_loss_frac =
                tgt.delta_pf > 0.0 ? dc_loss_pf / tgt.delta_pf : 0.0;
            const double q = (x > 0.0)
                ? 1.0 / x
                : std::numeric_limits<double>::infinity();

            std::string status;
            const bool hard_x        = x >= 1.0;
            const bool hard_loss     = dc_loss_frac >= 0.10;
            const bool hard_q        = q < params.min_branch_q;
            const bool warn_paper_x  = x >= 0.1;
            const bool warn_loss     = dc_loss_frac >= 0.01;
            const bool warn_q        = q < params.warn_branch_q;
            if (hard_x || hard_loss || hard_q) {
                status = "HARD_FAIL";
                out.any_hard_fail = true;
            } else if (warn_paper_x || warn_loss || warn_q) {
                status = "WARN";
                out.any_warn = true;
            } else {
                status = "PASS";
            }

            PaperAuditRow row;
            row.cell = tgt.cell;
            row.branch = tgt.branch;
            row.freq_ghz = f_ghz;
            row.delta_pf = tgt.delta_pf;
            row.r_on_ohm = out.r_on_ohm;
            row.x = x;
            row.dc_loss_pf = dc_loss_pf;
            row.dc_loss_frac = dc_loss_frac;
            row.branch_q = q;
            row.status = std::move(status);

            if (x > out.worst_x) out.worst_x = x;
            if (dc_loss_frac > out.worst_dc_loss_frac) out.worst_dc_loss_frac = dc_loss_frac;
            if (q < out.min_q) out.min_q = q;
            out.rows.push_back(std::move(row));
        }
    }
    return out;
}

void print_paper_audit_table(const PaperAuditSummary& audit) {
    std::cout << "Paper-target branch audit (RF_BPF.docx Table I(b), "
              << "omega*Ron*dC threshold per Section III-B):\n";
    std::cout << "  Cell  Br  Freq_GHz  dC_pF    Ron_ohm   x=wRC      dC_loss_pF  loss_%   Q_eff      Status\n";
    if (audit.rows.empty()) {
        std::cout << "  (no rows; W_total <= 0 or Ron not finite)\n";
        return;
    }
    const std::ios_base::fmtflags fl = std::cout.flags();
    for (const auto& r : audit.rows) {
        std::cout << "  " << std::setw(4) << std::left << r.cell
                  << "  " << std::setw(2) << r.branch
                  << "  " << std::right << std::fixed << std::setprecision(2)
                  << std::setw(8) << r.freq_ghz
                  << "  " << std::setw(7) << std::setprecision(2) << r.delta_pf
                  << "  " << std::setw(8) << std::setprecision(2) << r.r_on_ohm
                  << "  " << std::scientific << std::setprecision(3)
                  << std::setw(10) << r.x
                  << "  " << std::fixed << std::setprecision(4)
                  << std::setw(10) << r.dc_loss_pf
                  << "  " << std::setw(7) << std::setprecision(2)
                  << (r.dc_loss_frac * 100.0)
                  << "  " << std::scientific << std::setprecision(3)
                  << std::setw(10)
                  << (std::isfinite(r.branch_q) ? r.branch_q : 0.0)
                  << "  " << std::left << r.status << "\n";
        std::cout.flags(fl);
    }
    std::cout.flags(fl);
}

void print_paper_audit_verdict(
    const PaperAuditSummary& audit,
    const std::vector<double>& freqs_ghz,
    const RFModelParams& params) {
    constexpr double kPaperFirstPassWum = 200.0;
    const double w_total = audit.w_total_um;
    const double ratio = (w_total > 0.0)
        ? kPaperFirstPassWum / w_total
        : std::numeric_limits<double>::infinity();

    double dc_max_pf = 0.0;
    for (const auto& t : paper_branch_targets()) {
        if (t.delta_pf > dc_max_pf) dc_max_pf = t.delta_pf;
    }
    const double f_top_ghz = freqs_ghz.empty()
        ? 5.0
        : *std::max_element(freqs_ghz.begin(), freqs_ghz.end());
    const double omega_top = 2.0 * PI * f_top_ghz * 1.0e9;
    const double r_on_max_rel_ohm =
        (omega_top * dc_max_pf > 0.0)
            ? 0.1 / (omega_top * dc_max_pf * 1.0e-12)
            : std::numeric_limits<double>::infinity();
    // Per-paper K_R that this binary uses for the two-NMOS stack:
    // K_R = ron_ref_ohm * w_ref_um * (stack_factor * body_factor); for the
    // default bodytie + 2-stack settings this collapses to ron_ref * w_ref.
    const double k_r_eff_ohm_um = (audit.r_on_ohm > 0.0 && w_total > 0.0)
        ? audit.r_on_ohm * w_total
        : params.ron_ref_ohm * params.w_ref_um;
    const double w_req_um = (r_on_max_rel_ohm > 0.0)
        ? k_r_eff_ohm_um / r_on_max_rel_ohm
        : std::numeric_limits<double>::infinity();

    std::cout << "Paper-audit summary:\n";
    std::cout << "  W_total_per_transistor_um         = "
              << std::fixed << std::setprecision(4) << w_total << "\n";
    std::cout << "  paper_first_pass_W_sw_um          = "
              << kPaperFirstPassWum << "\n";
    std::cout << "  W_paper / W_actual                = "
              << std::setprecision(2) << ratio << "x\n";
    std::cout << "  Ron_stack_ohm                     = "
              << std::setprecision(2) << audit.r_on_ohm << "\n";
    std::cout << "  K_R_effective_ohm_um (Ron*W)      = "
              << std::setprecision(1) << k_r_eff_ohm_um << "\n";
    std::cout << "  worst_x (omega*Ron*dC)            = "
              << std::scientific << std::setprecision(3) << audit.worst_x << "\n";
    std::cout << "  worst_dC_loss_fraction            = "
              << std::fixed << std::setprecision(4) << audit.worst_dc_loss_frac << "\n";
    std::cout << "  min_branch_Q_over_grid            = "
              << std::scientific << std::setprecision(3)
              << (std::isfinite(audit.min_q) ? audit.min_q : 0.0) << "\n";
    std::cout << "  W_req_um @ dC_max=" << std::fixed << std::setprecision(2)
              << dc_max_pf << " pF, f=" << f_top_ghz
              << " GHz, x<0.1 = "
              << std::setprecision(0) << w_req_um
              << " um (or use parallel segmentation)\n";
    if (audit.any_hard_fail) {
        std::cout << "Paper-audit verdict: FAIL -- n33_ckt_rf W_total = "
                  << std::setprecision(4) << w_total
                  << " um is " << std::setprecision(2) << ratio
                  << "x undersized vs paper first-pass W_sw = "
                  << std::setprecision(0) << kPaperFirstPassWum
                  << " um; C3-A and C6-A branches exceed paper's "
                  << "omega*Ron*dC threshold by orders of magnitude.\n";
    } else if (audit.any_warn) {
        std::cout << "Paper-audit verdict: WARN -- some branches "
                  << "exceed paper's x<0.1 first-order limit; segmentation "
                  << "or larger W is recommended.\n";
    } else {
        std::cout << "Paper-audit verdict: PASS_FIRST_ORDER -- all branches "
                  << "satisfy x<0.1 for this MOS sizing; full S-parameter "
                  << "verification still required.\n";
    }
    (void)params;  // reserved for future thresholds; suppress -Wunused
}

void print_pcell_warnings(
    double c_drawn_pf,
    const MOSSwitchParams& mos,
    const std::vector<double>& freqs_ghz,
    const RFModelParams& params,
    bool calibrated_from_json) {
    std::vector<std::string> warnings;
    if (!calibrated_from_json) {
        push_unique(warnings, "No calibration JSON supplied; default coefficients are SMIC N130 first-order estimates and must be replaced by extracted n33_ckt_rf/mim2_rf_2mask PDK data.");
    }
    if (!has_mim_loss_input(params)) {
        push_unique(warnings, "MIM Q/ESR is not supplied. The capacitor PCell density is checked, but RF loss is not fully verified. Extract MIM Q/ESR from the PDK RF model.");
    } else if (supplied(params.mim_q) && supplied(params.mim_esr_ohm)) {
        push_unique(warnings, "Both mim_q and mim_esr_ohm are supplied; mim_esr_ohm is used directly in the on-state RF branch model.");
    }

    const double ron = ron_stack_ohm(mos, params);
    double min_q = std::numeric_limits<double>::infinity();
    for (double freq_ghz : freqs_ghz) {
        min_q = std::min(min_q, switched_branch_on_eff(
            c_drawn_pf, freq_ghz, mos, params, 0.0).q_eff);
    }

    if (min_q < params.warn_branch_q) {
        push_unique(warnings, "Branch Q is low. Insertion loss may be high. Verify with PDK RF S-parameter simulation.");
    }
    if (min_q < params.min_branch_q) {
        std::ostringstream msg;
        msg << "Exact branch Q falls below min_branch_q="
            << std::fixed << std::setprecision(3) << params.min_branch_q
            << "; capacitance match alone is not enough.";
        push_unique(warnings, msg.str());
    }
    if (ron > params.max_ron_stack_ohm) {
        push_unique(warnings, "Ron_stack is too high for a pF-level RF switched-capacitor branch.");
    } else if (ron > params.warn_ron_stack_ohm) {
        push_unique(warnings, "Ron_stack may be acceptable only after full S-parameter verification.");
    }

    if (nearly_equal(mos.channel_width_um, 1.0) &&
        mos.fingers == 4 && mos.multiplier == 1) {
        push_unique(warnings, "Detected 4 um total MOS width. This is far below the current first-pass reference width of 200 um. For pF-level RF switched-capacitor branches this will produce excessive Ron and poor branch Q. Do not use this as the final switch size for large delta branches.");
    }
    if (c_drawn_pf >= 1.0 && (ron > params.max_ron_stack_ohm || min_q < params.min_branch_q)) {
        push_unique(warnings, "PCell MOS size is physically too small for this switched-capacitor branch. It may be acceptable only for leakage-isolation or tiny dummy structures, not for a pF-level RF on-state capacitor branch. Increase total MOS width, segment the branch, or rerun optimization with branch-specific MOS sizing.");
    }

    std::cout << "Warnings:\n";
    if (warnings.empty()) {
        std::cout << "  - none\n";
    } else {
        for (const auto& warning : warnings) {
            std::cout << "  - " << warning << "\n";
        }
    }
}

bool print_pcell_check(const CommandLine& args, const RFModelParams& params, bool calibrated_from_json) {
    MOSSwitchParams mos = make_mos_from_command_line(args, args.mos_multiplier);
    const double area_um2 = args.cap_width_um * args.cap_length_um;
    const double c_drawn_ff = area_um2 * args.density_ff_per_um2;
    const double c_drawn_pf = c_drawn_ff / 1000.0;
    const double w_total_um = mos.total_width_per_transistor_um();
    const double ron = ron_stack_ohm(mos, params);
    const double coff_pf = switched_branch_off_eff_pf(c_drawn_pf, mos, params, 0.0);
    const std::string verdict = determine_pcell_verdict(
        c_drawn_pf, args.target_pf, mos, args.freqs_ghz, params, calibrated_from_json);
    const PaperAuditSummary audit =
        audit_mos_against_paper_targets(mos, params, args.freqs_ghz);

    std::cout << "============================================================\n";
    std::cout << "PCELL RF CONSISTENCY CHECK\n";
    std::cout << "============================================================\n\n";

    std::cout << "MIM PCell nominal drawn value:\n";
    std::cout << "  cap_model              = " << args.cap_model << "\n";
    std::cout << "  width_um               = " << std::fixed << std::setprecision(4)
              << args.cap_width_um << "\n";
    std::cout << "  length_um              = " << args.cap_length_um << "\n";
    std::cout << "  density_ff_per_um2     = " << args.density_ff_per_um2 << "\n";
    std::cout << "  area_um2               = " << area_um2 << "\n";
    std::cout << "  C_drawn_fF             = " << std::setprecision(4) << c_drawn_ff << "\n";
    std::cout << "  C_drawn_pF             = " << std::setprecision(6) << c_drawn_pf << "\n";
    if ((args.cap_model == "mim2_rf_2mask" ||
         args.cap_model == "mim2_shield_rf_2mask") &&
        nearly_equal(args.cap_width_um, 25.0) &&
        nearly_equal(args.cap_length_um, 25.0) &&
        nearly_equal(args.density_ff_per_um2, 2.1) &&
        nearly_equal(c_drawn_pf, 1.3125)) {
        std::cout << "  Cadence screenshot Case A = PASS; 25 * 25 * 2.1 fF = 1312.5 fF = 1.3125 pF\n";
    }
    // Per RF_BPF.docx Section III-E: CFIX,drawn must be <= CN79,target,
    // because off-state residuals of disabled branches add to it.
    // Therefore C_drawn_pF > target_pf is wrong direction, not a margin.
    bool target_direction_wrong = false;
    if (supplied(args.target_pf) && c_drawn_pf > args.target_pf * 1.001) {
        std::cout << "  C_FIX drawn = " << std::fixed << std::setprecision(6)
                  << c_drawn_pf << " pF vs target <= " << args.target_pf
                  << " pF -- direction WRONG (paper III-E: CFIX,drawn = "
                  << "CN79,target - C77,off - C78,off - Ccommon, so drawn "
                  << "must be <= target, not >)\n";
        target_direction_wrong = true;
    }
    std::cout << "\n";

    std::cout << "MOS PCell first-pass switch model:\n";
    print_mos_fields(mos, "  ");
    std::cout << "  W_total_per_transistor_um = " << std::setprecision(4)
              << w_total_um << "\n";
    std::cout << "  Ron_stack_ohm             = " << std::setprecision(4)
              << ron << "\n";
    std::cout << "  Coff_branch_pF            = " << std::setprecision(6)
              << coff_pf << "\n";
    if (nearly_equal(args.mos_width_um, 1.0) &&
        args.mos_fingers == 4 && args.mos_multiplier == 1) {
        std::cout << "  Cadence screenshot Case B = W_total arithmetic PASS; "
                  << "RF switch suitability HARD FAIL for pF-level delta branches "
                  << "(see paper-audit table below)\n";
    }
    std::cout << "\n";

    std::cout << "Value separation:\n";
    std::cout << "  PCell nominal drawn capacitance C_drawn_pF = "
              << std::fixed << std::setprecision(6) << c_drawn_pf << " pF\n";
    std::cout << "  On-state effective RF capacitance Ceff_on = imag(Y_on)/omega; see table\n";
    std::cout << "  Off-state residual capacitance Coff_branch = "
              << coff_pf << " pF\n";
    std::cout << "  Target ideal capacitance C_target = "
              << optional_pf_text(args.target_pf) << "\n";
    std::cout << "  Extracted value = "
              << optional_pf_text(args.extracted_ceff_pf)
              << "; not available from screenshots alone\n\n";

    std::cout << "RF on-state table (this MIM only):\n";
    std::cout << "  Freq_GHz   Xc_ohm   Ron_stack_ohm   ESR_mim_ohm   Q_approx   Q_exact   Ceff_on_pF   G_on_S\n";
    for (double freq_ghz : args.freqs_ghz) {
        const BranchRFValues rf = switched_branch_on_eff(
            c_drawn_pf, freq_ghz, mos, params, 0.0);
        std::cout << "  " << std::setw(7) << std::setprecision(2) << freq_ghz
                  << "   " << std::setw(7) << std::setprecision(3) << rf.x_cap_ohm
                  << "   " << std::setw(13) << std::setprecision(4) << rf.ron_stack_ohm
                  << "   " << std::setw(11) << std::setprecision(5) << rf.esr_mim_ohm
                  << "   " << std::setw(8) << std::setprecision(3) << rf.q_approx
                  << "   " << std::setw(7) << std::setprecision(3) << rf.q_eff
                  << "   " << std::setw(10) << std::setprecision(6) << rf.c_series_eff_pf
                  << "   " << std::setw(8) << std::setprecision(6) << rf.g_eff_s
                  << "\n";
    }
    std::cout << "\n";

    print_paper_audit_table(audit);
    std::cout << "\n";
    print_paper_audit_verdict(audit, args.freqs_ghz, params);
    std::cout << "\n";

    std::cout << "Thresholds:\n";
    std::cout << "  min_branch_q        = " << params.min_branch_q << "\n";
    std::cout << "  warn_branch_q       = " << params.warn_branch_q << "\n";
    std::cout << "  max_ron_stack_ohm   = " << params.max_ron_stack_ohm << "\n";
    std::cout << "  warn_ron_stack_ohm  = " << params.warn_ron_stack_ohm << "\n\n";

    const bool overall_hard_fail = audit.any_hard_fail || target_direction_wrong;
    std::cout << "Final verdict: "
              << (overall_hard_fail ? std::string("FAIL") : verdict);
    if (overall_hard_fail) {
        std::cout << " (paper-target audit: "
                  << (audit.any_hard_fail ? "MOS undersized" : "MIM oversized vs CN79")
                  << ")";
    }
    std::cout << "\n\n";
    print_pcell_warnings(c_drawn_pf, mos, args.freqs_ghz, params, calibrated_from_json);

    std::cout << "\n============================================================\n";
    std::cout << "This checker can confirm nominal PCell density arithmetic. MOS Ron/Q/Coff and final RF capacitance still require PDK RF S-parameter simulation and PEX.\n";
    return !overall_hard_fail;
}

bool print_pcell_self_test(const RFModelParams& params, bool calibrated_from_json) {
    CommandLine example;
    example.check_pcell = true;
    example.cap_model = "mim2_rf_2mask";
    example.cap_width_um = 25.0;
    example.cap_length_um = 25.0;
    example.density_ff_per_um2 = 2.1;
    // C3 fixed-cap target from RF_BPF.docx Table I(b). Paper requires
    // CFIX,drawn <= 0.85 pF, but the screenshot draws 1.3125 pF, so the
    // self-test is expected to flag a target-direction violation in addition
    // to the MOS undersizing.
    example.target_pf = 0.85;
    example.mos_model = "n33_ckt_rf";
    example.mos_width_um = 1.0;
    example.mos_length_nm = 350.0;
    example.mos_fingers = 4;
    example.mos_multiplier = 1;
    example.bodytie = true;
    example.freqs_ghz = {3.3, 3.8, 4.2, 4.6, 5.0};
    return print_pcell_check(example, params, calibrated_from_json);
}

struct FilterTzResult {
    double ftz1_ghz = 0.0;
    double ftz2_ghz = 0.0;
    double ftz3_ghz = 0.0;
    double ftz4_ghz = 0.0;
    bool ascending = false;
};

double filter_tz_frequency_ghz(double inductance_nh, double capacitance_pf) {
    const double inductance_h = inductance_nh * 1.0e-9;
    const double capacitance_f = capacitance_pf * 1.0e-12;
    if (inductance_h <= 0.0 || capacitance_f <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 1.0 / (2.0 * PI * std::sqrt(inductance_h * capacitance_f)) * 1.0e-9;
}

FilterTzResult compute_filter_tz_result(const CommandLine& args) {
    FilterTzResult result;
    result.ftz1_ghz = filter_tz_frequency_ghz(
        args.filter_l3_nh, args.filter_c3_pf + args.filter_c4_pf);
    result.ftz2_ghz = filter_tz_frequency_ghz(
        args.filter_l4_nh, args.filter_c6_pf + args.filter_c7_pf);
    result.ftz3_ghz = filter_tz_frequency_ghz(args.filter_l2_nh, args.filter_c9_pf);
    result.ftz4_ghz = filter_tz_frequency_ghz(args.filter_l1_nh, args.filter_c1_pf);
    result.ascending =
        result.ftz1_ghz < result.ftz2_ghz &&
        result.ftz2_ghz < result.ftz3_ghz &&
        result.ftz3_ghz < result.ftz4_ghz;
    return result;
}

void print_filter_verification_json(const CommandLine& args, const FilterTzResult& tz) {
    std::cout << std::setprecision(12);
    std::cout << "{\n";
    std::cout << "  \"type\": \"transmission_zero_formula_check\",\n";
    std::cout << "  \"inputs\": {"
              << "\"L1_nH\": " << args.filter_l1_nh
              << ", \"L2_nH\": " << args.filter_l2_nh
              << ", \"L3_nH\": " << args.filter_l3_nh
              << ", \"L4_nH\": " << args.filter_l4_nh
              << ", \"C1_pF\": " << args.filter_c1_pf
              << ", \"C3_pF\": " << args.filter_c3_pf
              << ", \"C4_pF\": " << args.filter_c4_pf
              << ", \"C6_pF\": " << args.filter_c6_pf
              << ", \"C7_pF\": " << args.filter_c7_pf
              << ", \"C9_pF\": " << args.filter_c9_pf
              << "},\n";
    std::cout << "  \"transmission_zeros_ghz\": {"
              << "\"fTZ1\": " << tz.ftz1_ghz
              << ", \"fTZ2\": " << tz.ftz2_ghz
              << ", \"fTZ3\": " << tz.ftz3_ghz
              << ", \"fTZ4\": " << tz.ftz4_ghz
              << "},\n";
    std::cout << "  \"ascending_order\": " << (tz.ascending ? "true" : "false") << "\n";
    std::cout << "}\n";
}

void print_filter_verification_csv(const FilterTzResult& tz) {
    std::cout << "fTZ1_GHz,fTZ2_GHz,fTZ3_GHz,fTZ4_GHz,ascending_order\n";
    std::cout << std::setprecision(12)
              << tz.ftz1_ghz << ","
              << tz.ftz2_ghz << ","
              << tz.ftz3_ghz << ","
              << tz.ftz4_ghz << ","
              << (tz.ascending ? "true" : "false") << "\n";
}

void print_filter_verification(const CommandLine& args) {
    const FilterTzResult tz = compute_filter_tz_result(args);
    if (args.json) {
        print_filter_verification_json(args, tz);
        return;
    }
    if (args.csv) {
        print_filter_verification_csv(tz);
        return;
    }

    std::cout << "============================================================\n";
    std::cout << "TRANSMISSION-ZERO FORMULA CHECK\n";
    std::cout << "============================================================\n\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "fTZ1 = 1/(2*pi*sqrt((C3+C4)*L3)) = "
              << tz.ftz1_ghz << " GHz\n";
    std::cout << "fTZ2 = 1/(2*pi*sqrt((C6+C7)*L4)) = "
              << tz.ftz2_ghz << " GHz\n";
    std::cout << "fTZ3 = 1/(2*pi*sqrt(C9*L2))      = "
              << tz.ftz3_ghz << " GHz\n";
    std::cout << "fTZ4 = 1/(2*pi*sqrt(C1*L1))      = "
              << tz.ftz4_ghz << " GHz\n\n";
    std::cout << "TZ ascending order fTZ1 < fTZ2 < fTZ3 < fTZ4 = "
              << (tz.ascending ? "PASS" : "REVIEW") << "\n";
    std::cout << "This is not S-parameter verification.\n";
    std::cout << "It only checks approximate LC transmission-zero formulas.\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        CommandLine args = parse_command_line(argc, argv);
        RFModelParams params;
        const bool calibrated_from_json =
            load_calibration_json(args.calibration_json, params);
        apply_cli_overrides(args, params);
        validate_model_params(params);

        if (args.verify_filter) {
            print_filter_verification(args);
            return 0;
        }

        if (args.check_pcell) {
            const bool ok =
                args.pcell_self_test
                    ? print_pcell_self_test(params, calibrated_from_json)
                    : print_pcell_check(args, params, calibrated_from_json);
            if (args.strict && !ok) {
                return 2;
            }
            return 0;
        }

        Targets targets{args.cn77, args.cn78, args.cn79};
        PrelayoutConfig cfg = make_prelayout_config_from_args(args);
        PrelayoutResult result = solve_prelayout_result(args, targets, cfg);
        apply_conservative_crosscheck_verdict(args, targets, result);
        if (args.json) {
            print_prelayout_json(result);
        } else if (args.csv) {
            print_prelayout_csv(result);
        } else {
            print_prelayout_report(result);
            if (args.sweep_parasitics) {
                print_parasitic_sweep(args, targets);
            }
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        std::cerr << "Run with --help for usage.\n";
        return 1;
    }
}
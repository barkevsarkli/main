// version6 -- mixed-activation study, extended to a small CNN (MNIST and CIFAR-10)
//
// One process = one run = one thread. See README.md for the build and PROMPT.md /
// DECISIONS.md for the protocol. Relative to version5 (whose sources were imported
// unchanged in the first version6 commit):
//   * --arch mlp2 (version5 network), mlp1 (one hidden layer), cnn (one conv block +
//     one hidden dense layer); fixed kernel banks for the conv layer (kernels.h)
//   * independent RNG streams for weights, masks and data order (--rng-mode split);
//     --rng-mode legacy reproduces version5 bit for bit for mlp2
//   * NaN/Inf check on every logit and the loss at every step
//   * validation CE loss per epoch, epochs_completed, provenance and hashes in the row
//   * diagnostics (--stats-out), optional collapse abort, one result file per run
//
// Usage: main --help

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "dataset.h"
#include "kernels.h"
#include "model.h"
#include "net.h"
#include "provenance.h"
#include "rng_util.h"
#include "stats.h"
#include "train_util.h"
#include "util.h"

namespace {

struct Flag { const char* name; const char* def; const char* help; bool is_switch; };

// Every flag the binary accepts, with its default. Unknown flags are errors.
const Flag FLAGS[] = {
    {"--arch",              "cnn",        "mlp2 | mlp1 | cnn", false},
    {"--dataset",           "mnist",      "mnist | cifar10", false},
    {"--data",              "..",         "directory holding train-images-idx3-ubyte/, train-labels-idx1-ubyte/, CIFAR10/", false},
    {"--hidden",            "128",        "width H of every hidden dense layer", false},
    {"--seed",              "1",          "run seed", false},
    {"--epochs",            "40",         "training epochs", false},
    {"--lr",                "0.01",       "SGD learning rate (batch size 1)", false},
    {"--loss",              "ce",         "ce (linear outputs + softmax) | mse (sigmoid outputs)", false},
    {"--rng-mode",          "split",      "split (independent init/mask/order streams) | legacy (version5; mlp only)", false},
    {"--conv-k",            "5",          "conv kernel size K (odd)", false},
    {"--conv-channels",     "8",          "conv output channels C (8 for kernel banks)", false},
    {"--pool",              "2",          "max-pool size and stride P (1, 2, 4)", false},
    {"--kernel-mode",       "random",     "random | bank_init | frozen", false},
    {"--kernel-family",     "none",       "gauss | sharpen | sobel | laplacian | gabor | random_frozen (bank modes)", false},
    {"--kernel-seed",       "20261018",   "seed of the random_frozen bank", false},
    {"--kernel-dir",        "kernels",    "directory with kernels/<dataset>_<family>_k<K>.json", false},
    {"--act1",              "",           "convenience: sets --conv-act1 and --hid-act1", false},
    {"--act2",              "",           "convenience: sets --conv-act2 and --hid-act2", false},
    {"--ratio",             "",           "convenience: sets --conv-ratio and --hid-ratio", false},
    {"--conv-act1",         "relu",       "first activation of the conv channel mask", false},
    {"--conv-act2",         "relu",       "second activation of the conv channel mask", false},
    {"--conv-ratio",        "0",          "share of conv channels using conv-act2", false},
    {"--conv-mask-offset",  "0",          "1 = rotate the conv mask by one channel (complement at ratio 0.5)", false},
    {"--hid-act1",          "relu",       "first activation of the hidden-layer mask(s)", false},
    {"--hid-act2",          "relu",       "second activation of the hidden-layer mask(s)", false},
    {"--hid-ratio",         "0",          "share of hidden neurons using hid-act2", false},
    {"--layout",            "interleave", "interleave | block | random (mask layout, all layers)", false},
    {"--abort-on-collapse", "0",          "N > 0: stop after N epochs with |train loss - ln 10| < 0.01 and val acc < 15%", false},
    {"--stats-out",         "",           "write diagnostics (long CSV) to this path", false},
    {"--stats-every",       "1",          "diagnostics every E epochs (plus epoch 0 and the last epoch)", false},
    {"--dump-init",         "",           "write the initial weights of every layer to this path (text)", false},
    {"--run-id",            "manual",     "identifier written into the result and stats rows", false},
    {"--stage",             "manual",     "stage label written into the result row", false},
    {"--out",               "row.csv",    "result file: header + one row, written atomically (overwrites)", false},
    {"--no-mmap",           "",           "read datasets with std::ifstream instead of mapping them", true},
    {"--quiet",             "",           "no per-epoch progress lines", true},
    {"--help",              "",           "print this help", true},
    // testing hooks
    {"--dtype",             "float32",    "[test] float32 | float64 (float64 is for equivalence tests only)", false},
    {"--max-steps",         "0",          "[test] stop after this many SGD steps in total (0 = no limit)", false},
    {"--dump-final",        "",           "[test] write the final weights of every layer to this path (text)", false},
    {"--debug-order",       "0",          "[test] print the first N training indices of epoch 1", false},
    {"--no-clip",           "",           "[test] disable the dL/dz clip (gradient checks only)", true},
};

void print_help()
{
    std::printf("version6 trainer. Flags (default in brackets):\n");
    for (const Flag& f : FLAGS)
        std::printf("  %-20s %-12s %s\n", f.name, f.is_switch ? "" : (std::string("[") + f.def + "]").c_str(), f.help);
}

struct Args
{
    std::map<std::string, std::string> v;
    std::map<std::string, bool> set;
    std::string s(const char* k) const { return v.at(k); }
    int i(const char* k) const { return std::stoi(v.at(k)); }
    float f(const char* k) const { return std::stof(v.at(k)); }
    bool on(const char* k) const { return set.count(k) && set.at(k); }
};

Args parse_args(int argc, char** argv)
{
    Args a;
    for (const Flag& f : FLAGS) a.v[f.name] = f.def;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        const Flag* fl = nullptr;
        for (const Flag& f : FLAGS) if (k == f.name) fl = &f;
        if (!fl) { std::cerr << "unknown argument: " << k << " (see --help)\n"; std::exit(1); }
        a.set[k] = true;
        if (fl->is_switch) continue;
        if (i + 1 >= argc) { std::cerr << "missing value for " << k << "\n"; std::exit(1); }
        a.v[k] = argv[++i];
    }
    return a;
}

[[noreturn]] void die(const std::string& m) { std::cerr << "error: " << m << "\n"; std::exit(1); }

// Canonical description of one run: what the result row's identity columns hold. Fields
// that do not apply to an architecture are blanked ("none" / 0) so equivalent runs have
// equal keys; sweep/common.py applies the same rules on the Python side.
struct Config
{
    std::string arch, dataset, loss, rng_mode, kernel_mode, kernel_family, layout;
    std::string conv_act1, conv_act2, hid_act1, hid_act2;
    int seed, epochs, hidden, conv_k, conv_channels, pool, conv_mask_offset, abort_on_collapse;
    uint32_t kernel_seed;
    float lr, conv_ratio, hid_ratio;
};

// Homogeneous masks are written as (a, a, 0) whichever way they were requested.
void canon_mask(std::string& a1, std::string& a2, float& ratio)
{
    if (ratio <= 0.0f || a1 == a2) { a2 = a1; ratio = 0.0f; }
    else if (ratio >= 1.0f) { a1 = a2; ratio = 0.0f; }
}

Config make_config(const Args& a)
{
    Config c;
    c.arch = a.s("--arch"); c.dataset = a.s("--dataset"); c.loss = a.s("--loss");
    c.rng_mode = a.s("--rng-mode"); c.layout = a.s("--layout");
    c.seed = a.i("--seed"); c.epochs = a.i("--epochs"); c.hidden = a.i("--hidden"); c.lr = a.f("--lr");
    c.abort_on_collapse = a.i("--abort-on-collapse");

    if (c.arch != "mlp1" && c.arch != "mlp2" && c.arch != "cnn") die("--arch must be mlp1, mlp2 or cnn");
    if (c.dataset != "mnist" && c.dataset != "cifar10") die("--dataset must be mnist or cifar10");
    if (c.loss != "ce" && c.loss != "mse") die("--loss must be ce or mse");
    if (c.rng_mode != "split" && c.rng_mode != "legacy") die("--rng-mode must be split or legacy");
    if (c.rng_mode == "legacy" && c.arch == "cnn") die("--rng-mode legacy exists only for the version5 MLPs");
    if (c.layout != "interleave" && c.layout != "block" && c.layout != "random") die("bad --layout");
    if (c.hidden < 1 || c.epochs < 0 || c.seed < 0) die("bad --hidden/--epochs/--seed");

    // Activations: convenience flags first, explicit per-layer flags win.
    c.conv_act1 = a.on("--act1") ? a.s("--act1") : a.s("--conv-act1");
    c.conv_act2 = a.on("--act2") ? a.s("--act2") : a.s("--conv-act2");
    c.conv_ratio = a.on("--ratio") ? a.f("--ratio") : a.f("--conv-ratio");
    c.hid_act1 = a.on("--act1") ? a.s("--act1") : a.s("--hid-act1");
    c.hid_act2 = a.on("--act2") ? a.s("--act2") : a.s("--hid-act2");
    c.hid_ratio = a.on("--ratio") ? a.f("--ratio") : a.f("--hid-ratio");
    if (a.on("--conv-act1")) c.conv_act1 = a.s("--conv-act1");
    if (a.on("--conv-act2")) c.conv_act2 = a.s("--conv-act2");
    if (a.on("--conv-ratio")) c.conv_ratio = a.f("--conv-ratio");
    if (a.on("--hid-act1")) c.hid_act1 = a.s("--hid-act1");
    if (a.on("--hid-act2")) c.hid_act2 = a.s("--hid-act2");
    if (a.on("--hid-ratio")) c.hid_ratio = a.f("--hid-ratio");
    for (const std::string* s : {&c.conv_act1, &c.conv_act2, &c.hid_act1, &c.hid_act2}) {
        if (*s == "linear") die("linear is reserved for the output layer");
        act_from_string(*s);
    }
    if (c.conv_ratio < 0 || c.conv_ratio > 1 || c.hid_ratio < 0 || c.hid_ratio > 1) die("ratios must be in [0, 1]");
    canon_mask(c.hid_act1, c.hid_act2, c.hid_ratio);

    if (c.arch == "cnn") {
        c.conv_k = a.i("--conv-k"); c.conv_channels = a.i("--conv-channels"); c.pool = a.i("--pool");
        c.kernel_mode = a.s("--kernel-mode"); c.kernel_family = a.s("--kernel-family");
        c.kernel_seed = (uint32_t)std::stoul(a.s("--kernel-seed"));
        c.conv_mask_offset = a.i("--conv-mask-offset");
        if (c.conv_k < 1 || c.conv_k % 2 == 0) die("--conv-k must be odd");
        if (c.conv_channels < 1) die("--conv-channels must be positive");
        if (c.pool != 1 && c.pool != 2 && c.pool != 4) die("--pool must be 1, 2 or 4");
        if (c.conv_mask_offset != 0 && c.conv_mask_offset != 1) die("--conv-mask-offset must be 0 or 1");
        if (c.kernel_mode == "random") {
            if (a.on("--kernel-family") && c.kernel_family != "none") die("--kernel-family needs --kernel-mode bank_init or frozen");
            c.kernel_family = "none";
        } else if (c.kernel_mode == "bank_init" || c.kernel_mode == "frozen") {
            bool ok = false;
            for (const char* f : KERNEL_FAMILIES) ok = ok || c.kernel_family == f;
            if (!ok) die("--kernel-family must name a bank for --kernel-mode " + c.kernel_mode);
            if (c.conv_channels != N_BANK_FILTERS) die("kernel banks have exactly 8 filters: use --conv-channels 8");
        } else {
            die("--kernel-mode must be random, bank_init or frozen");
        }
        if (c.kernel_family != "random_frozen") c.kernel_seed = 0;
        canon_mask(c.conv_act1, c.conv_act2, c.conv_ratio);
        if (c.conv_ratio == 0.0f) c.conv_mask_offset = 0;     // a homogeneous mask has no offset
    } else {
        for (const char* k : {"--conv-k", "--conv-channels", "--pool", "--kernel-mode", "--kernel-family",
                              "--kernel-seed", "--conv-act1", "--conv-act2", "--conv-ratio", "--conv-mask-offset"})
            if (a.on(k)) die(std::string(k) + " does not apply to --arch " + c.arch);
        c.conv_k = c.conv_channels = c.pool = c.conv_mask_offset = 0;
        c.kernel_mode = c.kernel_family = "none";
        c.kernel_seed = 0;
        c.conv_act1 = c.conv_act2 = "none";
        c.conv_ratio = 0.0f;
    }
    return c;
}

struct Metrics
{
    float accuracy = 0, macro_f1 = 0;
    double loss = 0;                       // mean CE (or MSE) over the set
    int confusion[OUTPUT_SIZE][OUTPUT_SIZE] = {};
};

template <class T>
Metrics evaluate(Model<T>& net, const Dataset& ds, const std::vector<uint32_t>& idx, bool ce)
{
    Metrics m;
    int correct = 0;
    double loss = 0;
    for (uint32_t i : idx) {
        const T* o = net.forward(ds.input(i));
        const int p = (int)(std::max_element(o, o + OUTPUT_SIZE) - o);
        const int t = ds.label(i);
        m.confusion[t][p]++;
        if (p == t) ++correct;
        loss += sample_loss(o, t, ce, (T*)nullptr);
    }
    m.accuracy = 100.0f * correct / idx.size();
    m.loss = loss / (double)idx.size();

    float f1_sum = 0;
    for (int c = 0; c < OUTPUT_SIZE; ++c) {
        int tp = m.confusion[c][c], fp = 0, fn = 0;
        for (int k = 0; k < OUTPUT_SIZE; ++k) if (k != c) { fp += m.confusion[k][c]; fn += m.confusion[c][k]; }
        float prec = (tp + fp) ? (float)tp / (tp + fp) : 0.0f;
        float rec  = (tp + fn) ? (float)tp / (tp + fn) : 0.0f;
        f1_sum += (prec + rec > 0) ? 2 * prec * rec / (prec + rec) : 0.0f;
    }
    m.macro_f1 = 100.0f * f1_sum / OUTPUT_SIZE;
    return m;
}

// Diagnostics exist for the float32 production path only (--stats-out is refused otherwise).
template <class T> Model<float>* as_float(Model<T>& m)
{
    if constexpr (std::is_same<T, float>::value) return &m;
    else return nullptr;
}

template <class T>
void dump_weights(Model<T>& net, const std::string& path)
{
    std::ostringstream s;
    const char* names[] = {"W", "b"};
    std::vector<std::string> layer_names;
    if (net.conv) layer_names.push_back("conv");
    for (size_t L = 0; L < net.dense.size(); ++L)
        layer_names.push_back(L + 1 == net.dense.size() ? "output" : "hidden" + std::to_string(L + 1));
    auto p = net.params();
    for (size_t i = 0; i < p.size(); ++i) {
        s << "tensor " << layer_names[i / 2] << "." << names[i % 2] << " " << p[i]->size() << "\n";
        char buf[40];
        for (size_t k = 0; k < p[i]->size(); ++k) {
            std::snprintf(buf, sizeof buf, "%.17g", (double)(*p[i])[k]);
            s << buf << (k + 1 == p[i]->size() ? "\n" : " ");
        }
    }
    if (!util::write_atomic(path, s.str())) die("cannot write " + path);
}

template <class T>
int run(const Args& a, const Config& c)
{
    const auto t0 = std::chrono::steady_clock::now();
    const bool quiet = a.on("--quiet");
    const bool ce = (c.loss == "ce");
    const bool legacy = (c.rng_mode == "legacy");

    // --- data ------------------------------------------------------------
    std::unique_ptr<Dataset> ds;
    try { ds = load_dataset(c.dataset, a.s("--data"), !a.on("--no-mmap")); }
    catch (const std::exception& e) { die(std::string(e.what()) + "\n  (is --data pointing at the folder that holds the datasets?)"); }
    Split sp = make_split(c.dataset, *ds);
    util::Fnv1a split_h;
    split_h.u32s(sp.train); split_h.u32s(sp.val); split_h.u32s(sp.test);

    // --- random streams --------------------------------------------------
    // legacy: one mt19937(seed) in version5 order (masks, weights, then every shuffle).
    // split:  three independent generators, so model choices never shift each other's draws.
    std::mt19937 legacy_rng((uint32_t)c.seed);
    std::mt19937 init_rng(detu::stream_seed((uint32_t)c.seed, detu::STREAM_INIT));
    std::mt19937 mask_rng(detu::stream_seed((uint32_t)c.seed, detu::STREAM_MASK));
    std::mt19937 order_rng(detu::stream_seed((uint32_t)c.seed, detu::STREAM_ORDER));
    std::mt19937& r_init  = legacy ? legacy_rng : init_rng;
    std::mt19937& r_mask  = legacy ? legacy_rng : mask_rng;
    std::mt19937& r_order = legacy ? legacy_rng : order_rng;

    // --- network ---------------------------------------------------------
    ModelSpec spec;
    spec.arch = c.arch;
    spec.in_c = ds->channels(); spec.in_h = ds->height(); spec.in_w = ds->width();
    spec.hidden = c.hidden;
    spec.legacy = legacy;
    spec.out_act = ce ? ACT_LINEAR : ACT_SIGMOID;
    if (c.arch == "cnn") {
        spec.conv_k = c.conv_k; spec.conv_channels = c.conv_channels; spec.pool = c.pool;
        spec.kernel_mode = c.kernel_mode;
        spec.conv_mask = make_mask(c.conv_channels, act_from_string(c.conv_act1), act_from_string(c.conv_act2),
                                   c.conv_ratio, c.layout, r_mask);
        if (c.conv_mask_offset) spec.conv_mask = rotate_mask(spec.conv_mask);
    }
    const int n_hidden = (c.arch == "mlp2") ? 2 : 1;
    for (int L = 0; L < n_hidden; ++L)
        spec.hid_masks.push_back(make_mask(c.hidden, act_from_string(c.hid_act1), act_from_string(c.hid_act2),
                                           c.hid_ratio, c.layout, r_mask));

    Model<T> net(spec);
    net.init_weights(r_init);
    if (c.arch == "cnn" && c.kernel_mode != "random") {
        BankFile bank = load_bank(bank_path(a.s("--kernel-dir"), c.dataset, c.kernel_family, c.conv_k));
        if (bank.dataset != c.dataset || bank.family != c.kernel_family || bank.k != c.conv_k)
            die("kernel bank file does not match the requested dataset/family/K");
        if (c.kernel_family == "random_frozen" && bank.kernel_seed != c.kernel_seed)
            die("random_frozen bank on disk was built with kernel seed " + std::to_string(bank.kernel_seed));
        net.set_conv_weights(bank_weights(bank.filters, bank.gains, c.conv_k, spec.in_c));
    }
    if (a.on("--no-clip")) net.set_clip(T(INFINITY));
    const std::string init_hash = net.hash_all(), init_hash_dense = net.hash_dense();
    const std::string kernel_hash = net.hash_conv();
    if (!a.s("--dump-init").empty()) dump_weights(net, a.s("--dump-init"));

    std::string mask_desc;
    {
        std::map<std::string, int> cnt;
        if (net.conv) for (Act x : spec.conv_mask) cnt["conv:" + act_to_string(x)]++;
        for (Act x : spec.hid_masks[0]) cnt["hid:" + act_to_string(x)]++;
        for (auto& kv : cnt) mask_desc += " " + kv.first + "=" + std::to_string(kv.second);
    }
    if (!quiet)
        std::cout << "version6 | " << c.arch << " " << c.dataset << " H=" << c.hidden << " seed=" << c.seed
                  << " lr=" << c.lr << " rng=" << c.rng_mode
                  << (net.conv ? " K=" + std::to_string(c.conv_k) + " " + c.kernel_mode + "/" + c.kernel_family : "")
                  << " | mask:" << mask_desc << " | train/val/test = " << sp.train.size() << "/"
                  << sp.val.size() << "/" << sp.test.size() << "\n";

    // --- diagnostics -----------------------------------------------------
    const std::string stats_path = a.s("--stats-out");
    const int stats_every = std::max(1, a.i("--stats-every"));
    std::unique_ptr<StatsCollector> stats;
    if (!stats_path.empty()) {
        if (!std::is_same<T, float>::value) die("--stats-out needs --dtype float32");
        std::vector<uint32_t> probe(sp.val.begin(), sp.val.begin() + std::min<size_t>(1000, sp.val.size()));
        stats.reset(new StatsCollector(a.s("--run-id"), probe));
        Model<float>& nf = *as_float(net);
        stats->attach(nf);
        stats->record_probe(0, nf, *ds);
        stats->record_train(0, nf);
    }

    // --- training --------------------------------------------------------
    std::vector<float> epoch_loss, epoch_val_acc, epoch_val_f1, epoch_val_loss, epoch_train_acc, epoch_train_acc_clean;
    std::vector<double> epoch_seconds;         // wall time of each epoch (training + evaluation + diagnostics)
    std::string status = "ok";
    int status_epoch = 0;
    const long long max_steps = std::stoll(a.s("--max-steps"));
    long long steps = 0;
    const int debug_order = a.i("--debug-order");
    const float lr = c.lr;
    T d_loss[OUTPUT_SIZE];

    std::vector<std::vector<T>> best_state;
    float best_val_seen = -1.0f;
    bool have_snapshot = false;
    CollapseWatch watch(c.abort_on_collapse);

    for (int epoch = 0; epoch < c.epochs && status == "ok"; ++epoch) {
        const auto te0 = std::chrono::steady_clock::now();
        detu::shuffle(sp.train, r_order);
        if (epoch == 0 && debug_order > 0) {
            std::cout << "ORDER";
            for (int k = 0; k < debug_order && k < (int)sp.train.size(); ++k) std::cout << " " << sp.train[k];
            std::cout << "\n";
        }
        if (stats) stats->begin_epoch(*as_float(net));
        double total_loss = 0;
        int correct = 0;
        size_t seen = 0;

        for (uint32_t i : sp.train) {
            const uint8_t label = ds->label(i);
            const T* o = net.forward(ds->input(i));
            const int p = (int)(std::max_element(o, o + OUTPUT_SIZE) - o);
            if (p == label) ++correct;
            const double l = sample_loss(o, label, ce, d_loss);
            total_loss += l;
            ++seen;
            if (!all_finite(o, OUTPUT_SIZE) || !std::isfinite(l)) { status = "nan"; status_epoch = epoch + 1; break; }
            net.backward(d_loss, T(lr));
            if (max_steps > 0 && ++steps >= max_steps) break;
        }
        if (status != "ok") break;          // a non-finite epoch is not evaluated or recorded

        const float tr_acc = 100.0f * correct / sp.train.size();
        Metrics tr_clean = evaluate(net, *ds, sp.train_eval, ce);
        Metrics v = evaluate(net, *ds, sp.val, ce);
        epoch_loss.push_back(total_loss / sp.train.size());
        epoch_train_acc.push_back(tr_acc);
        epoch_train_acc_clean.push_back(tr_clean.accuracy);
        epoch_val_acc.push_back(v.accuracy);
        epoch_val_f1.push_back(v.macro_f1);
        epoch_val_loss.push_back((float)v.loss);

        if (v.accuracy > best_val_seen) {
            best_val_seen = v.accuracy;
            net.save_state(best_state);
            have_snapshot = true;
        }

        const bool last = (epoch + 1 == c.epochs) || (max_steps > 0 && steps >= max_steps);
        if (stats) {
            Model<float>& nf = *as_float(net);
            stats->record_train(epoch + 1, nf);
            if ((epoch + 1) % stats_every == 0 || last) stats->record_probe(epoch + 1, nf, *ds);
        }

        epoch_seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - te0).count());
        if (!quiet)
            std::cout << "epoch " << epoch + 1 << "/" << c.epochs << "  loss=" << epoch_loss.back()
                      << "  train_acc=" << tr_acc << "  train_clean=" << tr_clean.accuracy
                      << "  val_acc=" << v.accuracy << "  val_f1=" << v.macro_f1
                      << "  val_loss=" << v.loss << std::endl;

        if (watch.update(epoch_loss.back(), v.accuracy)) { status = "collapsed_abort"; status_epoch = epoch + 1; }
        if (max_steps > 0 && steps >= max_steps) break;
    }
    const int epochs_completed = (int)epoch_val_acc.size();

    // --- final evaluation --------------------------------------------------
    // Reported twice: at the last parameters (test_acc) and at the best-validation
    // epoch's parameters (test_acc_best). After a non-finite step the last parameters are
    // evaluated as they are; status and status_epoch say why they are unusable.
    if (!a.s("--dump-final").empty()) dump_weights(net, a.s("--dump-final"));
    Metrics val  = evaluate(net, *ds, sp.val, ce);
    Metrics test = evaluate(net, *ds, sp.test, ce);
    Metrics test_best = test;
    if (have_snapshot) {
        net.load_state(best_state);
        test_best = evaluate(net, *ds, sp.test, ce);
    }
    int best_ep = epoch_val_acc.empty() ? 0 :
        (int)(std::max_element(epoch_val_acc.begin(), epoch_val_acc.end()) - epoch_val_acc.begin()) + 1;
    float best_val = epoch_val_acc.empty() ? 0.0f : *std::max_element(epoch_val_acc.begin(), epoch_val_acc.end());
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    if (!quiet)
        std::cout << "RESULT status=" << status << " test_acc=" << test.accuracy << " test_f1=" << test.macro_f1
                  << " test_acc_best=" << test_best.accuracy << " test_f1_best=" << test_best.macro_f1
                  << " val_acc=" << val.accuracy << " (" << secs << " s)\n";

    // --- result row --------------------------------------------------------
    std::vector<int> conf, conf_best;
    for (int t = 0; t < OUTPUT_SIZE; ++t) for (int p = 0; p < OUTPUT_SIZE; ++p) conf.push_back(test.confusion[t][p]);
    for (int t = 0; t < OUTPUT_SIZE; ++t) for (int p = 0; p < OUTPUT_SIZE; ++p) conf_best.push_back(test_best.confusion[t][p]);

    std::vector<std::pair<std::string, std::string>> cols;
    auto put = [&](const char* k, const auto& v) { std::ostringstream s; s << v; cols.emplace_back(k, s.str()); };
    put("run_id", a.s("--run-id")); put("stage", a.s("--stage"));
    put("arch", c.arch); put("dataset", c.dataset); put("seed", c.seed); put("lr", c.lr); put("epochs", c.epochs);
    put("loss", c.loss); put("rng_mode", c.rng_mode); put("conv_k", c.conv_k); put("conv_channels", c.conv_channels);
    put("pool", c.pool); put("kernel_mode", c.kernel_mode); put("kernel_family", c.kernel_family);
    put("kernel_seed", c.kernel_seed); put("conv_act1", c.conv_act1); put("conv_act2", c.conv_act2);
    put("conv_ratio", c.conv_ratio); put("conv_mask_offset", c.conv_mask_offset); put("hid_act1", c.hid_act1);
    put("hid_act2", c.hid_act2); put("hid_ratio", c.hid_ratio); put("layout", c.layout); put("hidden", c.hidden);
    put("abort_on_collapse", c.abort_on_collapse);
    put("status", status); put("status_epoch", status_epoch); put("epochs_completed", epochs_completed);
    put("seconds", secs);
    put("train_acc", epoch_train_acc.empty() ? 0.0f : epoch_train_acc.back());
    put("val_acc", val.accuracy); put("val_f1", val.macro_f1);
    put("test_acc", test.accuracy); put("test_f1", test.macro_f1);
    put("best_val_acc", best_val); put("best_val_epoch", best_ep);
    put("test_acc_best", test_best.accuracy); put("test_f1_best", test_best.macro_f1);
    put("train_acc_clean", epoch_train_acc_clean.empty() ? 0.0f : epoch_train_acc_clean.back());
    put("epoch_loss", util::join(epoch_loss)); put("epoch_train_acc", util::join(epoch_train_acc));
    put("epoch_train_acc_clean", util::join(epoch_train_acc_clean)); put("epoch_val_acc", util::join(epoch_val_acc));
    put("epoch_val_f1", util::join(epoch_val_f1)); put("epoch_val_loss", util::join(epoch_val_loss));
    put("test_confusion", util::join(conf)); put("test_confusion_best", util::join(conf_best));
    put("git_hash", V6_GIT_HASH); put("git_dirty", V6_GIT_DIRTY); put("compiler", V6_COMPILER);
    put("build_flags", V6_BUILD_FLAGS); put("platform", V6_PLATFORM);
    put("n_params_trainable", net.n_params_trainable()); put("n_params_total", net.n_params_total());
    put("init_hash", init_hash); put("init_hash_dense", init_hash_dense); put("kernel_hash", kernel_hash);
    put("split_hash", split_h.hex());
    put("dtype", std::is_same<T, float>::value ? "float32" : "float64");
    put("epoch_seconds", util::join(epoch_seconds));

    std::string header, row;
    for (size_t i = 0; i < cols.size(); ++i) {
        header += (i ? "," : "") + cols[i].first;
        row += (i ? "," : "") + util::csv_field(cols[i].second);
    }
    if (!util::write_atomic(a.s("--out"), header + "\n" + row + "\n")) die("cannot write " + a.s("--out"));
    if (stats && !util::write_atomic(stats_path, StatsCollector::header() + stats->text()))
        die("cannot write " + stats_path);
    return status == "ok" ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv)
{
    Args a = parse_args(argc, argv);
    if (a.on("--help")) { print_help(); return 0; }
    try {
        Config c = make_config(a);
        const std::string dt = a.s("--dtype");
        if (dt == "float32") return run<float>(a, c);
        if (dt == "float64") return run<double>(a, c);
        die("--dtype must be float32 or float64");
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

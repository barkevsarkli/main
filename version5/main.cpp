// version5 -- hybrid-activation MLP experiment runner (MNIST and CIFAR-10)
//
// Differences from v2/v3 (see README):
//   * per-neuron activation masks (ratio + layout) instead of a hardcoded even/odd split
//   * output layer is uniform (linear+softmax for CE, sigmoid for MSE); hidden layers
//     are the only place hybridisation happens, so the effect is isolated
//   * consistent leaky_relu slope in forward and backward
//   * fixed, seed-independent train/val/test split (45k/5k/10k) shared by every run
//   * per-epoch shuffling of the training order (seeded)
//   * accuracy, macro-F1 and confusion matrix on validation (every epoch) and test (end)
//   * learning rate, hidden size, epochs, loss, dataset, data path all on the CLI
//   * std::mt19937 instead of rand() so seeds mean the same thing on every platform
//
// Usage:
//   ./main --act1 tanh --act2 relu --ratio 0.5 --layout interleave --hidden 12
//     --seed 50 --epochs 10 --lr 0.01 --loss ce --dataset mnist --data ../ --out results.csv
//
// ratio is the fraction of hidden neurons that use act2.  ratio 0 -> homogeneous act1.
//
// --dataset selects mnist (784 inputs, 10 classes) or cifar10 (3072 inputs, 10 classes).
// Both have 10 classes, so the output layer, confusion matrix, macro-F1 and the CSV
// schema are identical; only the input width differs.  The split differs in provenance:
// MNIST has no separate test file in this repo, so its 10k test set is a fixed slice of
// the 60k training file, while CIFAR-10 uses the official test_batch.bin and takes only
// the 5k validation set out of the 50k training batches.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <map>
#include <memory>
#include "dataset.h"
#include "net.h"
#include "rng_util.h"

#define OUTPUT_SIZE 10
#define SPLIT_SEED 12345u   // fixed permutation for the train/val/test split
#define N_TEST 10000
#define N_VAL   5000

struct Metrics {
    float accuracy = 0, macro_f1 = 0;
    int confusion[OUTPUT_SIZE][OUTPUT_SIZE] = {};
};

struct Args {
    std::string act1 = "relu", act2 = "relu", layout = "interleave", loss = "ce";
    std::string dataset = "mnist";
    std::string data = "../", out = "results.csv";
    float ratio = 0.0f, lr = 0.01f;
    int hidden = 12, seed = 50, epochs = 10;
    bool quiet = false;
};

static Args parse_args(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << name << "\n"; exit(1); }
            return argv[++i];
        };
        if      (k == "--act1")   a.act1   = need("--act1");
        else if (k == "--act2")   a.act2   = need("--act2");
        else if (k == "--ratio")  a.ratio  = std::stof(need("--ratio"));
        else if (k == "--layout") a.layout = need("--layout");
        else if (k == "--hidden") a.hidden = std::stoi(need("--hidden"));
        else if (k == "--seed")   a.seed   = std::stoi(need("--seed"));
        else if (k == "--epochs") a.epochs = std::stoi(need("--epochs"));
        else if (k == "--lr")     a.lr     = std::stof(need("--lr"));
        else if (k == "--loss")   a.loss   = need("--loss");
        else if (k == "--dataset") a.dataset = need("--dataset");
        else if (k == "--data")   a.data   = need("--data");
        else if (k == "--out")    a.out    = need("--out");
        else if (k == "--quiet")  a.quiet  = true;
        else { std::cerr << "unknown argument: " << k << "\n"; exit(1); }
    }
    if (a.loss != "ce" && a.loss != "mse") { std::cerr << "--loss must be ce or mse\n"; exit(1); }
    if (a.dataset != "mnist" && a.dataset != "cifar10") { std::cerr << "--dataset must be mnist or cifar10\n"; exit(1); }
    return a;
}

// Forward through the stack, return index of the max output.
static int predict(std::vector<Layer>& net, const float* x)
{
    const float* h = x;
    for (auto& L : net) { L.forward(h); h = L.outputs(); }
    const float* o = net.back().outputs();
    return (int)(std::max_element(o, o + OUTPUT_SIZE) - o);
}

static Metrics evaluate(std::vector<Layer>& net, const Dataset& ds, const std::vector<uint32_t>& idx)
{
    Metrics m;
    int correct = 0;
    for (uint32_t i : idx) {
        int p = predict(net, ds.input(i));
        int t = ds.label(i);
        m.confusion[t][p]++;
        if (p == t) ++correct;
    }
    m.accuracy = 100.0f * correct / idx.size();

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

template <class T> static std::string join(const std::vector<T>& v, const char* sep = "|")
{
    std::ostringstream s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s << sep; s << v[i]; }
    return s.str();
}

int main(int argc, char** argv)
{
    Args a = parse_args(argc, argv);
    auto t0 = std::chrono::steady_clock::now();

    // --- data ------------------------------------------------------------
    if (a.quiet) std::cout.setstate(std::ios::failbit);         // silence loader chatter
    std::unique_ptr<Dataset> ds;
    std::vector<uint32_t> test_idx, val_idx, train_idx;

    if (a.dataset == "mnist") {
        std::string img_path = a.data + "/train-images-idx3-ubyte/train-images-idx3-ubyte";
        std::string lbl_path = a.data + "/train-labels-idx1-ubyte/train-labels-idx1-ubyte";
        ds.reset(new MnistDataset(img_path, lbl_path));         // everything in "train"

        // Fixed split: same permutation for every run, independent of --seed.
        uint32_t N = ds->size();
        std::vector<uint32_t> perm(N);
        std::iota(perm.begin(), perm.end(), 0u);
        { std::mt19937 split_rng(SPLIT_SEED); detu::shuffle(perm, split_rng); }
        test_idx .assign(perm.begin(),                  perm.begin() + N_TEST);
        val_idx  .assign(perm.begin() + N_TEST,         perm.begin() + N_TEST + N_VAL);
        train_idx.assign(perm.begin() + N_TEST + N_VAL, perm.end());
    } else {
        ds.reset(new Cifar10Dataset(a.data));

        // CIFAR-10 ships its own test set, so only train/val is drawn by permutation.
        // Same fixed SPLIT_SEED, so the split is identical for every run and every seed.
        std::vector<uint32_t> perm(CIFAR10_N_TRAIN);
        std::iota(perm.begin(), perm.end(), 0u);
        { std::mt19937 split_rng(SPLIT_SEED); detu::shuffle(perm, split_rng); }
        val_idx  .assign(perm.begin(),         perm.begin() + N_VAL);
        train_idx.assign(perm.begin() + N_VAL, perm.end());
        test_idx.resize(CIFAR10_N_TEST);
        std::iota(test_idx.begin(), test_idx.end(), CIFAR10_N_TRAIN);   // the official test batch
    }
    if (a.quiet) std::cout.clear();
    const int input_size = ds->input_size();

    // Clean training accuracy is measured on a fixed 10 000-image subset of the training
    // set rather than all 45 000. The subset is drawn once with SPLIT_SEED, so it is the
    // same images for every run, every seed and every configuration -- paired comparisons
    // are unaffected. Evaluating all 45 000 each epoch nearly doubled the cost of a run
    // for no extra precision: 10 000 images match the test set's own sample size, which is
    // what the generalisation gap is compared against.
    std::vector<uint32_t> train_eval_idx(train_idx);
    { std::mt19937 sub_rng(SPLIT_SEED + 1u); detu::shuffle(train_eval_idx, sub_rng); }
    if (train_eval_idx.size() > (size_t)N_TEST) train_eval_idx.resize(N_TEST);

    // --- network ---------------------------------------------------------
    std::mt19937 rng(a.seed);
    Act act1 = act_from_string(a.act1), act2 = act_from_string(a.act2);
    std::vector<Act> mask1 = make_mask(a.hidden, act1, act2, a.ratio, a.layout, rng);
    std::vector<Act> mask2 = make_mask(a.hidden, act1, act2, a.ratio, a.layout, rng);
    std::vector<Act> mask_out(OUTPUT_SIZE, a.loss == "ce" ? ACT_LINEAR : ACT_SIGMOID);

    std::vector<Layer> net;
    net.reserve(3);
    net.emplace_back(input_size, a.hidden, mask1, rng);
    net.emplace_back(a.hidden,  a.hidden, mask2, rng);
    net.emplace_back(a.hidden,  OUTPUT_SIZE, mask_out, rng);

    std::map<Act,int> count;
    for (Act x : mask1) count[x]++;
    std::cout << "version5 | " << a.dataset << " | " << a.act1 << "/" << a.act2 << " ratio=" << a.ratio << " layout=" << a.layout
              << " hidden=" << a.hidden << " seed=" << a.seed << " lr=" << a.lr << " epochs=" << a.epochs
              << " loss=" << a.loss << " | layer-1 mask:";
    for (auto& kv : count) std::cout << " " << act_to_string(kv.first) << "=" << kv.second;
    std::cout << " | train/val/test = " << train_idx.size() << "/" << val_idx.size() << "/" << test_idx.size() << "\n";

    // --- training --------------------------------------------------------
    std::vector<float> epoch_loss, epoch_val_acc, epoch_val_f1, epoch_train_acc, epoch_train_acc_clean;
    std::string status = "ok";
    float d_loss[OUTPUT_SIZE];

    // Early stopping: keep the parameters from the best-validation epoch and report the
    // test set under those, rather than under whatever the last epoch left behind. Without
    // this, a fixed epoch budget rewards whichever configuration happens to peak near the
    // end, and the hybrids converge faster than their parents.
    std::vector<std::vector<float>> best_W(net.size()), best_b(net.size());
    float best_val_seen = -1.0f;
    bool  have_snapshot = false;

    for (int epoch = 0; epoch < a.epochs && status == "ok"; ++epoch) {
        detu::shuffle(train_idx, rng);
        double total_loss = 0; int correct = 0;

        for (uint32_t i : train_idx) {
            const uint8_t label = ds->label(i);
            int p = predict(net, ds->input(i));
            const float* o = net.back().outputs();
            if (p == label) ++correct;

            if (a.loss == "ce") {
                float mx = *std::max_element(o, o + OUTPUT_SIZE), Z = 0;
                float prob[OUTPUT_SIZE];
                for (int j = 0; j < OUTPUT_SIZE; ++j) { prob[j] = std::exp(o[j] - mx); Z += prob[j]; }
                for (int j = 0; j < OUTPUT_SIZE; ++j) { prob[j] /= Z; d_loss[j] = prob[j] - (j == label ? 1.0f : 0.0f); }
                total_loss += -std::log(std::max(prob[label], 1e-12f));
            } else {
                float l = 0;
                for (int j = 0; j < OUTPUT_SIZE; ++j) {
                    float e = o[j] - (j == label ? 1.0f : 0.0f);
                    l += e * e; d_loss[j] = 2 * e;
                }
                total_loss += l / OUTPUT_SIZE;
            }
            if (std::isnan(o[0])) { status = "nan"; break; }

            net[2].backward(d_loss, a.lr, true);
            net[1].backward(net[2].d_inputs(), a.lr, true);
            net[0].backward(net[1].d_inputs(), a.lr, false);
        }

        // `tr_acc` accumulates while the weights are still moving, so it understates the
        // model that finished the epoch. `tr_clean` re-evaluates the finished model on the
        // whole training set, which is what a generalisation gap needs.
        float tr_acc = 100.0f * correct / train_idx.size();
        Metrics tr_clean = evaluate(net, *ds, train_eval_idx);
        Metrics v = evaluate(net, *ds, val_idx);
        epoch_loss.push_back(total_loss / train_idx.size());
        epoch_train_acc.push_back(tr_acc);
        epoch_train_acc_clean.push_back(tr_clean.accuracy);
        epoch_val_acc.push_back(v.accuracy);
        epoch_val_f1.push_back(v.macro_f1);

        if (v.accuracy > best_val_seen) {
            best_val_seen = v.accuracy;
            for (size_t L = 0; L < net.size(); ++L) net[L].save_state(best_W[L], best_b[L]);
            have_snapshot = true;
        }

        std::cout << "epoch " << epoch + 1 << "/" << a.epochs << "  loss=" << epoch_loss.back()
                  << "  train_acc=" << tr_acc << "  train_clean=" << tr_clean.accuracy
                  << "  val_acc=" << v.accuracy << "  val_f1=" << v.macro_f1 << "\n";
    }

    // --- final evaluation --------------------------------------------------
    // Reported twice: at the last epoch's parameters (test_acc, for continuity with the
    // earlier sweeps) and at the best-validation epoch's parameters (test_acc_best, the
    // early-stopping result the article should quote).
    Metrics val  = evaluate(net, *ds, val_idx);
    Metrics test = evaluate(net, *ds, test_idx);
    Metrics test_best = test;
    if (have_snapshot) {
        for (size_t L = 0; L < net.size(); ++L) net[L].load_state(best_W[L], best_b[L]);
        test_best = evaluate(net, *ds, test_idx);
    }
    int best_ep = epoch_val_acc.empty() ? 0 :
        (int)(std::max_element(epoch_val_acc.begin(), epoch_val_acc.end()) - epoch_val_acc.begin()) + 1;
    float best_val = epoch_val_acc.empty() ? 0.0f : *std::max_element(epoch_val_acc.begin(), epoch_val_acc.end());
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::cout << "RESULT status=" << status << " test_acc=" << test.accuracy << " test_f1=" << test.macro_f1
              << " test_acc_best=" << test_best.accuracy << " test_f1_best=" << test_best.macro_f1
              << " val_acc=" << val.accuracy << " (" << secs << " s)\n";

    // --- CSV -------------------------------------------------------------
    std::vector<int> conf, conf_best;
    for (int t = 0; t < OUTPUT_SIZE; ++t) for (int p = 0; p < OUTPUT_SIZE; ++p) conf.push_back(test.confusion[t][p]);
    for (int t = 0; t < OUTPUT_SIZE; ++t) for (int p = 0; p < OUTPUT_SIZE; ++p) conf_best.push_back(test_best.confusion[t][p]);

    bool need_header = true;
    { std::ifstream f(a.out); need_header = !(f.good() && f.peek() != std::ifstream::traits_type::eof()); }
    std::ostringstream row;
    if (need_header)
        // The first 23 columns are unchanged from the earlier sweeps, so every existing
        // reader keeps working; the early-stopping and clean-training columns are appended.
        row << "seed,act1,act2,ratio,layout,hidden,lr,epochs,loss,status,seconds,"
               "train_acc,val_acc,val_f1,test_acc,test_f1,best_val_acc,best_val_epoch,"
               "epoch_loss,epoch_train_acc,epoch_val_acc,epoch_val_f1,test_confusion,"
               "dataset,train_acc_clean,test_acc_best,test_f1_best,"
               "epoch_train_acc_clean,test_confusion_best\n";
    row << a.seed << "," << a.act1 << "," << a.act2 << "," << a.ratio << "," << a.layout << ","
        << a.hidden << "," << a.lr << "," << a.epochs << "," << a.loss << "," << status << "," << secs << ","
        << (epoch_train_acc.empty() ? 0.0f : epoch_train_acc.back()) << ","
        << val.accuracy << "," << val.macro_f1 << "," << test.accuracy << "," << test.macro_f1 << ","
        << best_val << "," << best_ep << ","
        << join(epoch_loss) << "," << join(epoch_train_acc) << "," << join(epoch_val_acc) << ","
        << join(epoch_val_f1) << "," << join(conf) << ","
        << a.dataset << ","
        << (epoch_train_acc_clean.empty() ? 0.0f : epoch_train_acc_clean.back()) << ","
        << test_best.accuracy << "," << test_best.macro_f1 << ","
        << join(epoch_train_acc_clean) << "," << join(conf_best) << "\n";
    std::ofstream out(a.out, std::ios::app);
    if (!out) { std::cerr << "cannot open " << a.out << "\n"; return 1; }
    out << row.str();
    return status == "ok" ? 0 : 2;
}

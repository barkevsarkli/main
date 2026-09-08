// version5 -- hybrid-activation MLP experiment runner (MNIST)
//
// Differences from v2/v3 (see README):
//   * per-neuron activation masks (ratio + layout) instead of a hardcoded even/odd split
//   * output layer is uniform (linear+softmax for CE, sigmoid for MSE); hidden layers
//     are the only place hybridisation happens, so the effect is isolated
//   * consistent leaky_relu slope in forward and backward
//   * fixed, seed-independent train/val/test split (45k/5k/10k) shared by every run
//   * per-epoch shuffling of the training order (seeded)
//   * accuracy, macro-F1 and confusion matrix on validation (every epoch) and test (end)
//   * learning rate, hidden size, epochs, loss, data path all on the CLI
//   * std::mt19937 instead of rand() so seeds mean the same thing on every platform
//
// Usage:
//   ./main --act1 tanh --act2 relu --ratio 0.5 --layout interleave --hidden 12
//     --seed 50 --epochs 10 --lr 0.01 --loss ce --data ../ --out results.csv
//
// ratio is the fraction of hidden neurons that use act2.  ratio 0 -> homogeneous act1.

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
#include "data.h"
#include "net.h"

#define INPUT_SIZE 784
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
        else if (k == "--data")   a.data   = need("--data");
        else if (k == "--out")    a.out    = need("--out");
        else if (k == "--quiet")  a.quiet  = true;
        else { std::cerr << "unknown argument: " << k << "\n"; exit(1); }
    }
    if (a.loss != "ce" && a.loss != "mse") { std::cerr << "--loss must be ce or mse\n"; exit(1); }
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

static Metrics evaluate(std::vector<Layer>& net, Image* imgs, const std::vector<uint32_t>& idx)
{
    Metrics m;
    int correct = 0;
    for (uint32_t i : idx) {
        int p = predict(net, imgs[i].normalized_pixels);
        int t = imgs[i].label;
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
    std::string img_path = a.data + "/train-images-idx3-ubyte/train-images-idx3-ubyte";
    std::string lbl_path = a.data + "/train-labels-idx1-ubyte/train-labels-idx1-ubyte";
    if (a.quiet) std::cout.setstate(std::ios::failbit);         // silence loader chatter
    Data data(img_path, lbl_path, 0.0f);                           // everything in "train"
    if (a.quiet) std::cout.clear();
    Image* imgs = data.get_train_images();
    uint32_t N = data.get_train_size();

    // Fixed split: same permutation for every run, independent of --seed.
    std::vector<uint32_t> perm(N);
    std::iota(perm.begin(), perm.end(), 0u);
    { std::mt19937 split_rng(SPLIT_SEED); std::shuffle(perm.begin(), perm.end(), split_rng); }
    std::vector<uint32_t> test_idx (perm.begin(),                 perm.begin() + N_TEST);
    std::vector<uint32_t> val_idx  (perm.begin() + N_TEST,        perm.begin() + N_TEST + N_VAL);
    std::vector<uint32_t> train_idx(perm.begin() + N_TEST + N_VAL, perm.end());

    // --- network ---------------------------------------------------------
    std::mt19937 rng(a.seed);
    Act act1 = act_from_string(a.act1), act2 = act_from_string(a.act2);
    std::vector<Act> mask1 = make_mask(a.hidden, act1, act2, a.ratio, a.layout, rng);
    std::vector<Act> mask2 = make_mask(a.hidden, act1, act2, a.ratio, a.layout, rng);
    std::vector<Act> mask_out(OUTPUT_SIZE, a.loss == "ce" ? ACT_LINEAR : ACT_SIGMOID);

    std::vector<Layer> net;
    net.reserve(3);
    net.emplace_back(INPUT_SIZE, a.hidden, mask1, rng);
    net.emplace_back(a.hidden,  a.hidden, mask2, rng);
    net.emplace_back(a.hidden,  OUTPUT_SIZE, mask_out, rng);

    std::map<Act,int> count;
    for (Act x : mask1) count[x]++;
    std::cout << "version5 | " << a.act1 << "/" << a.act2 << " ratio=" << a.ratio << " layout=" << a.layout
              << " hidden=" << a.hidden << " seed=" << a.seed << " lr=" << a.lr << " epochs=" << a.epochs
              << " loss=" << a.loss << " | layer-1 mask:";
    for (auto& kv : count) std::cout << " " << act_to_string(kv.first) << "=" << kv.second;
    std::cout << " | train/val/test = " << train_idx.size() << "/" << val_idx.size() << "/" << test_idx.size() << "\n";

    // --- training --------------------------------------------------------
    std::vector<float> epoch_loss, epoch_val_acc, epoch_val_f1, epoch_train_acc;
    std::string status = "ok";
    float d_loss[OUTPUT_SIZE];

    for (int epoch = 0; epoch < a.epochs && status == "ok"; ++epoch) {
        std::shuffle(train_idx.begin(), train_idx.end(), rng);
        double total_loss = 0; int correct = 0;

        for (uint32_t i : train_idx) {
            const Image& im = imgs[i];
            int p = predict(net, im.normalized_pixels);
            const float* o = net.back().outputs();
            if (p == im.label) ++correct;

            if (a.loss == "ce") {
                float mx = *std::max_element(o, o + OUTPUT_SIZE), Z = 0;
                float prob[OUTPUT_SIZE];
                for (int j = 0; j < OUTPUT_SIZE; ++j) { prob[j] = std::exp(o[j] - mx); Z += prob[j]; }
                for (int j = 0; j < OUTPUT_SIZE; ++j) { prob[j] /= Z; d_loss[j] = prob[j] - im.one_hot_encoded_label[j]; }
                total_loss += -std::log(std::max(prob[im.label], 1e-12f));
            } else {
                float l = 0;
                for (int j = 0; j < OUTPUT_SIZE; ++j) {
                    float e = o[j] - im.one_hot_encoded_label[j];
                    l += e * e; d_loss[j] = 2 * e;
                }
                total_loss += l / OUTPUT_SIZE;
            }
            if (std::isnan(o[0])) { status = "nan"; break; }

            net[2].backward(d_loss, a.lr, true);
            net[1].backward(net[2].d_inputs(), a.lr, true);
            net[0].backward(net[1].d_inputs(), a.lr, false);
        }

        float tr_acc = 100.0f * correct / train_idx.size();
        Metrics v = evaluate(net, imgs, val_idx);
        epoch_loss.push_back(total_loss / train_idx.size());
        epoch_train_acc.push_back(tr_acc);
        epoch_val_acc.push_back(v.accuracy);
        epoch_val_f1.push_back(v.macro_f1);
        std::cout << "epoch " << epoch + 1 << "/" << a.epochs << "  loss=" << epoch_loss.back()
                  << "  train_acc=" << tr_acc << "  val_acc=" << v.accuracy << "  val_f1=" << v.macro_f1 << "\n";
    }

    // --- final evaluation --------------------------------------------------
    Metrics val  = evaluate(net, imgs, val_idx);
    Metrics test = evaluate(net, imgs, test_idx);
    int best_ep = epoch_val_acc.empty() ? 0 :
        (int)(std::max_element(epoch_val_acc.begin(), epoch_val_acc.end()) - epoch_val_acc.begin()) + 1;
    float best_val = epoch_val_acc.empty() ? 0.0f : *std::max_element(epoch_val_acc.begin(), epoch_val_acc.end());
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::cout << "RESULT status=" << status << " test_acc=" << test.accuracy << " test_f1=" << test.macro_f1
              << " val_acc=" << val.accuracy << " (" << secs << " s)\n";

    // --- CSV -------------------------------------------------------------
    std::vector<int> conf;
    for (int t = 0; t < OUTPUT_SIZE; ++t) for (int p = 0; p < OUTPUT_SIZE; ++p) conf.push_back(test.confusion[t][p]);

    bool need_header = true;
    { std::ifstream f(a.out); need_header = !(f.good() && f.peek() != std::ifstream::traits_type::eof()); }
    std::ostringstream row;
    if (need_header)
        row << "seed,act1,act2,ratio,layout,hidden,lr,epochs,loss,status,seconds,"
               "train_acc,val_acc,val_f1,test_acc,test_f1,best_val_acc,best_val_epoch,"
               "epoch_loss,epoch_train_acc,epoch_val_acc,epoch_val_f1,test_confusion\n";
    row << a.seed << "," << a.act1 << "," << a.act2 << "," << a.ratio << "," << a.layout << ","
        << a.hidden << "," << a.lr << "," << a.epochs << "," << a.loss << "," << status << "," << secs << ","
        << (epoch_train_acc.empty() ? 0.0f : epoch_train_acc.back()) << ","
        << val.accuracy << "," << val.macro_f1 << "," << test.accuracy << "," << test.macro_f1 << ","
        << best_val << "," << best_ep << ","
        << join(epoch_loss) << "," << join(epoch_train_acc) << "," << join(epoch_val_acc) << ","
        << join(epoch_val_f1) << "," << join(conf) << "\n";
    std::ofstream out(a.out, std::ios::app);
    if (!out) { std::cerr << "cannot open " << a.out << "\n"; return 1; }
    out << row.str();
    return status == "ok" ? 0 : 2;
}

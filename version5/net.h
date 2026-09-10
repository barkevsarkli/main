#ifndef NET_H
#define NET_H

#include <string>
#include <vector>
#include <random>

// Per-neuron activation functions.  Every neuron in a layer carries its own
// Act value, so a layer can mix functions in any pattern (even/odd, blocks,
// random masks, or anything an assignment method produces later).
enum Act { ACT_RELU = 0, ACT_LEAKY_RELU = 1, ACT_TANH = 2, ACT_SIGMOID = 3, ACT_LINEAR = 4 };

Act         act_from_string(const std::string& s);
std::string act_to_string(Act a);

// Slope used by leaky ReLU on the negative side.  The v2/v3 code used 0.1 on
// even neurons and 0.01 on odd neurons in forward, but 0.1 for both in
// backward.  version5 uses a single constant everywhere.
#define LEAKY_SLOPE 0.1f

// Element-wise clip on d(loss)/d(pre-activation), kept from v2 as a safety
// net against exploding gradients.  Rarely active with a sane learning rate.
#define GRAD_CLIP 10.0f

class Layer
{
public:
    Layer(int n_in, int n_out, const std::vector<Act>& acts, std::mt19937& rng);

    void forward(const float* in);
    // Batch-size-1 SGD: computes d_inputs with the *current* weights, then
    // applies the update in the same pass (no separate gradient buffers).
    // If compute_d_inputs is false (first hidden layer) the d_inputs pass is
    // skipped, which roughly halves the cost of that layer's backward step.
    void backward(const float* d_values, float lr, bool compute_d_inputs = true);

    // Weight snapshot, for early stopping: the run keeps the parameters from the epoch
    // with the best validation accuracy and evaluates the test set with those, instead of
    // whatever the last epoch happened to leave behind.
    void save_state(std::vector<float>& w_out, std::vector<float>& b_out) const { w_out = W; b_out = b; }
    void load_state(const std::vector<float>& w_in, const std::vector<float>& b_in) { W = w_in; b = b_in; }

    const float* outputs()  const { return out.data(); }
    const float* d_inputs() const { return d_in.data(); }
    int size() const { return n_out; }
    const std::vector<Act>& activations() const { return act; }

private:
    int n_in, n_out;
    std::vector<float> W, b, out, d_in;
    std::vector<Act>   act;
    const float* in;
};

// Build a per-neuron mask for a hidden layer of size n.
//   ratio  = fraction of neurons that receive act2 (0 -> all act1, 1 -> all act2)
//   layout = "interleave" (act2 spread evenly; ratio 0.5 == v2's even/odd),
//            "block"      (act1 first, act2 last),
//            "random"     (seeded shuffle of the block mask)
std::vector<Act> make_mask(int n, Act act1, Act act2, float ratio,
                           const std::string& layout, std::mt19937& rng);

#endif

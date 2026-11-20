#include <algorithm>
#include <cstddef>
#include <vector>

// One integer label replaces each sparse one-hot row. Quantities for every
// candidate cluster are accumulated in one scan of the current probability row.
extern "C" void assign_mse(const double* probability, int n, int sweeps,
                           int initial_slots, int* labels) {
    std::fill(labels, labels + n, -1);
    int slots = initial_slots;
    std::vector<int> counts(slots, 0);
    std::vector<double> column_sum(slots, 0.0);
    for (int it = 0; it < sweeps; ++it) {
        for (int i = 0; i < n; ++i) {
            if (labels[i] >= 0) --counts[labels[i]];
            labels[i] = -1;
            const double* row = probability + static_cast<std::size_t>(i) * n;
            std::fill(column_sum.begin(), column_sum.end(), 0.0);
            for (int j = 0; j < n; ++j) {
                int label = labels[j];
                if (label >= 0) column_sum[label] += row[j];
            }

            int best = 0;
            double best_score = counts[0] - 2.0 * column_sum[0];
            for (int k = 1; k < slots; ++k) {
                double score = counts[k] - 2.0 * column_sum[k];
                if (score < best_score) {
                    best_score = score;
                    best = k;
                }
            }
            if (2.0 * best_score + 1.0 - 2.0 * row[i] < 0.0) {
                labels[i] = best;
                ++counts[best];
            }
            if (best == slots - 1) {
                slots *= 2;
                counts.resize(slots, 0);
                column_sum.resize(slots, 0.0);
            }
        }
    }
}

// SPDX-License-Identifier: Apache-2.0
// Independent glue; linked CGAL terms remain GPL-3.0-or-later OR commercial.
// Independent exact-arithmetic control, not a production geometry query.
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/number_utils.h>
#include <CGAL/version.h>
#include <iostream>
#include <string>
#include <vector>

#ifdef __FAST_MATH__
#error Exact-arithmetic control forbids fast math
#endif
static_assert(CGAL_VERSION_NR == 1060011000, "Require pinned CGAL 6.0.1");
using FT = CGAL::Exact_predicates_exact_constructions_kernel::FT;
constexpr std::size_t TERMS = 131072;

static void phase(const std::string& arm, const char* name, bool exact_zero = false) {
    std::cout << "{\"arm\":\"" << arm << "\",\"terms\":" << TERMS
              << ",\"phase\":\"" << name << "\",\"exact_zero_verified\":"
              << (exact_zero ? "true" : "false") << "}" << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string arm = argv[1];
    if (arm != "sequential" && arm != "balanced") return 2;
    phase(arm, "before_build");
    {
        std::vector<FT> terms; terms.reserve(TERMS);
        for (std::size_t i = 0; i < TERMS; ++i) terms.emplace_back(i % 2 ? -1 : 1);
        FT sum(0);
        if (arm == "sequential") {
            for (const FT& term : terms) sum += term;
        } else {
            while (terms.size() > 1) {
                std::vector<FT> next; next.reserve(terms.size() / 2);
                for (std::size_t i = 0; i < terms.size(); i += 2)
                    next.emplace_back(terms[i] + terms[i + 1]);
                terms.swap(next);
            }
            sum = terms[0];
        }
        phase(arm, "before_exact_force");
        // Force the lazy DAG, rather than accepting an interval-filtered == 0.
        const bool zero = CGAL::exact(sum) == 0;
        if (!zero) return 3;
        phase(arm, "exact_force_complete", true);
        phase(arm, "before_teardown", true);
    }
    phase(arm, "teardown_complete", true);
    return 0;
}

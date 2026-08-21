#pragma once

#include <vector>

namespace midigpt::sampling {

struct SelectionMask {
    std::vector<std::vector<bool>> selected;   // [track][bar] — true = generate here
    std::vector<bool> autoregressive;          // per track
    std::vector<bool> ignore;                  // per track — excluded from context
    // Per track: true = this track's selected bars are Humanize targets
    // (skeleton pitch/duration fixed, only Velocity/Delta regenerated) rather
    // than plain AR/infill generation. Takes priority over `autoregressive`
    // for tracks where both would otherwise apply.
    std::vector<bool> humanize;
};

} // namespace midigpt::sampling

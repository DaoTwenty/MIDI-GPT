#include "grammar_constraint.h"

namespace midigpt::masking {

void GrammarConstraint::step(int token, const tokenizer::Vocabulary& vocab) {
    auto [type, value] = vocab.decode(token);
    current_state_ = type;

    switch (type) {
        case TokenType::Track:
            track_count_++;
            is_drum_ = (value == 1);
            bar_count_ = 0;
            break;
        case TokenType::Bar:
            bar_count_++;
            in_bar_ = true;
            timestep_ = -1;
            beat_length_ = 4;
            has_notes_in_block_ = false;
            break;
        case TokenType::TimeSig:
            if (vocab.config().time_signatures) {
                auto [num, den] = vocab.config().time_signatures->decode(value);
                beat_length_ = 4 * num / den;
            }
            break;
        case TokenType::BarEnd:
            in_bar_ = false;
            break;
        case TokenType::TimeAbsolutePos:
            timestep_ = value;
            break;
        case TokenType::FillInStart:
            in_infill_ = true;
            has_notes_in_block_ = false;
            timestep_ = -1; // Reset time to start of fill block
            break;
        case TokenType::FillInEnd:
            in_infill_ = false;
            break;
        case TokenType::HumanizeStart:
            in_humanize_ = true;
            humanize_groups_done_ = 0;
            break;
        case TokenType::HumanizeEnd:
            in_humanize_ = false;
            break;
        case TokenType::HumanizeSkeletonStart:
            in_humanize_skeleton_ = true;
            break;
        case TokenType::HumanizeSkeletonEnd:
            in_humanize_skeleton_ = false;
            break;
        case TokenType::VelocityLevel:
            if (in_humanize_) humanize_groups_done_++;
            break;
        case TokenType::NoteOnset:
        case TokenType::NotePitch:
            has_notes_in_block_ = true;
            break;
        default:
            break;
    }
}

void GrammarConstraint::apply(std::vector<bool>& mask, const tokenizer::Vocabulary& vocab) const {
    auto allow = [&](TokenType type) {
        auto [start, end] = vocab.range(type);
        if (start != -1) {
            for (int i = start; i < end; ++i) mask[i] = false;
        }
    };

    // Default: everything disallowed
    std::fill(mask.begin(), mask.end(), true);

    switch (current_state_) {

        case TokenType::PieceStart:
            allow(TokenType::NumBars);
            allow(TokenType::Track);
            break;

        case TokenType::NumBars:
            allow(TokenType::Track);
            break;

        case TokenType::Track:
            allow(TokenType::Instrument);
            if (exact_bars_ < 0) {
                allow(TokenType::Bar);
                allow(TokenType::TrackEnd);
            } else if (bar_count_ < exact_bars_) {
                allow(TokenType::Bar);
            } else {
                allow(TokenType::TrackEnd);
            }
            break;

        case TokenType::Instrument:
            allow(TokenType::NoteDensity);
            allow(TokenType::MinPolyphony);
            allow(TokenType::MaxPolyphony);
            allow(TokenType::MinNoteDuration);
            allow(TokenType::MaxNoteDuration);
            allow(TokenType::OnsetPolyphony);
            allow(TokenType::PitchRange);
            if (exact_bars_ < 0) {
                allow(TokenType::Bar);
                allow(TokenType::TrackEnd);
            } else if (bar_count_ < exact_bars_) {
                allow(TokenType::Bar);
            } else {
                allow(TokenType::TrackEnd);
            }
            break;

        // After any track-level attribute, allow more attrs or bar
        case TokenType::NoteDensity:
        case TokenType::MinPolyphony:
        case TokenType::MaxPolyphony:
        case TokenType::MinNoteDuration:
        case TokenType::MaxNoteDuration:
        case TokenType::OnsetPolyphony:
        case TokenType::PitchRange:
            allow(TokenType::NoteDensity);
            allow(TokenType::MinPolyphony);
            allow(TokenType::MaxPolyphony);
            allow(TokenType::MinNoteDuration);
            allow(TokenType::MaxNoteDuration);
            allow(TokenType::OnsetPolyphony);
            allow(TokenType::PitchRange);
            if (exact_bars_ < 0) {
                allow(TokenType::Bar);
                allow(TokenType::TrackEnd);
            } else if (bar_count_ < exact_bars_) {
                allow(TokenType::Bar);
            } else {
                allow(TokenType::TrackEnd);
            }
            break;

        case TokenType::Bar:
            allow(TokenType::TimeSig);
            allow(TokenType::Tension);
            allow(TokenType::PitchClassSet);
            allow(TokenType::BarLevelVelocityRange);
            allow(TokenType::MaskBar);
            allow(TokenType::TimeAbsolutePos);
            // Encoder skips TimeAbsolutePos when onset==0, so allow direct
            // entry into the note tokens at bar start (position 0).
            allow(TokenType::VelocityLevel);
            allow(TokenType::NoteOnset);
            allow(TokenType::NotePitch);
            allow(TokenType::DeltaDirection);
            allow(TokenType::Delta);
            if (!require_notes_ || has_notes_in_block_) allow(TokenType::BarEnd);
            allow(TokenType::FillInPlaceholder);
            allow(TokenType::HumanizeSkeletonStart);
            break;

        case TokenType::TimeSig:
            allow(TokenType::Tension);
            allow(TokenType::PitchClassSet);
            allow(TokenType::BarLevelVelocityRange);
            allow(TokenType::MaskBar);
            allow(TokenType::TimeAbsolutePos);
            allow(TokenType::VelocityLevel);
            allow(TokenType::NoteOnset);
            allow(TokenType::NotePitch);
            allow(TokenType::DeltaDirection);
            allow(TokenType::Delta);
            if (!require_notes_ || has_notes_in_block_) allow(TokenType::BarEnd);
            allow(TokenType::FillInPlaceholder);
            allow(TokenType::HumanizeSkeletonStart);
            break;

        case TokenType::Tension:
        case TokenType::PitchClassSet:
        case TokenType::BarLevelVelocityRange:
            allow(TokenType::Tension);
            allow(TokenType::PitchClassSet);
            allow(TokenType::BarLevelVelocityRange);
            allow(TokenType::MaskBar);
            allow(TokenType::TimeAbsolutePos);
            allow(TokenType::VelocityLevel);
            allow(TokenType::NoteOnset);
            allow(TokenType::NotePitch);
            allow(TokenType::DeltaDirection);
            allow(TokenType::Delta);
            if (!require_notes_ || has_notes_in_block_) allow(TokenType::BarEnd);
            allow(TokenType::HumanizeSkeletonStart);
            break;

        case TokenType::MaskBar:
            allow(TokenType::BarEnd);
            break;

        case TokenType::FillInPlaceholder:
            allow(TokenType::BarEnd);
            break;

        case TokenType::BarEnd:
            // Exact-bar enforcement for autoregressive: only allow track/piece
            // end when we have generated exactly the requested number of bars.
            allow(TokenType::FillInStart);
            if (exact_bars_ < 0) {
                allow(TokenType::TrackEnd);
                allow(TokenType::PieceEnd);
                allow(TokenType::Bar);
            } else if (bar_count_ < exact_bars_) {
                allow(TokenType::Bar);
            } else {
                allow(TokenType::TrackEnd);
                allow(TokenType::PieceEnd);
            }
            break;

        case TokenType::TimeAbsolutePos:
            if (in_humanize_skeleton_) {
                // Skeleton notes carry no Velocity/Delta — those are withheld
                // and arrive later via the appendix.
                allow(TokenType::NoteOnset);
                allow(TokenType::NotePitch);
                if (!require_notes_ || has_notes_in_block_) allow(TokenType::HumanizeSkeletonEnd);
            } else {
                // After time position: velocity may change, or carry from previous
                // VelocityLevel — orig allows TimeAbsolutePos -> NoteOnset directly.
                allow(TokenType::VelocityLevel);
                allow(TokenType::NoteOnset);
                allow(TokenType::NotePitch);
                allow(TokenType::DeltaDirection);
                allow(TokenType::Delta);
                if (in_infill_) {
                    if (!require_notes_ || has_notes_in_block_) allow(TokenType::FillInEnd);
                } else {
                    if (!require_notes_ || has_notes_in_block_) allow(TokenType::BarEnd);
                }
            }
            break;

        case TokenType::VelocityLevel:
            if (in_humanize_) {
                // Humanize per-note group: VelocityLevel [DeltaDirection] [Delta],
                // then either the next note's VelocityLevel or, once every
                // fixed note has a group, HumanizeEnd.
                allow(TokenType::DeltaDirection);
                allow(TokenType::Delta);
                if (humanize_groups_done_ < humanize_total_notes_) {
                    allow(TokenType::VelocityLevel);
                } else {
                    allow(TokenType::HumanizeEnd);
                }
            } else {
                // orig: VelocityLevel -> {NoteOnset, Delta}
                allow(TokenType::NoteOnset);
                allow(TokenType::NotePitch);
                allow(TokenType::Delta);
            }
            break;

        case TokenType::DeltaDirection:
            allow(TokenType::Delta);
            break;

        case TokenType::Delta:
            if (in_humanize_) {
                if (humanize_groups_done_ < humanize_total_notes_) {
                    allow(TokenType::VelocityLevel);
                } else {
                    allow(TokenType::HumanizeEnd);
                }
            } else {
                // orig: Delta -> {Delta, DeltaDirection, NoteOnset, FillInEnd}
                allow(TokenType::Delta);
                allow(TokenType::DeltaDirection);
                allow(TokenType::NoteOnset);
                allow(TokenType::NotePitch);
                if (in_infill_) {
                    if (!require_notes_ || has_notes_in_block_) allow(TokenType::FillInEnd);
                }
            }
            break;

        case TokenType::NoteOnset:
        case TokenType::NotePitch:
            if (is_drum_) {
                // Drums: no duration token, go to next note or end
                if (in_humanize_skeleton_) {
                    allow(TokenType::NoteOnset);
                    allow(TokenType::NotePitch);
                    allow(TokenType::TimeAbsolutePos);
                    allow(TokenType::HumanizeSkeletonEnd);
                } else {
                    allow(TokenType::VelocityLevel);
                    allow(TokenType::NoteOnset);
                    allow(TokenType::NotePitch);
                    allow(TokenType::TimeAbsolutePos);
                    allow(TokenType::DeltaDirection);
                    allow(TokenType::Delta);
                    if (in_infill_) {
                        allow(TokenType::FillInEnd);
                    } else {
                        allow(TokenType::BarEnd);
                    }
                }
            } else {
                // Melodic: must have duration next
                allow(TokenType::NoteDuration);
            }
            break;

        case TokenType::NoteDuration:
            // After duration: another note (same onset), new time, or end
            if (in_humanize_skeleton_) {
                allow(TokenType::NoteOnset);
                allow(TokenType::NotePitch);
                allow(TokenType::TimeAbsolutePos);
                allow(TokenType::HumanizeSkeletonEnd);
            } else {
                allow(TokenType::VelocityLevel);
                allow(TokenType::NoteOnset);
                allow(TokenType::NotePitch);
                allow(TokenType::TimeAbsolutePos);
                allow(TokenType::DeltaDirection);
                allow(TokenType::Delta);
                if (in_infill_) {
                    allow(TokenType::FillInEnd);
                } else {
                    allow(TokenType::BarEnd);
                }
            }
            break;

        case TokenType::TrackEnd:
            allow(TokenType::PieceEnd);
            if (max_tracks_ < 0 || track_count_ < max_tracks_) {
                allow(TokenType::Track);
            }
            break;

        case TokenType::FillInStart:
            // Encoder skips TimeAbsolutePos when onset==0 (same as Bar), so
            // note tokens must be reachable directly. FillInEnd is withheld
            // here (and by require_notes_ downstream) to prevent empty fills.
            allow(TokenType::TimeAbsolutePos);
            allow(TokenType::VelocityLevel);
            allow(TokenType::NoteOnset);
            allow(TokenType::NotePitch);
            allow(TokenType::DeltaDirection);
            allow(TokenType::Delta);
            break;

        case TokenType::FillInEnd:
            allow(TokenType::BarEnd);
            allow(TokenType::FillInStart);
            allow(TokenType::HumanizeStart);
            allow(TokenType::PieceEnd);
            break;

        case TokenType::HumanizeStart:
            // Mirrors FillInStart, but the appendix here only ever carries
            // expressive tokens (no TimeAbsolutePos/NoteOnset/NoteDuration) —
            // each group starts with VelocityLevel. An empty block (0 notes)
            // goes straight to HumanizeEnd.
            if (humanize_total_notes_ > 0) {
                allow(TokenType::VelocityLevel);
            } else {
                allow(TokenType::HumanizeEnd);
            }
            break;

        case TokenType::HumanizeEnd:
            allow(TokenType::HumanizeStart);
            allow(TokenType::PieceEnd);
            break;

        case TokenType::HumanizeSkeletonStart:
            // In-place skeleton: pitch/onset/duration only, no Velocity/Delta
            // (those are withheld and arrive later via the appendix).
            allow(TokenType::TimeAbsolutePos);
            allow(TokenType::NoteOnset);
            allow(TokenType::NotePitch);
            if (!require_notes_ || has_notes_in_block_) allow(TokenType::HumanizeSkeletonEnd);
            break;

        case TokenType::HumanizeSkeletonEnd:
            allow(TokenType::BarEnd);
            break;

        default:
            // Safety fallback: allow everything
            std::fill(mask.begin(), mask.end(), false);
            break;
    }

    // TimeAbsolutePos: enforce monotonicity and bar boundary
    if (in_bar_ && vocab.has(TokenType::TimeAbsolutePos)) {
        auto [tap_start, tap_end] = vocab.range(TokenType::TimeAbsolutePos);
        if (tap_start != -1) {
            int bar_ticks = beat_length_ * vocab.config().resolution;
            for (int i = tap_start; i < tap_end; ++i) {
                auto [_, pos] = vocab.decode(i);
                if (pos <= timestep_ || pos >= bar_ticks) {
                    mask[i] = true;
                }
            }
        }
    }

    // Autoregressive mode: forbid every FillIn-*/Humanize-* token everywhere.
    // The encoder turns off multi-fill/multi-humanize in AR mode (no
    // FillInPlaceholder and no withheld velocity/delta in the prompt), so the
    // model has no business emitting these appendix-block markers.
    if (is_autoregressive_) {
        for (TokenType t : {TokenType::FillInStart, TokenType::FillInEnd,
                            TokenType::FillInPlaceholder,
                            TokenType::HumanizeStart, TokenType::HumanizeEnd,
                            TokenType::HumanizeSkeletonStart, TokenType::HumanizeSkeletonEnd}) {
            auto [s, e] = vocab.range(t);
            if (s != -1) for (int i = s; i < e; ++i) mask[i] = true;
        }
    }

    // Mask track start/end if requested (single-track generation)
    if (mask_track_start_) {
        auto [start, end] = vocab.range(TokenType::Track);
        if (start != -1) for (int i = start; i < end; ++i) mask[i] = true;
    }
    if (mask_track_end_) {
        auto [start, end] = vocab.range(TokenType::TrackEnd);
        if (start != -1) for (int i = start; i < end; ++i) mask[i] = true;
    }
}

} // namespace midigpt::masking

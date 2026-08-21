#pragma once

namespace midigpt {

enum class TokenType {
    PieceStart = 0,
    NoteOnset = 1,
    NoteOffset = 2,
    NotePitch = 3,
    NonPitch = 4,
    Velocity = 5,
    TimeDelta = 6,
    TimeAbsolutePos = 7,
    Instrument = 8,
    Bar = 9,
    BarEnd = 10,
    Track = 11,
    TrackEnd = 12,
    DrumTrack = 13,
    FillIn = 14,
    FillInPlaceholder = 15,
    FillInStart = 16,
    FillInEnd = 17,
    Header = 18,
    VelocityLevel = 19,
    Genre = 20,
    NoteDensity = 21,       // TOKEN_DENSITY_LEVEL
    TimeSig = 22,           // TOKEN_TIME_SIGNATURE
    Segment = 23,
    SegmentEnd = 24,
    SegmentFillIn = 25,
    NoteDuration = 26,
    AvPolyphony = 27,
    MinPolyphony = 28,
    MaxPolyphony = 29,
    MinNoteDuration = 30,
    MaxNoteDuration = 31,
    NumBars = 32,
    MinPolyphonyHard = 33,
    MaxPolyphonyHard = 34,
    MinNoteDurationHard = 35,
    MaxNoteDurationHard = 36,
    RestPercentage = 37,
    PitchClass = 38,
    PitchClassCount = 39,
    BarLevelOnsetDensity = 40,
    BarLevelOnsetPolyphonyMin = 41,
    BarLevelOnsetPolyphonyMax = 42,
    TrackLevelOnsetDensity = 43,
    TrackLevelOnsetPolyphonyMin = 44,
    TrackLevelOnsetPolyphonyMax = 45,
    TrackLevelOnsetDensityMin = 46,
    TrackLevelOnsetDensityMax = 47,
    TrackLevelPitchRangeMin = 48,
    TrackLevelPitchRangeMax = 49,
    KeySignature = 50,
    BarLevelPitchClassSet = 51,
    TrackLevelSilenceProportionMin = 52,
    TrackLevelSilenceProportionMax = 53,
    ValenceSpotify = 54,
    EnergySpotify = 55,
    DanceabilitySpotify = 56,
    Danceability = 57,
    Tension = 58,           // TOKEN_BAR_LEVEL_TENSION
    ContainsNoteDurationThirtySecond = 59,
    ContainsNoteDurationSixteenth = 60,
    ContainsNoteDurationEighth = 61,
    ContainsNoteDurationQuarter = 62,
    ContainsNoteDurationHalf = 63,
    ContainsNoteDurationWhole = 64,
    WnbdSyncopation = 65,
    Repetition = 66,
    Delta = 68,
    DeltaDirection = 69,
    None = 70,
    MaskBar = 71,
    TensionDrum = 77,       // TOKEN_BAR_LEVEL_TENSION_DRUM

    // Piece-level switchable mode tokens (future models trained on both modes).
    UseVelocity    = 86,    // 0 = no velocity tokens, 1 = use velocity tokens
    UseMicrotiming = 87,    // 0 = no delta tokens,    1 = use delta tokens

    // Track-level expressiveness control (NOMML median metric depth, 0-12).
    TrackLevelNomml = 88,

    // Humanize scope markers: bracket an appendix block (after TrackEnd) that
    // carries the real Velocity/Delta tokens for a bar/track whose note
    // skeleton (pitch/onset/duration) was already emitted in place without
    // them. Mirrors FillInStart/FillInEnd's placeholder-in-place +
    // appendix-in-order pattern, but for expressive tokens instead of whole
    // bars. Appended at the end of the enum (never insert mid-range) so
    // existing checkpoints' token IDs never shift.
    HumanizeStart = 89,
    HumanizeEnd = 90,

    // Piece-level flag (mirrors how PieceStart's domain signals "has infill
    // blocks", but as its own token like UseVelocity/UseMicrotiming rather
    // than bit-packed into PieceStart): 1 = this piece has one or more
    // Humanize appendix blocks after TrackEnd, 0 = it doesn't.
    HumanizeActive = 91,

    // In-place skeleton brackets for a Humanize bar: the note skeleton
    // (pitch/onset/duration) is emitted normally between these markers, with
    // Velocity/Delta withheld there (they arrive later via the
    // HumanizeStart/HumanizeEnd appendix block). Distinct token types from
    // HumanizeStart/HumanizeEnd — which mark only the appendix — so the two
    // roles are never ambiguous by token id alone, and no positional
    // heuristics (e.g. "before/after the last TrackEnd") are needed to tell
    // them apart in the grammar, session state, or decoder.
    HumanizeSkeletonStart = 92,
    HumanizeSkeletonEnd = 93,

    // Bar-level dynamic-range-of-velocity attribute control (max - min
    // velocity among the bar's notes). Python-defined attribute class
    // (attributes/velocity.py); token_type/size are wired in dynamically via
    // EncoderConfig::add_attribute_token_domains(), same as every other
    // attribute control — this enum value only reserves the token id.
    BarLevelVelocityRange = 94,

    // Aliases for refactor names if needed
    OnsetPolyphony = 42,    // Use MaxPolyphony as alias for OnsetPolyphony
    PitchRange = 49,
    NoteDurationDist = 26,
    SilenceProportion = 53,
    PitchClassSet = 51,
    PieceEnd = 70           // Use None for PieceEnd if not present
};

enum class TrackType { Melodic = 0, Drum = 1 };

// Used by attribute constraints: ANY means unconstrained
enum class BooleanEnum { Any = 0, False = 1, True = 2 };

} // namespace midigpt

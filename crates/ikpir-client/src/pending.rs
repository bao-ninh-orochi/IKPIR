//! [`PendingDelta`] — the client's rolling public `ΔD`, accumulated while in
//! [`ClientUpdateMode::Rewind`](ikpir_common::ClientUpdateMode::Rewind).

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

use ikpir_common::SegmentRowDeltas;

/// Rolling public `ΔD = D_head − D_pin`, one `BTreeMap<(row, cell_offset),
/// signed delta>` per PIR segment, accumulated since the client's pinned hint.
///
/// # Purpose
///
/// A [`ClientUpdateMode::Rewind`](ikpir_common::ClientUpdateMode::Rewind) client
/// never patches its hint; it pins `H₀` at its bootstrap epoch and rolls this
/// sparse per-cell sum forward. Three consumers read it: the response-rewind
/// correction (a whole segment's map — [`Self::segment`]), the post-decode row
/// fix-up (one row — [`Self::row`]), and garbage collection, which materialises
/// each segment back into [`SegmentRowDeltas`] ([`Self::as_row_deltas`]) to fold
/// into the hint.
///
/// # A flat running sum, not a history
///
/// Only the *summed* effect since the pinned epoch is kept, so memory is
/// proportional to touched cells, not to the number of deltas folded. Because
/// the server publishes each per-cell delta as `new − old` (both in `[0, p)`),
/// the sum telescopes to `current − pin` per cell, hence stays in `(−p, p)` and
/// can never overflow `i64`; garbage collection can therefore only target
/// exactly the current head.
#[derive(Debug, Clone)]
pub(crate) struct PendingDelta {
    per_segment: Vec<BTreeMap<(u32, u16), i64>>,
}

impl PendingDelta {
    /// A fresh, empty accumulator for an `arity`-segment client.
    pub(crate) fn new(arity: usize) -> Self {
        Self {
            per_segment: vec![BTreeMap::new(); arity],
        }
    }

    /// Total nonzero `(segment, row, offset)` cells pending across every segment
    /// — the `Θ(τ·w)` staleness quantity the per-query correction scales with.
    pub(crate) fn cells(&self) -> usize {
        self.per_segment.iter().map(BTreeMap::len).sum()
    }

    /// Reset to empty (after a garbage-collection fold, or a resync).
    pub(crate) fn clear(&mut self) {
        for seg in &mut self.per_segment {
            seg.clear();
        }
    }

    /// Segment `seg`'s whole accumulated map — every touched cell anywhere in
    /// the segment. Consumed by the response-rewind correction `resp -= qᵀ·ΔD`,
    /// which must correct for changes in any row, not just the queried one.
    pub(crate) fn segment(&self, seg: usize) -> &BTreeMap<(u32, u16), i64> {
        &self.per_segment[seg]
    }

    /// Deltas for exactly `row` in segment `seg`, ascending by offset — the
    /// post-decode fix-up patches only the queried row.
    ///
    /// A `BTreeMap` keyed by `(row, offset)` orders lexicographically, so the
    /// range `(row, 0)..=(row, u16::MAX)` is exactly `row`'s entries in
    /// ascending offset order, with no secondary index.
    pub(crate) fn row(&self, seg: usize, row: u32) -> impl Iterator<Item = (u16, i64)> + '_ {
        self.per_segment[seg]
            .range((row, 0u16)..=(row, u16::MAX))
            .map(|(&(_, off), &d)| (off, d))
    }

    /// Materialise segment `seg` as [`SegmentRowDeltas`] (rows ascending,
    /// offsets ascending within each row) for
    /// [`client_patch_state`](ikpir_common::IncrementalPirBackend::client_patch_state)
    /// — used by garbage collection to fold the whole accumulator into the hint
    /// in one call.
    pub(crate) fn as_row_deltas(&self, seg: usize) -> SegmentRowDeltas {
        let mut out: SegmentRowDeltas = Vec::new();
        for (&(row, off), &delta) in &self.per_segment[seg] {
            match out.last_mut() {
                Some((last_row, cells)) if *last_row == row => cells.push((off, delta)),
                _ => out.push((row, vec![(off, delta)])),
            }
        }
        out
    }

    /// Fold one epoch's per-segment sparse deltas into the accumulator: per-cell
    /// sum, dropping any cell that nets to zero so the map stays sparse.
    ///
    /// `per_segment.len()` must equal this accumulator's arity — the caller
    /// (`IkpirClient::accumulate_delta`) validates it before calling.
    ///
    /// # Returns
    ///
    /// `false` on `i64` overflow while summing — unreachable with real PIR cell
    /// deltas (the running sum stays in `(−p, p)`), but checked rather than
    /// assumed.
    pub(crate) fn merge(&mut self, per_segment: &[SegmentRowDeltas]) -> bool {
        for (seg, rows) in per_segment.iter().enumerate() {
            for (row, cells) in rows {
                for &(off, delta) in cells {
                    match self.per_segment[seg].entry((*row, off)) {
                        Entry::Vacant(v) => {
                            if delta != 0 {
                                v.insert(delta);
                            }
                        }
                        Entry::Occupied(mut o) => match o.get().checked_add(delta) {
                            Some(0) => {
                                o.remove();
                            }
                            Some(sum) => *o.get_mut() = sum,
                            None => return false,
                        },
                    }
                }
            }
        }
        true
    }
}

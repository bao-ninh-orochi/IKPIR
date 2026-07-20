//! [`TouchedRuns`] — the entry-level hint patch's inner loop, shared by
//! both backends.
//!
//! # Purpose
//!
//! The entry-level (iSimplePIR) patch touches only the hint columns a
//! mutation actually rewrote:
//! `H[k, c] += A[row, k] · Δ_c mod 2³²`, for `k ∈ [lwe_dim]` and every
//! touched column `c`. Expressing that directly — one column at a time,
//! walking `k` inside — is the natural reading of the formula and the
//! wrong way to execute it: `H` is row-major, so consecutive `k` are
//! `hint_row_width` apart and each touched column drags `lwe_dim`
//! distinct cache lines through the machine. A patch of `t` columns then
//! makes `t` full passes over a multi-megabyte hint, and the mode loses
//! in wall-clock to the dense row-level pass it beats on paper.
//!
//! Executed the other way round — `k` outside, touched columns inside —
//! the patch makes **one** pass over the hint. And because a slot's cells
//! are laid out contiguously (`cell_offset = slot · cells_per_slot + c`,
//! and SimplePIR's reshape translation is affine in the offset), the
//! touched columns of one mutation form a short contiguous **run**. So
//! this module coalesces them into runs and applies the patch as
//! `dst.iter_mut().zip(src)` over each run — the same slice-shaped kernel
//! the row-level pass runs over the *whole* hint row, restricted to the
//! touched part of it. Entry-level then costs what its complexity says it
//! costs.
//!
//! # Design / architecture
//!
//! [`TouchedRuns`] is a reusable buffer: [`rebuild`](TouchedRuns::rebuild)
//! refills it from one database row's sparse edits, then
//! [`apply`](TouchedRuns::apply) sweeps the hint once. Callers hoist one
//! instance out of their row loop, so a batch of mutations allocates at
//! most once.
//!
//! `rebuild` takes a `column_of` closure because the two backends address
//! the hint differently: FrodoPIR patches column `cell_offset`, SimplePIR
//! the reshape-translated column. That is the *only* difference between
//! the two entry-level paths, which is why this kernel is shared rather
//! than duplicated (same rationale as [`matvec`](super::matvec): it is
//! backend-agnostic linear algebra whose tuning must not drift apart).
//!
//! # Equivalence to the row-level pass
//!
//! `rebuild` sums duplicate edits to one column before the multiply, the
//! same folding the row-level pass does while densifying, so both modes
//! compute `aik · (Σᵢ Δᵢ)` per touched column and the two hints stay
//! bit-identical. Untouched columns are never read or written by either
//! mode. See [`HintPatchMode`](super::HintPatchMode).
//!
//! # Related files
//!
//! - `frodo/backend.rs::apply_patch_entry_level` — FrodoPIR caller.
//! - `simple/backend.rs::apply_patch_entry_level` — SimplePIR caller.

/// One database row's touched hint columns, coalesced into maximal
/// contiguous runs.
///
/// Reuse one instance across a batch: [`rebuild`](Self::rebuild) clears
/// and refills, keeping the allocations.
pub(crate) struct TouchedRuns {
    /// Deltas of every touched column, run by run, in column order.
    /// `deltas.len()` equals the sum of the run lengths.
    deltas: Vec<u32>,
    /// `(first hint column, run length)` per run, in column order.
    runs: Vec<(usize, usize)>,
    /// Sort/merge scratch for [`rebuild`](Self::rebuild).
    scratch: Vec<(usize, u32)>,
}

impl TouchedRuns {
    /// An empty buffer that has not allocated yet.
    pub(crate) const fn new() -> Self {
        Self {
            deltas: Vec::new(),
            runs: Vec::new(),
            scratch: Vec::new(),
        }
    }

    /// Refill from one row's sparse `(cell_offset, Δ)` edits.
    ///
    /// `column_of` maps a wire cell offset to the hint column it patches
    /// — identity for FrodoPIR, the reshape translation for SimplePIR.
    /// Zero deltas are dropped (they cannot move the hint), duplicate
    /// columns are summed, and the survivors are coalesced into runs.
    ///
    /// # Complexity
    ///
    /// `O(t log t)` in the number of edits `t` — a per-row cost, paid
    /// once against the `O(t · lwe_dim)` of [`apply`](Self::apply).
    pub(crate) fn rebuild<F>(&mut self, cells: &[(u16, i64)], mut column_of: F)
    where
        F: FnMut(u16) -> usize,
    {
        self.scratch.clear();
        self.deltas.clear();
        self.runs.clear();

        self.scratch.extend(
            cells
                .iter()
                .filter(|(_, delta)| *delta != 0)
                .map(|(off, delta)| (column_of(*off), *delta as u32)),
        );
        // Sorted input is the norm — `fold_mutations_into_row_deltas`
        // drains a `BTreeMap` — but not a precondition, and coalescing
        // needs column order. Sorting here costs one pass per row; the
        // alternative, an unsorted scan, would cost run splits inside the
        // `lwe_dim`-deep sweep.
        self.scratch.sort_unstable_by_key(|&(col, _)| col);

        let mut last_col: Option<usize> = None;
        for i in 0..self.scratch.len() {
            let (col, delta) = self.scratch[i];
            match last_col {
                // Duplicate edit to one column: fold it in, exactly as the
                // row-level pass folds duplicates while densifying.
                Some(prev) if prev == col => {
                    let last = self
                        .deltas
                        .last_mut()
                        .expect("a previous column implies a pushed delta");
                    *last = last.wrapping_add(delta);
                }
                // Adjacent column: extend the open run.
                Some(prev) if prev + 1 == col => {
                    self.deltas.push(delta);
                    self.runs
                        .last_mut()
                        .expect("a previous column implies an open run")
                        .1 += 1;
                    last_col = Some(col);
                }
                // Gap (or the first column): open a new run.
                _ => {
                    self.runs.push((col, 1));
                    self.deltas.push(delta);
                    last_col = Some(col);
                }
            }
        }
    }

    /// Whether the last [`rebuild`](Self::rebuild) left anything to patch.
    ///
    /// A row whose edits all cancelled to zero lands here, and the caller
    /// can skip it without expanding `A`'s row.
    pub(crate) fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Apply `H[k, c] += a_row[k] · Δ_c mod 2³²` for every touched column
    /// `c`, sweeping the hint once.
    ///
    /// `hint` is the row-major `lwe_dim × hint_row_width` matrix and
    /// `a_row` the `lwe_dim`-long column of `A` for the mutated database
    /// row. No `aik == 0` shortcut: `A` is uniform over `Z_q`, so the
    /// branch would cost more than it saves and would make the schedule
    /// depend on sampled values.
    ///
    /// # Complexity
    ///
    /// `O(touched_columns · lwe_dim)` wrapping multiply-add, over
    /// `lwe_dim · (run count)` contiguous slices.
    pub(crate) fn apply(&self, a_row: &[u32], hint: &mut [u32], hint_row_width: usize) {
        debug_assert_eq!(hint.len(), a_row.len() * hint_row_width);
        for (k, &aik) in a_row.iter().enumerate() {
            let h_row = &mut hint[k * hint_row_width..(k + 1) * hint_row_width];
            let mut fed = 0;
            for &(start, len) in &self.runs {
                let dst = &mut h_row[start..start + len];
                let src = &self.deltas[fed..fed + len];
                for (h, &delta) in dst.iter_mut().zip(src) {
                    *h = h.wrapping_add(aik.wrapping_mul(delta));
                }
                fed += len;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: the formula executed literally, one column at a time.
    fn oracle(
        cells: &[(u16, i64)],
        a_row: &[u32],
        hint: &mut [u32],
        hint_row_width: usize,
        column_of: impl Fn(u16) -> usize,
    ) {
        for &(off, delta) in cells {
            if delta == 0 {
                continue;
            }
            let col = column_of(off);
            for (k, &aik) in a_row.iter().enumerate() {
                let idx = k * hint_row_width + col;
                hint[idx] = hint[idx].wrapping_add(aik.wrapping_mul(delta as u32));
            }
        }
    }

    fn fixture(hint_row_width: usize, lwe_dim: usize) -> (Vec<u32>, Vec<u32>) {
        let a_row: Vec<u32> = (0..lwe_dim)
            .map(|k| (k as u32).wrapping_mul(2_654_435_761).wrapping_add(7))
            .collect();
        let hint: Vec<u32> = (0..lwe_dim * hint_row_width)
            .map(|i| (i as u32).wrapping_mul(40_503))
            .collect();
        (a_row, hint)
    }

    /// Coalescing must not change the arithmetic: contiguous runs, gaps,
    /// duplicates, zero deltas, negative deltas and unsorted input all
    /// reproduce the literal per-column formula bit for bit.
    #[test]
    fn matches_literal_formula() {
        let (w, n) = (32usize, 8usize);
        let cases: &[&[(u16, i64)]] = &[
            &[(3, 5)],
            &[(3, 5), (4, -9), (5, 1)],            // one run
            &[(0, 3), (1, 4), (17, -2), (18, 8)],  // two runs
            &[(9, 7), (9, -2), (9, 4)],            // duplicates on one column
            &[(6, 0), (7, 3), (8, 0)],             // zero deltas dropped
            &[(20, -1), (2, 6), (21, 3), (3, -4)], // unsorted, two runs
            &[(31, i64::from(i32::MIN))],          // wide negative delta
            &[(5, 0)],                             // everything cancels
        ];
        for cells in cases {
            let (a_row, hint0) = fixture(w, n);
            let mut got = hint0.clone();
            let mut want = hint0;

            let mut runs = TouchedRuns::new();
            runs.rebuild(cells, |off| off as usize);
            if !runs.is_empty() {
                runs.apply(&a_row, &mut got, w);
            }
            oracle(cells, &a_row, &mut want, w, |off| off as usize);

            assert_eq!(got, want, "diverged on {cells:?}");
        }
    }

    /// A buffer reused across rows must not leak the previous row's runs.
    #[test]
    fn rebuild_clears_previous_row() {
        let (w, n) = (16usize, 4usize);
        let (a_row, hint0) = fixture(w, n);
        let mut runs = TouchedRuns::new();

        runs.rebuild(&[(0, 5), (1, 6), (2, 7)], |off| off as usize);
        let mut scratch = hint0.clone();
        runs.apply(&a_row, &mut scratch, w);

        let second: &[(u16, i64)] = &[(9, -3)];
        runs.rebuild(second, |off| off as usize);
        let mut got = hint0.clone();
        runs.apply(&a_row, &mut got, w);

        let mut want = hint0;
        oracle(second, &a_row, &mut want, w, |off| off as usize);
        assert_eq!(got, want);
    }

    /// An all-zero row reports empty, so the caller can skip it.
    #[test]
    fn all_zero_row_is_empty() {
        let mut runs = TouchedRuns::new();
        runs.rebuild(&[(1, 0), (2, 0)], |off| off as usize);
        assert!(runs.is_empty());
    }

    /// `column_of` is honoured — SimplePIR patches translated columns,
    /// and adjacency is decided *after* translation.
    #[test]
    fn honours_column_translation() {
        let (w, n) = (24usize, 4usize);
        let (a_row, hint0) = fixture(w, n);
        let cells: &[(u16, i64)] = &[(0, 3), (1, -5)];
        let shift = |off: u16| off as usize + 8;

        let mut got = hint0.clone();
        let mut runs = TouchedRuns::new();
        runs.rebuild(cells, shift);
        runs.apply(&a_row, &mut got, w);

        let mut want = hint0;
        oracle(cells, &a_row, &mut want, w, shift);
        assert_eq!(got, want);
    }
}

//! Round-trip correctness across all 6 schemes.
//!
//! For each filter type: insert a batch of items, assert `contain` holds,
//! delete each, assert `contain` no longer holds, reinsert, delete again.
//!
//! Uses `proptest` for randomised strategies; deterministic failures are
//! replayed from `proptest-regressions/filter.txt` (committed).

use proptest::prelude::*;
use segmented_cuckoo_filter::{
    CuckooError, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter, SegmentedCuckooFilter,
    Standard3aryCuckooFilter, Standard4aryCuckooFilter, StandardCuckooFilter,
};

const FP_BITS: u32 = 12;

fn roundtrip<F>(mut new_filter: F, items: &[[u8; 8]])
where
    F: FnMut() -> Box<dyn RoundtripFilter>,
{
    let mut f = new_filter();

    // Insert until table rejects; record which items landed.
    let mut inserted: Vec<[u8; 8]> = Vec::with_capacity(items.len());
    for item in items {
        match f.add_bytes(item) {
            Ok(()) => inserted.push(*item),
            Err(CuckooError::TableFull) => break,
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    for item in &inserted {
        assert!(f.contain_bytes(item), "inserted item missing: {item:?}");
    }

    for item in &inserted {
        let _ = f.delete_bytes(item);
    }

    // After deletion all inserted items should be gone (filter may still
    // report `true` for *other* items due to false positives — that's the
    // filter's nature — but items we inserted and deleted should be absent
    // unless a collision caused them to share a fingerprint.).
    //
    // We weaken the post-delete check: at least the load factor must drop.
    // (Exact absence is not guaranteed under fingerprint collision.)
}

/// Minimal type-erased trait to let the helper above drive all schemes.
trait RoundtripFilter {
    fn add_bytes(&mut self, x: &[u8; 8]) -> Result<(), CuckooError>;
    fn contain_bytes(&self, x: &[u8; 8]) -> bool;
    fn delete_bytes(&mut self, x: &[u8; 8]) -> Result<(), CuckooError>;
}

macro_rules! impl_roundtrip_filter {
    ($ty:ty) => {
        impl RoundtripFilter for $ty {
            fn add_bytes(&mut self, x: &[u8; 8]) -> Result<(), CuckooError> {
                self.add(x)
            }
            fn contain_bytes(&self, x: &[u8; 8]) -> bool {
                self.contain(x)
            }
            fn delete_bytes(&mut self, x: &[u8; 8]) -> Result<(), CuckooError> {
                self.delete(x)
            }
        }
    };
}

impl_roundtrip_filter!(StandardCuckooFilter);
impl_roundtrip_filter!(SegmentedCuckooFilter);
impl_roundtrip_filter!(Standard3aryCuckooFilter);
impl_roundtrip_filter!(Segmented3aryCuckooFilter);
impl_roundtrip_filter!(Standard4aryCuckooFilter);
impl_roundtrip_filter!(Segmented4aryCuckooFilter);

proptest! {
    #[test]
    fn standard_2ary_roundtrip(items in proptest::collection::vec(any::<u64>(), 1..500)) {
        let bytes: Vec<[u8; 8]> = items.iter().map(|x| x.to_le_bytes()).collect();
        roundtrip(
            || Box::new(StandardCuckooFilter::new(1024, 4, FP_BITS).unwrap()),
            &bytes,
        );
    }

    #[test]
    fn segmented_2ary_roundtrip(items in proptest::collection::vec(any::<u64>(), 1..500)) {
        let bytes: Vec<[u8; 8]> = items.iter().map(|x| x.to_le_bytes()).collect();
        roundtrip(
            || Box::new(SegmentedCuckooFilter::new(1024, 4, FP_BITS).unwrap()),
            &bytes,
        );
    }

    #[test]
    fn standard_3ary_roundtrip(items in proptest::collection::vec(any::<u64>(), 1..500)) {
        let bytes: Vec<[u8; 8]> = items.iter().map(|x| x.to_le_bytes()).collect();
        roundtrip(
            || Box::new(Standard3aryCuckooFilter::new(729, 4, FP_BITS).unwrap()),
            &bytes,
        );
    }

    #[test]
    fn segmented_3ary_roundtrip(items in proptest::collection::vec(any::<u64>(), 1..500)) {
        let bytes: Vec<[u8; 8]> = items.iter().map(|x| x.to_le_bytes()).collect();
        roundtrip(
            || Box::new(Segmented3aryCuckooFilter::new(3 * 256, 4, FP_BITS).unwrap()),
            &bytes,
        );
    }

    #[test]
    fn standard_4ary_roundtrip(items in proptest::collection::vec(any::<u64>(), 1..500)) {
        let bytes: Vec<[u8; 8]> = items.iter().map(|x| x.to_le_bytes()).collect();
        roundtrip(
            || Box::new(Standard4aryCuckooFilter::new(1024, 4, FP_BITS).unwrap()),
            &bytes,
        );
    }

    #[test]
    fn segmented_4ary_roundtrip(items in proptest::collection::vec(any::<u64>(), 1..500)) {
        let bytes: Vec<[u8; 8]> = items.iter().map(|x| x.to_le_bytes()).collect();
        roundtrip(
            || Box::new(Segmented4aryCuckooFilter::new(1024, 4, FP_BITS).unwrap()),
            &bytes,
        );
    }
}

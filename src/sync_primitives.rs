// ABOUTME: Crate-internal mutex alias selected by the `std` feature
// ABOUTME: parking_lot with std, spin without; both expose `.lock()` without poisoning

/// Mutex used by crate internals that must work with and without `std`.
///
/// Both backends return a guard directly from `lock()` (no poisoning), so
/// call sites are identical.
#[cfg(feature = "std")]
pub(crate) type Mutex<T> = parking_lot::Mutex<T>;

/// Mutex used by crate internals that must work with and without `std`.
#[cfg(not(feature = "std"))]
pub(crate) type Mutex<T> = spin::Mutex<T>;

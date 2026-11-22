//! The safety experiment in one place.
//!
//! Every array read in the hot loops of `eri_tab.rs` and `fock.rs` goes through
//! `ld::<CHECKED>(slice, i)`.  `CHECKED` is a compile-time constant, so the
//! compiler builds two separate copies of each kernel:
//!   CHECKED = true  -> `s[i]`: Rust inserts `if i >= s.len() { panic!() }`
//!   CHECKED = false -> `unsafe { *s.get_unchecked(i) }`: no check; an index
//!                      bug here would silently read garbage memory, exactly
//!                      like C/C++/Fortran.

#[inline(always)]
pub fn ld<const CHECKED: bool>(s: &[f64], i: usize) -> f64 {
    if CHECKED {
        s[i]
    } else {
        // SAFETY: callers guarantee i < s.len(); the CHECKED=true build of the
        // very same code is run by the test-suite and would panic otherwise.
        unsafe { *s.get_unchecked(i) }
    }
}

#[inline(always)]
pub fn st<const CHECKED: bool>(s: &mut [f64], i: usize, v: f64) {
    if CHECKED {
        s[i] = v
    } else {
        // SAFETY: as for `ld`.
        unsafe { *s.get_unchecked_mut(i) = v }
    }
}

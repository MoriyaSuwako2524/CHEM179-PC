//! Read-only view of `hfperf.basis.pack_basis(funcs)` (structure of arrays).
pub struct Basis<'a> {
    pub centers: &'a [f64], // nbf*3
    pub lmn: &'a [i32],     // nbf*3
    pub offsets: &'a [i32], // nbf+1
    pub exps: &'a [f64],    // nprim_total
    pub coefs: &'a [f64],   // nprim_total (primitive + contraction norms included)
}

impl<'a> Basis<'a> {
    pub fn new(centers: &'a [f64], lmn: &'a [i32], offsets: &'a [i32], exps: &'a [f64],
               coefs: &'a [f64]) -> Result<Self, String> {
        let n = offsets.len().checked_sub(1).ok_or("offsets must have nbf+1 entries")?;
        if centers.len() != 3 * n || lmn.len() != 3 * n {
            return Err(format!("centers/lmn must be ({n}, 3)"));
        }
        if exps.len() != coefs.len() || offsets[n] as usize != exps.len() {
            return Err("offsets[-1] must equal len(exps) == len(coefs)".into());
        }
        if offsets.windows(2).any(|w| w[0] > w[1] || w[0] < 0) || lmn.iter().any(|&x| x < 0) {
            return Err("offsets must be non-decreasing and lmn non-negative".into());
        }
        Ok(Basis { centers, lmn, offsets, exps, coefs })
    }
    #[inline]
    pub fn nbf(&self) -> usize { self.offsets.len() - 1 }
    #[inline]
    pub fn center(&self, i: usize) -> [f64; 3] {
        [self.centers[3 * i], self.centers[3 * i + 1], self.centers[3 * i + 2]]
    }
    #[inline]
    pub fn lmn(&self, i: usize) -> [i32; 3] {
        [self.lmn[3 * i], self.lmn[3 * i + 1], self.lmn[3 * i + 2]]
    }
    #[inline]
    pub fn prims(&self, i: usize) -> std::ops::Range<usize> {
        self.offsets[i] as usize..self.offsets[i + 1] as usize
    }
    pub fn max_l(&self) -> usize {
        (0..self.nbf()).map(|i| self.lmn(i).iter().sum::<i32>() as usize).max().unwrap_or(0)
    }
}

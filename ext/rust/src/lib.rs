//! PyO3 bindings: NumPy arrays in, NumPy arrays out.  Physics lives in the
//! other modules (no Python types there); this file only converts arrays and
//! releases the GIL (`py.allow_threads`) while the kernels run.
//! Stage 4 adds the Rayon entry points (par.rs): eri_sym8_par, jk_par, DirectJK.
mod basis;
mod boys;
mod eri;
mod eri_tab;
mod fock;
mod hermite;
mod idx;
mod one_e;
mod par;

use basis::Basis;
use numpy::ndarray::{Array2, Array4};
use numpy::{IntoPyArray, PyArray2, PyUntypedArrayMethods, PyArray4, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

type Arr2<'py> = Bound<'py, PyArray2<f64>>;
type Arr4<'py> = Bound<'py, PyArray4<f64>>;

/// Borrow the five pack_basis() arrays as one `Basis` (must be C-contiguous).
macro_rules! basis {
    ($c:expr, $l:expr, $o:expr, $e:expr, $k:expr) => {
        Basis::new($c.as_slice()?, $l.as_slice()?, $o.as_slice()?, $e.as_slice()?, $k.as_slice()?)
            .map_err(PyValueError::new_err)?
    };
}

fn to2(py: Python<'_>, n: usize, v: Vec<f64>) -> Arr2<'_> {
    Array2::from_shape_vec((n, n), v).expect("shape").into_pyarray_bound(py)
}

fn to4(py: Python<'_>, n: usize, v: Vec<f64>) -> Arr4<'_> {
    Array4::from_shape_vec((n, n, n, n), v).expect("shape").into_pyarray_bound(py)
}

fn check_l(bs: &Basis) -> PyResult<()> {
    if bs.max_l() > 2 {
        return Err(PyValueError::new_err("only s, p, d functions are supported"));
    }
    Ok(())
}

#[pyfunction]
fn overlap<'py>(py: Python<'py>, centers: PyReadonlyArray2<f64>, lmn: PyReadonlyArray2<i32>,
                offsets: PyReadonlyArray1<i32>, exps: PyReadonlyArray1<f64>,
                coefs: PyReadonlyArray1<f64>) -> PyResult<Arr2<'py>> {
    let bs = basis!(centers, lmn, offsets, exps, coefs);
    let v = py.allow_threads(|| one_e::overlap(&bs));
    Ok(to2(py, bs.nbf(), v))
}

#[pyfunction]
fn kinetic<'py>(py: Python<'py>, centers: PyReadonlyArray2<f64>, lmn: PyReadonlyArray2<i32>,
                offsets: PyReadonlyArray1<i32>, exps: PyReadonlyArray1<f64>,
                coefs: PyReadonlyArray1<f64>) -> PyResult<Arr2<'py>> {
    let bs = basis!(centers, lmn, offsets, exps, coefs);
    let v = py.allow_threads(|| one_e::kinetic(&bs));
    Ok(to2(py, bs.nbf(), v))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn nuclear<'py>(py: Python<'py>, centers: PyReadonlyArray2<f64>, lmn: PyReadonlyArray2<i32>,
                offsets: PyReadonlyArray1<i32>, exps: PyReadonlyArray1<f64>,
                coefs: PyReadonlyArray1<f64>, charges: PyReadonlyArray1<f64>,
                coords: PyReadonlyArray2<f64>) -> PyResult<Arr2<'py>> {
    let bs = basis!(centers, lmn, offsets, exps, coefs);
    let (z, r) = (charges.as_slice()?, coords.as_slice()?);
    if r.len() != 3 * z.len() {
        return Err(PyValueError::new_err("coords must be (natom, 3)"));
    }
    let v = py.allow_threads(|| one_e::nuclear(&bs, z, r));
    Ok(to2(py, bs.nbf(), v))
}

/// All N^4 ERIs, no symmetry: the naive algorithm, compiled.
#[pyfunction]
fn eri_naive<'py>(py: Python<'py>, centers: PyReadonlyArray2<f64>, lmn: PyReadonlyArray2<i32>,
                  offsets: PyReadonlyArray1<i32>, exps: PyReadonlyArray1<f64>,
                  coefs: PyReadonlyArray1<f64>) -> PyResult<Arr4<'py>> {
    let bs = basis!(centers, lmn, offsets, exps, coefs);
    check_l(&bs)?;
    let v = py.allow_threads(|| eri::eri_naive(&bs));
    Ok(to4(py, bs.nbf(), v))
}

/// 8-fold symmetry + Schwarz screening.  Returns (G, n_computed, n_screened).
///   reuse=False : call the naive per-quartet routine for every unique quartet
///   reuse=True  : tabulated Hermite E per primitive pair + iterative R (default)
///   checked     : (reuse=True only) bounds-checked or get_unchecked hot loop
#[pyfunction]
#[pyo3(signature = (centers, lmn, offsets, exps, coefs, screen=1e-12, reuse=true, checked=true))]
#[allow(clippy::too_many_arguments)]
fn eri_sym8<'py>(py: Python<'py>, centers: PyReadonlyArray2<f64>, lmn: PyReadonlyArray2<i32>,
                 offsets: PyReadonlyArray1<i32>, exps: PyReadonlyArray1<f64>,
                 coefs: PyReadonlyArray1<f64>, screen: f64, reuse: bool,
                 checked: bool) -> PyResult<(Arr4<'py>, u64, u64)> {
    let bs = basis!(centers, lmn, offsets, exps, coefs);
    check_l(&bs)?;
    let (v, nc, ns) = py.allow_threads(|| match (reuse, checked) {
        (false, _) => eri::eri_sym8(&bs, screen),
        (true, true) => eri_tab::eri_sym8_tab::<true>(&bs, screen),
        (true, false) => eri_tab::eri_sym8_tab::<false>(&bs, screen),
    });
    Ok((to4(py, bs.nbf(), v), nc, ns))
}

/// (J, K) from the full tensor.  mode: "iter" (safe, idiomatic; default),
/// "checked" (indexing with bounds checks) or "unchecked" (get_unchecked).
#[pyfunction]
#[pyo3(signature = (g, d, mode="iter"))]
fn jk<'py>(py: Python<'py>, g: PyReadonlyArray4<f64>, d: PyReadonlyArray2<f64>,
           mode: &str) -> PyResult<(Arr2<'py>, Arr2<'py>)> {
    let n = d.shape()[0];
    let gs = g.shape();
    if d.shape()[1] != n || gs != [n, n, n, n] {
        return Err(PyValueError::new_err("need G (n,n,n,n) and D (n,n)"));
    }
    let (gsl, dsl) = (g.as_slice()?, d.as_slice()?);
    let (jm, km) = match mode {
        "iter" => py.allow_threads(|| fock::jk_iter(gsl, dsl, n)),
        "checked" => py.allow_threads(|| fock::jk_index::<true>(gsl, dsl, n)),
        // the shape check above is what makes the unchecked build sound
        "unchecked" => py.allow_threads(|| fock::jk_index::<false>(gsl, dsl, n)),
        _ => return Err(PyValueError::new_err("mode must be iter | checked | unchecked")),
    };
    Ok((to2(py, n, jm), to2(py, n, km)))
}

// ------------------------------------------------------------------ stage 4: Rayon threads

fn info_dict<'py>(py: Python<'py>, p: &par::ParInfo) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("computed", p.computed)?;
    d.set_item("screened", p.screened)?;
    d.set_item("busy", p.busy.clone())?;
    d.set_item("setup_s", p.setup_s)?;
    d.set_item("loop_s", p.loop_s)?;
    d.set_item("gather_s", p.gather_s)?;
    d.set_item("reduce_s", p.reduce_s)?;
    Ok(d)
}

/// Number of worker threads Rayon would use by default (RAYON_NUM_THREADS or #CPUs).
#[pyfunction]
fn rayon_threads() -> usize { rayon::current_num_threads() }

/// eri_sym8 (tabulated kernel, bounds-checked) with Rayon over pair rows.
/// mode: "steal" (work stealing) | "block" (one contiguous block of rows per thread).
/// Returns (G, info) with per-thread busy seconds in info["busy"].
#[pyfunction]
#[pyo3(signature = (centers, lmn, offsets, exps, coefs, screen=1e-12, threads=0, mode="steal"))]
#[allow(clippy::too_many_arguments)]
fn eri_sym8_par<'py>(py: Python<'py>, centers: PyReadonlyArray2<f64>, lmn: PyReadonlyArray2<i32>,
                     offsets: PyReadonlyArray1<i32>, exps: PyReadonlyArray1<f64>,
                     coefs: PyReadonlyArray1<f64>, screen: f64, threads: usize,
                     mode: &str) -> PyResult<(Arr4<'py>, Bound<'py, PyDict>)> {
    let bs = basis!(centers, lmn, offsets, exps, coefs);
    check_l(&bs)?;
    let block = match mode {
        "steal" => false,
        "block" => true,
        _ => return Err(PyValueError::new_err("mode must be steal | block")),
    };
    let (v, info) = py.allow_threads(|| par::with_threads(threads, || par::eri_sym8_par(&bs, screen, block)));
    Ok((to4(py, bs.nbf(), v), info_dict(py, &info)?))
}

/// (J, K) from the full tensor, rows in parallel.
#[pyfunction]
#[pyo3(signature = (g, d, threads=0))]
fn jk_par<'py>(py: Python<'py>, g: PyReadonlyArray4<f64>, d: PyReadonlyArray2<f64>,
               threads: usize) -> PyResult<(Arr2<'py>, Arr2<'py>)> {
    let n = d.shape()[0];
    if d.shape()[1] != n || g.shape() != [n, n, n, n] {
        return Err(PyValueError::new_err("need G (n,n,n,n) and D (n,n)"));
    }
    let (gsl, dsl) = (g.as_slice()?, d.as_slice()?);
    let (jm, km) = py.allow_threads(|| par::with_threads(threads, || par::jk_par(gsl, dsl, n)));
    Ok((to2(py, n, jm), to2(py, n, km)))
}

/// Integral-direct J/K builder (no N^4 tensor).  rows: pair rows owned (default all;
/// used by the MPI driver); store=True keeps the values after the first build.
#[pyclass(name = "DirectJK")]
struct PyDirectJK {
    core: par::DirectCore,
}

#[pymethods]
impl PyDirectJK {
    #[new]
    #[pyo3(signature = (centers, lmn, offsets, exps, coefs, screen=1e-12, store=false, rows=None, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(py: Python<'_>, centers: PyReadonlyArray2<f64>, lmn: PyReadonlyArray2<i32>,
           offsets: PyReadonlyArray1<i32>, exps: PyReadonlyArray1<f64>, coefs: PyReadonlyArray1<f64>,
           screen: f64, store: bool, rows: Option<Vec<usize>>, threads: usize) -> PyResult<Self> {
        let bs = basis!(centers, lmn, offsets, exps, coefs);
        check_l(&bs)?;
        let core = py
            .allow_threads(|| par::with_threads(threads, || par::DirectCore::new(&bs, screen, store, rows)))
            .map_err(PyValueError::new_err)?;
        Ok(PyDirectJK { core })
    }

    /// (J, K) for density d.  mode: "fold" (alias "private") | "atomic".
    #[pyo3(signature = (d, threads=0, mode="fold"))]
    fn build<'py>(&mut self, py: Python<'py>, d: PyReadonlyArray2<f64>, threads: usize,
                  mode: &str) -> PyResult<(Arr2<'py>, Arr2<'py>)> {
        let n = self.core.n;
        if d.shape() != [n, n] {
            return Err(PyValueError::new_err("d must be (nbf, nbf)"));
        }
        let atomic = match mode {
            "fold" | "private" => false,
            "atomic" => true,
            _ => return Err(PyValueError::new_err("mode must be fold | atomic")),
        };
        let dsl = d.as_slice()?;
        let core = &mut self.core;
        let (jm, km) = py.allow_threads(|| par::with_threads(threads, || core.build(dsl, atomic)));
        Ok((to2(py, n, jm), to2(py, n, km)))
    }

    #[getter]
    fn nbf(&self) -> usize { self.core.n }
    #[getter]
    fn npair(&self) -> usize { self.core.npair() }
    #[getter]
    fn n_rows(&self) -> usize { self.core.rows.len() }
    #[getter]
    fn n_quartets(&self) -> usize { self.core.n_quartets() }
    #[getter]
    fn cached(&self) -> bool { self.core.cache.is_some() }
    #[getter]
    fn last_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> { info_dict(py, &self.core.last) }
}

#[pymodule]
fn hfperf_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(overlap, m)?)?;
    m.add_function(wrap_pyfunction!(kinetic, m)?)?;
    m.add_function(wrap_pyfunction!(nuclear, m)?)?;
    m.add_function(wrap_pyfunction!(eri_naive, m)?)?;
    m.add_function(wrap_pyfunction!(eri_sym8, m)?)?;
    m.add_function(wrap_pyfunction!(jk, m)?)?;
    m.add_function(wrap_pyfunction!(rayon_threads, m)?)?;
    m.add_function(wrap_pyfunction!(eri_sym8_par, m)?)?;
    m.add_function(wrap_pyfunction!(jk_par, m)?)?;
    m.add_class::<PyDirectJK>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

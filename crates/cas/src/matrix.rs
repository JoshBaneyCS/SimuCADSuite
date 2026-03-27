use std::fmt;

use simucad_core::error::CasError;

use crate::ast::Expr;
use crate::evaluator::{Environment, evaluate};
use crate::simplify::simplify;

// ---------------------------------------------------------------------------
// Matrix type
// ---------------------------------------------------------------------------

/// A symbolic matrix whose elements are CAS expressions.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<Expr>>,
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl Matrix {
    /// Create a matrix from row-major data, validating that all rows have the
    /// same length.
    pub fn new(data: Vec<Vec<Expr>>) -> Result<Self, CasError> {
        if data.is_empty() {
            return Err(CasError::DomainError(
                "Matrix must have at least one row".into(),
            ));
        }
        let cols = data[0].len();
        if cols == 0 {
            return Err(CasError::DomainError(
                "Matrix must have at least one column".into(),
            ));
        }
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(CasError::DomainError(format!(
                    "Row {} has {} columns, expected {}",
                    i,
                    row.len(),
                    cols
                )));
            }
        }
        Ok(Self {
            rows: data.len(),
            cols,
            data,
        })
    }

    /// Create an n×n identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                row.push(if i == j {
                    Expr::num(1.0)
                } else {
                    Expr::num(0.0)
                });
            }
            data.push(row);
        }
        Self {
            rows: n,
            cols: n,
            data,
        }
    }

    /// Create a matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let data = vec![vec![Expr::num(0.0); cols]; rows];
        Self { rows, cols, data }
    }

    /// Convenience constructor from numeric (`f64`) data.
    pub fn from_f64(data: Vec<Vec<f64>>) -> Self {
        let expr_data: Vec<Vec<Expr>> = data
            .into_iter()
            .map(|row| row.into_iter().map(Expr::num).collect())
            .collect();
        // from_f64 assumes well-formed input; panic on misuse.
        Self::new(expr_data).expect("from_f64: all rows must have the same length")
    }
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

impl Matrix {
    /// Element-wise addition. Dimensions must match.
    pub fn add(&self, other: &Matrix) -> Result<Matrix, CasError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(CasError::DomainError(format!(
                "Cannot add {}×{} matrix to {}×{} matrix",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(sr, or)| {
                sr.iter()
                    .zip(or.iter())
                    .map(|(a, b)| simplify(&Expr::add(a.clone(), b.clone())))
                    .collect()
            })
            .collect();
        Ok(Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    /// Element-wise subtraction. Dimensions must match.
    pub fn sub(&self, other: &Matrix) -> Result<Matrix, CasError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(CasError::DomainError(format!(
                "Cannot subtract {}×{} matrix from {}×{} matrix",
                other.rows, other.cols, self.rows, self.cols
            )));
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(sr, or)| {
                sr.iter()
                    .zip(or.iter())
                    .map(|(a, b)| simplify(&Expr::sub(a.clone(), b.clone())))
                    .collect()
            })
            .collect();
        Ok(Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    /// Matrix multiplication. `self.cols` must equal `other.rows`.
    pub fn mul(&self, other: &Matrix) -> Result<Matrix, CasError> {
        if self.cols != other.rows {
            return Err(CasError::DomainError(format!(
                "Cannot multiply {}×{} matrix by {}×{} matrix",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row = Vec::with_capacity(other.cols);
            for j in 0..other.cols {
                // sum over k: self[i][k] * other[k][j]
                let mut sum = Expr::num(0.0);
                for k in 0..self.cols {
                    let product = Expr::mul(self.data[i][k].clone(), other.data[k][j].clone());
                    sum = Expr::add(sum, product);
                }
                row.push(simplify(&sum));
            }
            data.push(row);
        }
        Ok(Matrix {
            rows: self.rows,
            cols: other.cols,
            data,
        })
    }

    /// Multiply every element by a scalar expression.
    pub fn scalar_mul(&self, scalar: &Expr) -> Matrix {
        let data = self
            .data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|e| simplify(&Expr::mul(scalar.clone(), e.clone())))
                    .collect()
            })
            .collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> Matrix {
        let mut data = Vec::with_capacity(self.cols);
        for j in 0..self.cols {
            let mut row = Vec::with_capacity(self.rows);
            for i in 0..self.rows {
                row.push(self.data[i][j].clone());
            }
            data.push(row);
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    /// Return the submatrix with the given row and column removed.
    pub fn minor(&self, row: usize, col: usize) -> Matrix {
        let data: Vec<Vec<Expr>> = self
            .data
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != row)
            .map(|(_, r)| {
                r.iter()
                    .enumerate()
                    .filter(|&(j, _)| j != col)
                    .map(|(_, e)| e.clone())
                    .collect()
            })
            .collect();
        Matrix {
            rows: self.rows - 1,
            cols: self.cols - 1,
            data,
        }
    }

    /// Compute the cofactor C_{row,col} = (-1)^{row+col} * det(minor).
    pub fn cofactor(&self, row: usize, col: usize) -> Result<Expr, CasError> {
        let minor_det = self.minor(row, col).determinant()?;
        let sign = if (row + col) % 2 == 0 { 1.0 } else { -1.0 };
        Ok(simplify(&Expr::mul(Expr::num(sign), minor_det)))
    }

    /// Compute the determinant of a square matrix using cofactor expansion
    /// along the first row.
    pub fn determinant(&self) -> Result<Expr, CasError> {
        if self.rows != self.cols {
            return Err(CasError::DomainError(format!(
                "Determinant requires a square matrix, got {}×{}",
                self.rows, self.cols
            )));
        }
        let n = self.rows;

        // 1×1
        if n == 1 {
            return Ok(self.data[0][0].clone());
        }

        // 2×2: ad - bc
        if n == 2 {
            let a = &self.data[0][0];
            let b = &self.data[0][1];
            let c = &self.data[1][0];
            let d = &self.data[1][1];
            let det = Expr::sub(
                Expr::mul(a.clone(), d.clone()),
                Expr::mul(b.clone(), c.clone()),
            );
            return Ok(simplify(&det));
        }

        // General: cofactor expansion along first row
        let mut det = Expr::num(0.0);
        for j in 0..n {
            let cof = self.cofactor(0, j)?;
            let term = Expr::mul(self.data[0][j].clone(), cof);
            det = Expr::add(det, term);
        }
        Ok(simplify(&det))
    }

    /// Compute the inverse of a square matrix.
    ///
    /// Uses the adjugate method: A^{-1} = adj(A) / det(A).
    pub fn inverse(&self) -> Result<Matrix, CasError> {
        if self.rows != self.cols {
            return Err(CasError::DomainError(format!(
                "Inverse requires a square matrix, got {}×{}",
                self.rows, self.cols
            )));
        }

        let det = self.determinant()?;

        // Check if determinant is zero (numerically).
        let env = Environment::new();
        let det_is_zero = match evaluate(&det, &env) {
            Ok(val) => val.abs() < 1e-12,
            Err(_) => {
                // Symbolic — check if it simplifies to Num(0).
                det.is_zero()
            }
        };
        if det_is_zero {
            return Err(CasError::DomainError(
                "Matrix is singular (determinant is zero)".into(),
            ));
        }

        let n = self.rows;
        let inv_det = Expr::div(Expr::num(1.0), det);

        // Build cofactor matrix, then transpose (= adjugate).
        let mut cofactor_data = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                row.push(self.cofactor(i, j)?);
            }
            cofactor_data.push(row);
        }
        let cofactor_matrix = Matrix {
            rows: n,
            cols: n,
            data: cofactor_data,
        };
        let adjugate = cofactor_matrix.transpose();

        // Multiply each element by 1/det.
        Ok(adjugate.scalar_mul(&simplify(&inv_det)))
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, row) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "[")?;
            for (j, elem) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", elem)?;
            }
            write!(f, "]")?;
        }
        write!(f, "]")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn mat2x2(a: f64, b: f64, c: f64, d: f64) -> Matrix {
        Matrix::from_f64(vec![vec![a, b], vec![c, d]])
    }

    fn eval_expr(e: &Expr) -> f64 {
        evaluate(e, &Environment::new()).unwrap()
    }

    fn eval_matrix(m: &Matrix) -> Vec<Vec<f64>> {
        m.data
            .iter()
            .map(|row| row.iter().map(|e| eval_expr(e)).collect())
            .collect()
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn matrices_approx_eq(a: &Matrix, b: &Matrix) -> bool {
        if a.rows != b.rows || a.cols != b.cols {
            return false;
        }
        let av = eval_matrix(a);
        let bv = eval_matrix(b);
        for (ar, br) in av.iter().zip(bv.iter()) {
            for (ae, be) in ar.iter().zip(br.iter()) {
                if !approx_eq(*ae, *be) {
                    return false;
                }
            }
        }
        true
    }

    // -- constructors --

    #[test]
    fn test_create_2x2() {
        let m = mat2x2(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
    }

    #[test]
    fn test_create_3x3() {
        let m = Matrix::from_f64(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
    }

    #[test]
    fn test_new_mismatched_rows() {
        let result = Matrix::new(vec![
            vec![Expr::num(1.0), Expr::num(2.0)],
            vec![Expr::num(3.0)],
        ]);
        assert!(result.is_err());
    }

    // -- identity --

    #[test]
    fn test_identity_mul() {
        let a = mat2x2(1.0, 2.0, 3.0, 4.0);
        let i = Matrix::identity(2);
        let result = a.mul(&i).unwrap();
        assert!(matrices_approx_eq(&result, &a));
    }

    // -- addition & subtraction --

    #[test]
    fn test_addition() {
        let a = mat2x2(1.0, 2.0, 3.0, 4.0);
        let b = mat2x2(5.0, 6.0, 7.0, 8.0);
        let c = a.add(&b).unwrap();
        let expected = mat2x2(6.0, 8.0, 10.0, 12.0);
        assert!(matrices_approx_eq(&c, &expected));
    }

    #[test]
    fn test_subtraction() {
        let a = mat2x2(5.0, 6.0, 7.0, 8.0);
        let b = mat2x2(1.0, 2.0, 3.0, 4.0);
        let c = a.sub(&b).unwrap();
        let expected = mat2x2(4.0, 4.0, 4.0, 4.0);
        assert!(matrices_approx_eq(&c, &expected));
    }

    // -- multiplication --

    #[test]
    fn test_mul_2x3_by_3x2() {
        let a = Matrix::from_f64(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let b = Matrix::from_f64(vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]);
        let c = a.mul(&b).unwrap();
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        let v = eval_matrix(&c);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!(approx_eq(v[0][0], 58.0));
        assert!(approx_eq(v[0][1], 64.0));
        assert!(approx_eq(v[1][0], 139.0));
        assert!(approx_eq(v[1][1], 154.0));
    }

    // -- transpose --

    #[test]
    fn test_transpose_of_transpose() {
        let a = Matrix::from_f64(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let att = a.transpose().transpose();
        assert!(matrices_approx_eq(&att, &a));
    }

    // -- determinant --

    #[test]
    fn test_det_2x2() {
        let m = mat2x2(1.0, 2.0, 3.0, 4.0);
        let det = eval_expr(&m.determinant().unwrap());
        assert!(approx_eq(det, -2.0));
    }

    #[test]
    fn test_det_3x3() {
        let m = Matrix::from_f64(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 0.0],
        ]);
        let det = eval_expr(&m.determinant().unwrap());
        assert!(approx_eq(det, 27.0));
    }

    #[test]
    fn test_det_identity() {
        let i = Matrix::identity(3);
        let det = eval_expr(&i.determinant().unwrap());
        assert!(approx_eq(det, 1.0));
    }

    // -- inverse --

    #[test]
    fn test_inverse_2x2() {
        let a = mat2x2(1.0, 2.0, 3.0, 4.0);
        let a_inv = a.inverse().unwrap();
        let product = a.mul(&a_inv).unwrap();
        let id = Matrix::identity(2);
        assert!(matrices_approx_eq(&product, &id));
    }

    #[test]
    fn test_singular_matrix_inverse_error() {
        let m = mat2x2(1.0, 2.0, 2.0, 4.0); // det = 0
        assert!(m.inverse().is_err());
    }

    // -- scalar multiplication --

    #[test]
    fn test_scalar_mul() {
        let a = mat2x2(1.0, 2.0, 3.0, 4.0);
        let scaled = a.scalar_mul(&Expr::num(3.0));
        let expected = mat2x2(3.0, 6.0, 9.0, 12.0);
        assert!(matrices_approx_eq(&scaled, &expected));
    }

    // -- dimension mismatch errors --

    #[test]
    fn test_add_dimension_mismatch() {
        let a = mat2x2(1.0, 2.0, 3.0, 4.0);
        let b = Matrix::from_f64(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_mul_dimension_mismatch() {
        let a = mat2x2(1.0, 2.0, 3.0, 4.0);
        let b = Matrix::from_f64(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);
        assert!(a.mul(&b).is_err());
    }

    // -- symbolic --

    #[test]
    fn test_symbolic_determinant() {
        // [[x, 1], [0, x]] → det = x*x - 1*0 = x^2
        let m = Matrix::new(vec![
            vec![Expr::var("x"), Expr::num(1.0)],
            vec![Expr::num(0.0), Expr::var("x")],
        ])
        .unwrap();
        let det = m.determinant().unwrap();
        // Evaluate with x = 3: should give 9
        let mut env = Environment::new();
        env.set("x", 3.0);
        let val = evaluate(&det, &env).unwrap();
        assert!(approx_eq(val, 9.0));
    }

    // -- display --

    #[test]
    fn test_display() {
        let m = mat2x2(1.0, 2.0, 3.0, 4.0);
        let s = format!("{}", m);
        assert_eq!(s, "[[1, 2], [3, 4]]");
    }
}

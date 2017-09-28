//! Math Operations.
use super::*;


///// Add /////

pub fn add<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Add::new(x.into(), y.into(), name)?)
}

/// Returns x + y element-wise.
///
/// Both x and y must be tensors of the same type. 
/// 
/// Name argument is optional, if an empty string is provided, 
/// the name will be generated automatically.
add_new_op!(Add,
    constructor: [add_new_op!(BIN CONSTRUCTOR: Add, Init: []);],
    digest: [DEFAULT_DIGEST: Add, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_add() {
    let mut context = Scope::new();
    let x = context.constant(&[2_i32], &[] as &[i32], "x").unwrap();
    let y = context.constant(&[2_i32], &[] as &[i32], "y").unwrap();
    let op = add(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [4_i32]});
}


///// AddN /////

pub fn add_n<S>(context: &mut Scope, values: Vec<Tensor>, name: S) -> Result<Tensor, ::Error>
where
    S: AsRef<Path>,
{
    context.install(AddN::new(values, name)?)
}

add_new_op!(AddN,
    constructor: [
        fn new<S: AsRef<Path>>(values: Vec<Tensor>, name: S) -> Result<AddN<'a>, ::Error> {
            let output_type = values[0].dtype;
            for x in &values {
                if &x.dtype != &output_type {
                    return Err(::Error::Stub);
                }
            }

            Ok(
                AddN {
                    ident: NodeIdent::new(),
                    name: generate_name!(is_none: name),
                    attributes: Vec::with_capacity(0),
                    elements: vec![],
                    input_lists: vec![(0, values)],
                    output_type: output_type,
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: AddN, DTYPE_ATTR],
    extra_funcs: [], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_add_n() {
    let mut context = Scope::new();
    let x = context.constant(&[2_i32], &[] as &[i32], "x").unwrap();
    let y = context.constant(&[2_i32], &[] as &[i32], "y").unwrap();
    let op = add_n(&mut context, vec![x.into(), y.into()], "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [4_i32]});
}


//// Cast /////

pub fn cast<Tx, S>(
    context: &mut Scope,
    tensor: Tx,
    ty: DataType,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Cast::new(tensor.into(), &[ty], name)?)
}

pub fn to_float<Tx, S>(context: &mut Scope, tensor: Tx, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Cast::new(tensor.into(), &[DataType::Float], name)?)
}

add_new_op!(Cast, 
    constructor: [
        fn new<S: AsRef<Path>>(x: Tensor, dst_type: &'a [DataType], name: S) -> Result<Cast<'a>, ::Error> {
            if &x.dtype == &dst_type[0] {
                // trivial cast
                return Err(::Error::Stub);
            }
            Ok(
                Cast {
                    ident: NodeIdent::new(),
                    elements: vec![x],
                    name: generate_name!(is_none: name),
                    attributes: vec![("DstT", false, dst_type.into())],
                    input_lists: vec![],
                    output_type: dst_type[0],
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: Cast, DTYPE_ATTR],
    extra_funcs: [], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_cast() {
    let mut context = Scope::new();
    let x = context.constant(&[0_i32, 1], &[2], "x").unwrap();
    let op = cast(&mut context, x, DataType::Double, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Double] == [0_f64, 1.]});
}


///// Conj /////

///  Returns the complex conjugate of a complex number.
///
///  Given a tensor `input` of complex numbers, this operation returns a tensor of
///  complex numbers that are the complex conjugate of each element in `input`. The
///  complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
///  real part and *b* is the imaginary part.
///
///  The complex conjugate returned by this operation is of the form \\(a - bj\\).
///
///  For example:
///
///      # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
///      tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
///
///  If `x` is real, it is returned unchanged.
///
///  Args:
///    x: `Tensor` to conjugate.  Must have numeric type.
///    name: A name for the operation (optional).
///
///  Returns:
///    A `Tensor` that is the conjugate of `x` (with the same type).
///
///    Error: If `x` is not a numeric tensor.
pub fn conj<Tx, S>(context: &mut Scope, x: Tx, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
{
    let x = x.into();
    if x.dtype.is_complex() {
        let scope = &mut context.name_scope(name.as_ref(), Some("Conj".as_ref()));
        unimplemented!()
    } else if x.dtype.is_floating() || x.dtype.is_integer() {
        Ok(x)
    } else {
        Err(::Error::Msg(format!("Expected numeric tensor, got dtype {:?}", x.dtype)),)
    }
}


///// Division /////

pub fn divide<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Div::new(x.into(), y.into(), name)?)
}

add_new_op!(Div,
    constructor: [add_new_op!(BIN CONSTRUCTOR: Div, Init: []);],
    digest: [DEFAULT_DIGEST: Div, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_divide() {
    let mut context = Scope::new();
    let x = context.constant(&[4_i32], &[] as &[i32], "x").unwrap();
    let y = context.constant(&[2_i32], &[] as &[i32], "y").unwrap();
    let op = divide(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [2_i32]});
}


///// Equal /////

pub fn equal<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Equal::new(x.into(), y.into(), name)?)
}

add_new_op!(Equal,
    constructor: [
        add_new_op!(BIN CONSTRUCTOR: Equal, Init: [output_type: DataType::Bool]);
    ],
    digest: [DEFAULT_DIGEST: Equal, DTYPE_ATTR],
    extra_funcs: [], 
    extra_attr: [
        output_type: DataType
    ],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_equal() {
    let mut context = Scope::new();
    let x = context.constant(&[1_i32, 2], &[2], "x").unwrap();
    let y = context.constant(&[1_i32, 2], &[2], "y").unwrap();
    let op = equal(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [true, true]});
}


///// Exp /////

pub fn exp<Tx, S>(context: &mut Scope, x: Tx, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Exp::new(x.into(), name)?)
}

add_new_op!(Exp, 
    constructor: [
        add_new_op!(UNARY CONSTRUCTOR: Exp, Init: []);
    ],
    digest: [DEFAULT_DIGEST: Exp, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_exp() {
    let mut context = Scope::new();
    let e = context.constant(&[1_f64], &[] as &[i32], "e").unwrap();
    let op = exp(&mut context, e, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Double] == [::std::f64::consts::E]});
}


///// Greater /////

pub fn greater<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Greater::new(x.into(), y.into(), name)?)
}

add_new_op!(Greater,
    constructor: [
        add_new_op!(BIN CONSTRUCTOR: Greater, Init: [output_type: DataType::Bool]);
    ],
    digest: [DEFAULT_DIGEST: Greater, DTYPE_ATTR],
    extra_funcs: [], 
    extra_attr: [
        output_type: DataType
    ],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_greater() {
    let mut context = Scope::new();
    let x = context.constant(&[1_i32, 2], &[2], "x").unwrap();
    let y = context.constant(&[0_i32, 3], &[2], "y").unwrap();
    let op = greater(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [true, false]});
}


///// MatMul /////

///   Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
///
///   The inputs must be matrices (or tensors of rank > 2, representing batches of
///   matrices), with matching inner dimensions, possibly after transposition.
///
///   Both matrices must be of the same type. The supported types are:
///   `float16`, `float32`, `float64`, `int32`, `complex64`, `complex128`.
///
///   Either matrix can be transposed or adjointed (conjugated and transposed) on
///   the fly by setting one of the corresponding flag to `True`. These are `False`
///   by default.
///
///   If one or both of the matrices contain a lot of zeros, a more efficient
///   multiplication algorithm can be used by setting the corresponding
///   `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
///   This optimization is only available for plain matrices (rank-2 tensors) with
///   datatypes `bfloat16` or `float32`.
///
///   For example:
///
///   ```python
///   # 2-D tensor `a`
///   a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.]
///                                                         [4. 5. 6.]]
///   # 2-D tensor `b`
///   b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) => [[7. 8.]
///                                                            [9. 10.]
///                                                            [11. 12.]]
///   c = tf.matmul(a, b) => [[58 64]
///                           [139 154]]
///
///
///   # 3-D tensor `a`
///   a = tf.constant(np.arange(1, 13, dtype=np.int32),
///                   shape=[2, 2, 3])                  => [[[ 1.  2.  3.]
///                                                          [ 4.  5.  6.]],
///                                                         [[ 7.  8.  9.]
///                                                          [10. 11. 12.]]]
///
///   # 3-D tensor `b`
///   b = tf.constant(np.arange(13, 25, dtype=np.int32),
///                   shape=[2, 3, 2])                   => [[[13. 14.]
///                                                           [15. 16.]
///                                                           [17. 18.]],
///                                                          [[19. 20.]
///                                                           [21. 22.]
///                                                           [23. 24.]]]
///   c = tf.matmul(a, b) => [[[ 94 100]
///                            [229 244]],
///                           [[508 532]
///                            [697 730]]]
///   ```
///
///   Args:
///     a: `Tensor` of type `float16`, `float32`, `float64`, `int32`, `complex64`,
///       `complex128` and rank > 1.
///     b: `Tensor` with same type and rank as `a`.
///     transpose_a: If `True`, `a` is transposed before multiplication.
///     transpose_b: If `True`, `b` is transposed before multiplication.
///     adjoint_a: If `True`, `a` is conjugated and transposed before
///       multiplication.
///     adjoint_b: If `True`, `b` is conjugated and transposed before
///       multiplication.
///     a_is_sparse: If `True`, `a` is treated as a sparse matrix.
///     b_is_sparse: If `True`, `b` is treated as a sparse matrix.
///     name: Name for the operation (optional).
///
///   Returns:
///     A `Tensor` of the same type as `a` and `b` where each inner-most matrix is
///     the product of the corresponding matrices in `a` and `b`, e.g. if all
///     transpose or adjoint attributes are `False`:
///
///     `output`[..., i, j] = sum_k (`a`[..., i, k] * `b`[..., k, j]),
///     for all indices i, j.
///
///     Note: This is matrix product, not element-wise product.
///
///     Error: If transpose_a and adjoint_a, or transpose_b and adjoint_b
///       are both set to True.
pub fn matmul<Tx, Ty, S>(
    context: &mut Scope,
    a: Tx,
    b: Ty,
    mut transpose_a: bool,
    mut transpose_b: bool,
    mut adjoint_a: bool,
    mut adjoint_b: bool,
    a_is_sparse: bool,
    b_is_sparse: bool,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    let scope = &mut context.name_scope(name.as_ref(), Some("MatMul".as_ref()));
    if transpose_a && adjoint_a {
        return Err(::Error::Msg("Only one of transpose_a and adjoint_a can be True.".to_owned(),),);
    }
    if transpose_b && adjoint_b {
        return Err(::Error::Msg("Only one of transpose_b and adjoint_b can be True.".to_owned(),),);
    }

    let mut a = a.into();
    let mut b = b.into();
    let a_shape = a.get_shape(scope);
    let b_shape = b.get_shape(scope);
    if (!a_is_sparse && !b_is_sparse) &&
       ((a_shape.dims().is_none() || a_shape.dims().unwrap() > 2) &&
        (b_shape.dims().is_none() || b_shape.dims().unwrap() > 2)) {
        // BatchMatmul does not support transpose, so we conjugate the matrix and
        // use adjoint instead. Conj() is a noop for real matrices.
        if transpose_a {
            a = conj(scope, a, "")?;
            adjoint_a = true;
        }
        if transpose_b {
            a = conj(scope, b, "")?;
            adjoint_b = true;
        }
    }

    // Neither matmul nor sparse_matmul support adjoint, so we conjugate
    // the matrix and use transpose instead. Conj() is a noop for real matrices.
    if adjoint_a {
        a = conj(scope, a, "")?;
        transpose_a = true;
    }
    if adjoint_b {
        b = conj(scope, b, "")?;
        transpose_b = true;
    }

    let sparse_matmul_a = match a.dtype {
        DataType::BFloat16 |
        DataType::Float => true,
        _ => false,
    };
    let sparse_matmul_b = match b.dtype {
        DataType::BFloat16 |
        DataType::Float => true,
        _ => false,
    };
    let mut use_sparse_matmul = sparse_matmul_a && sparse_matmul_b && (a_is_sparse || b_is_sparse);

    if [a.dtype, b.dtype].iter().find(|x| DataType::BFloat16 == **x).is_some() {
        // matmul currently doesn't handle bfloat16 inputs.
        use_sparse_matmul = true;
    }
    if use_sparse_matmul {
        //sparse_matmul(scope, a, b, transpose_a, transpose_b, a_is_sparse, b_is_sparse, "");
        unimplemented!()
    } else {
        if a.dtype != b.dtype {
            return Err(::Error::Msg("Matrix a and matrix b must be of the same type.".to_owned()),);
        }
        scope.install(MatMul::new(a, b, "")?.transpose_a(&[transpose_a]).transpose_b(&[transpose_b]))
    }
}

/// Multiply the matrix "a" by the matrix "b".
///
/// The inputs must be two-dimensional matrices and the inner dimension of
/// "a" (after being transposed if transpose_a is true) must match the
/// outer dimension of "b" (after being transposed if transposed_b is
/// true).
///
/// *Note*: The default kernel implementation for MatMul on GPUs uses
/// cublas.
///
/// transpose_a: If true, "a" is transposed before multiplication.
/// transpose_b: If true, "b" is transposed before multiplication.
add_new_op!(MatMul,
    constructor: [add_new_op!(BIN CONSTRUCTOR: MatMul, Init: []);],
    digest: [DEFAULT_DIGEST: MatMul, INPUT0],
    extra_funcs: [
        fn transpose_a(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("transpose_a", false, Attribute::Bool(val)));
            self
        }

        fn transpose_b(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("transpose_b", false, Attribute::Bool(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);


///// Multiply /////

pub fn multiply<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Mul::new(x.into(), y.into(), name)?)
}

add_new_op!(Mul,
    constructor: [add_new_op!(BIN CONSTRUCTOR: Mul, Init: []);],
    digest: [DEFAULT_DIGEST: Mul, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_multiply() {
    let mut context = Scope::new();
    let x = context.constant(&[4_i32], &[] as &[i32], "x").unwrap();
    let y = context.constant(&[2_i32], &[] as &[i32], "y").unwrap();
    let op = multiply(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [8_i32]});
}


///// Less /////

pub fn less<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Less::new(x.into(), y.into(), name)?)
}

add_new_op!(Less,
    constructor: [
        add_new_op!(BIN CONSTRUCTOR: Less, Init: [output_type: DataType::Bool]);
    ],
    digest: [DEFAULT_DIGEST: Less, DTYPE_ATTR],
    extra_funcs: [], 
    extra_attr: [
        output_type: DataType
    ],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_less() {
    let mut context = Scope::new();
    let x = context.constant(&[2_i32, 2], &[2], "x").unwrap();
    let y = context.constant(&[1_i32, 3], &[2], "y").unwrap();
    let op = less(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [false, true]});
}


///// Log /////

pub fn log<Tx, S>(context: &mut Scope, x: Tx, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Log::new(x.into(), name)?)
}

add_new_op!(Log, 
    constructor: [
        add_new_op!(UNARY CONSTRUCTOR: Log, Init: []);
    ],
    digest: [DEFAULT_DIGEST: Log, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_log() {
    let mut context = Scope::new();
    let e = context.constant(&[::std::f64::consts::E], &[] as &[i32], "e").unwrap();
    let op = log(&mut context, e, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Double] == [1.]});
}


///// Logical Not /////

pub fn logical_not<Tx, S>(context: &mut Scope, tensor: Tx, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
{
    let tensor = tensor.into();
    if tensor.dtype != DataType::Bool {
        return Err(::Error::Stub);
    }
    context.install(LogicalNot::new(tensor, name)?)
}

add_new_op!(LogicalNot, 
    constructor: [
        add_new_op!(UNARY CONSTRUCTOR: LogicalNot, Init: [output_type: DataType::Bool]);
    ],
    digest: [DEFAULT_DIGEST: LogicalNot, DTYPE_ATTR],
    extra_funcs: [], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_logical_not() {
    let mut context = Scope::new();
    let x = context.constant(&[true, false], &[2], "x").unwrap();
    let op = logical_not(&mut context, x, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [false, true]});
}


///// Pow /////

/// Computes the power of one value to another.
///
/// Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
/// corresponding elements in `x` and `y`. For example:
///
/// ```python
/// # tensor 'x' is [[2, 2]], [3, 3]]
/// # tensor 'y' is [[8, 16], [2, 3]]
/// tf.pow(x, y) ==> [[256, 65536], [9, 27]]
/// ```
///
/// Args:
///     x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
///     y: A `Tensor`. Must have the same type as `x`.
///     name: A name for the operation (optional).
///
/// Returns:
///     A `Tensor`. Has the same type as `x`.
pub fn pow<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Pow::new(x.into(), y.into(), name)?)
}

add_new_op!(Pow,
    constructor: [add_new_op!(BIN CONSTRUCTOR: Pow, Init: []);],
    digest: [DEFAULT_DIGEST: Pow, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_pow() {
    let mut context = Scope::new();
    let x = context.constant(&[2_i32], &[] as &[i32], "x").unwrap();
    let y = context.constant(&[2_i32], &[] as &[i32], "y").unwrap();
    let op = pow(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [4_i32]});
}


///// Reduce All /////

pub fn reduce_all<Tx, S>(
    context: &mut Scope,
    tensor: Tx,
    axis: &[i32],
    keep_dims: bool,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
{
    if axis.len() == 0 {
        // TODO: infer reduction to scalar
    }
    let dims = &[axis.len() as i64];
    let reduce = context.constant(axis, dims, name.as_ref())?;
    context.install(All::new(tensor.into(), reduce.into(), name)?.keep_dims(&[keep_dims]),)
}

add_new_op!(All, 
    constructor: [
        fn new<S: AsRef<Path>>(input: Tensor, axis: Tensor, name: S) -> Result<All<'a>, ::Error> {
            if input.dtype != DataType::Bool ||
               (axis.dtype != DataType::Int32 &&
                axis.dtype != DataType::Int64) {
                return Err(::Error::Stub);
            }
            Ok(
                All {
                    ident: NodeIdent::new(),
                    elements: vec![input, axis],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: vec![],
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: All, INPUT0],
    extra_funcs: [
        /// Default is false, must be an slice of len == 1.
        fn keep_dims(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("keep_dims", false, Attribute::Bool(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_reduce_all() {
    let mut context = Scope::new();
    let x = context.constant(&[true, true, true, false, true, true], &[2, 3], "x").unwrap();

    let op = reduce_all(&mut context, x, &[1], false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [true, false]});
    test_suite!(results; assert_len: {[0;Bool] == 2});

    let op = reduce_all(&mut context, x, &[0], false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [false, true, true]});
    test_suite!(results; assert_len: {[0;Bool] == 3});

    let op = reduce_all(&mut context, x, &[0, 1], false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [false]});
    test_suite!(results; assert_len: {[0;Bool] == 1});
}


///// Reduce LogSumExp /////

pub fn reduce_logsumexp<TeS, Tx, S>(
    context: &mut Scope,
    input: Tx,
    axis: &[TeS],
    keep_dims: bool,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
    TeS: ShapeSize,
{
    let scope = &mut context.name_scope(name.as_ref(), Some("ReduceLogSumExp".as_ref()));
    let input = input.into();
    if axis.len() == 0 {
        // TODO: infer reduction to scalar
    }

    let reduce_max = reduce_max(scope, input, axis, true, "")?;
    let my_max = stop_gradient(scope, reduce_max, "")?;

    let sub = sub(scope, input, my_max, "")?;
    let exp = exp(scope, sub, "")?;
    let reduce_sum = reduce_sum(scope, exp, axis, true, "")?;
    let log = log(scope, reduce_sum, "")?;

    let dims: Vec<i64>;
    let mut result = add(scope, log, my_max, "")?;
    if !keep_dims {
        let axis = if axis.len() == 0 {
            None
        } else {
            dims = shape_as_i64(axis);
            Some(dims.as_slice())
        };
        result = squeeze(scope, result, axis, "")?;
    }
    Ok(result)
}

#[test]
#[cfg(test)]
fn test_reduce_logsumexp() {
    let mut context = Scope::new();
    let x = context.constant(&[0_f64, 0., 0., 0., 0., 0.], &[2, 3], "x").unwrap();

    let op1 = reduce_logsumexp(&mut context, x, &[0, 1], false, "").unwrap();
    let results = test_suite!(run_op: [op1]; context, input: {});
    test_suite!(results; assert_len: {[0;Double] == 1});
    test_suite!(results; assert: {[0;Double] == [6_f64.ln()]});

    let op2 = reduce_logsumexp(&mut context, x, &[1], true, "").unwrap();
    let results = test_suite!(run_op: [op2]; context, input: {});
    test_suite!(results; assert_len: {[0;Double] == 2});
    test_suite!(results; assert: {[0;Double] == [3_f64.ln(), 3_f64.ln()]});
}


///// Reduce Sum /////

pub fn reduce_sum<TeS, Tx, S>(
    context: &mut Scope,
    tensor: Tx,
    axis: &[TeS],
    keep_dims: bool,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
    TeS: ShapeSize,
{
    if axis.len() == 0 {
        // TODO: infer reduction to scalar
    }
    let dims = &[axis.len() as i64];
    let reduce = context.constant(axis, dims, name.as_ref())?;
    context.install(Sum::new(tensor.into(), reduce.into(), name)?.keep_dims(&[keep_dims]),)
}

add_new_op!(Sum, 
    constructor: [
        fn new<S: AsRef<Path>>(input: Tensor, axis: Tensor, name: S) -> Result<Sum<'a>, ::Error> {
            if axis.dtype != DataType::Int32 &&
               axis.dtype != DataType::Int64 {
                return Err(::Error::Stub);
            }
            Ok(
                Sum {
                    ident: NodeIdent::new(),
                    elements: vec![input, axis],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: vec![],
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: Sum, INPUT0],
    extra_funcs: [
        /// Default is false, must be an slice of len == 1.
        fn keep_dims(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("keep_dims", false, Attribute::Bool(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_reduce_sum() {
    let mut context = Scope::new();
    let x = context.constant(&[1_i32, 2, 3, 4], &[2, 2], "x").unwrap();

    let op = reduce_sum(&mut context, x, &[0, 1], false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert_len: {[0;Int32] == 1});
    test_suite!(results; assert: {[0;Int32] == [10]});
}


///// Reduce Max /////

pub fn reduce_max<TeS, Tx, S>(
    context: &mut Scope,
    tensor: Tx,
    axis: &[TeS],
    keep_dims: bool,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
    TeS: ShapeSize,
{
    if axis.len() == 0 {
        // TODO: infer reduction to scalar
    }
    let dims = &[axis.len() as i64];
    let reduce = context.constant(axis, dims, name.as_ref())?;
    context.install(Max::new(tensor.into(), reduce.into(), name)?.keep_dims(&[keep_dims]),)
}

add_new_op!(Max, 
    constructor: [
        fn new<S: AsRef<Path>>(input: Tensor, axis: Tensor, name: S) -> Result<Max<'a>, ::Error> {
            if axis.dtype != DataType::Int32 &&
               axis.dtype != DataType::Int64 {
                return Err(::Error::Stub);
            }
            Ok(
                Max {
                    ident: NodeIdent::new(),
                    elements: vec![input, axis],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: vec![],
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: Max, INPUT0],
    extra_funcs: [
        /// Default is false, must be an slice of len == 1.
        fn keep_dims(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("keep_dims", false, Attribute::Bool(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_reduce_max() {
    let mut context = Scope::new();
    let x = context.constant(&[1_i32, 2, 3, 4], &[2, 2], "x").unwrap();

    let op = reduce_max(&mut context, x, &[0, 1], false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert_len: {[0;Int32] == 1});
    test_suite!(results; assert: {[0;Int32] == [4]});
}


///// Tanh /////

///  Computes hyperbolic tangent of `x` element-wise.
///
///  Args:
///    x: A Tensor with type `float`, `double`, `int32`,
///      `complex64`, `int64`, or `qint32`.
///    name: A name for the operation (optional).
///
///  Returns:
///    A Tensor respectively with the same type as `x` if
///    `x.dtype != qint32` otherwise the return type is `quint8`.
pub fn tanh<TeS, Tx, S>(
    context: &mut Scope,
    tensor: Tx,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path> {
    let scope = &mut context.name_scope(name.as_ref(), Some("Tanh".as_ref()));
    scope.install(Tanh::new(tensor.into(), name)?)
}

add_new_op!(Tanh,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: Tanh, Init: []);],
    digest: [DEFAULT_DIGEST: Tanh, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);


///// Minimum /////

/// Returns the min of x and y (i.e. x < y ? x : y) element-wise.
///
/// *NOTE*: `Minimum` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
///     x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
///     y: A `Tensor`. Must have the same type as `x`.
///     name: A name for the operation (optional).
///
/// Returns:
///     A `Tensor`. Has the same type as `x`.
pub fn minimum<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Minimum::new(x.into(), y.into(), name)?)
}

add_new_op!(Minimum,
    constructor: [add_new_op!(BIN CONSTRUCTOR: Minimum, Init: []);],
    digest: [DEFAULT_DIGEST: Minimum, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_minimum() {
    let mut context = Scope::new();
    let x = context.constant(&[4_i32, 3], &[2] as &[i32], "x").unwrap();
    let y = context.constant(&[2_i32, 6], &[2] as &[i32], "y").unwrap();

    let op = minimum(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [2_i32, 3]});
}


///// Stop Gradient /////

pub fn stop_gradient<Tx, S>(context: &mut Scope, tensor: Tx, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(StopGradient::new(tensor.into(), name)?)
}

add_new_op!(StopGradient, 
    constructor: [
        add_new_op!(UNARY CONSTRUCTOR: StopGradient, Init: []);
    ],
    digest: [DEFAULT_DIGEST: StopGradient, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);


///// Sub /////

pub fn sub<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Sub::new(x.into(), y.into(), name)?)
}

add_new_op!(Sub,
    constructor: [add_new_op!(BIN CONSTRUCTOR: Sub, Init: []);],
    digest: [DEFAULT_DIGEST: Sub, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_sub() {
    let mut context = Scope::new();
    let x = context.constant(&[4_i32], &[] as &[i32], "x").unwrap();
    let y = context.constant(&[2_i32], &[] as &[i32], "y").unwrap();
    let op = sub(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [2_i32]});
}


/////

pub fn unsorted_segment_sum<Tx, Ty, Tz, S>(
    context: &mut Scope,
    data: Tx,
    segment_ids: Ty,
    num_segments: Tz,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    Tz: Into<Tensor>,
    S: AsRef<Path>,
{
    unimplemented!()
}

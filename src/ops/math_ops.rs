//! Math Operations.
use super::*;
use TensorShape;

///// AccumulateN /////

///  Returns the element-wise sum of a list of tensors.
///
///  Optionally, pass `shape` and `tensor_dtype` for shape and type checking,
///  otherwise, these are inferred.
///
///  NOTE: This operation is not differentiable and cannot be used if inputs depend
///  on trainable variables. Please use `tf.add_n` for such cases.
///
///  Aside from differentiability, `tf.accumulate_n` performs the same operation as
///  `tf.add_n`, but does not wait for all of its inputs to be ready before
///  beginning to sum. This can save memory if inputs are ready at different times,
///  since minimum temporary storage is proportional to the output size rather than
///  the inputs size.
///
///  For example:
///
///  ```python
///  a = tf.constant([[1, 2], [3, 4]])
///  b = tf.constant([[5, 0], [0, 6]])
///  tf.accumulate_n([a, b, a])  # [[7, 4], [6, 14]]
///
///  # Explicitly pass shape and type
///  tf.accumulate_n([a, b, a], shape=[2, 2], tensor_dtype=tf.int32)  # [[7,  4],
///                                                                   #  [6, 14]]
///  ```
///
///  ### Args:
///    * inputs: A list of `Tensor` objects, each with same shape and type.
///    * shape: Shape of elements of `inputs`.
///    * tensor_dtype: The type of `inputs`.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    A `Tensor` of same shape and type as the elements of `inputs`.
pub fn accumulate_n<S: AsRef<Path>>(
    context: &mut Scope,
    list: Vec<Tensor>,
    shape: Option<TensorShape>,
    tensor_dtype: Option<DataType>,
    name: S,
) -> Result<Tensor> {
    unimplemented!()
}

///// Add /////

/// Returns x + y element-wise.
///
/// Both x and y must be tensors of the same type.
///
/// Name argument is optional, if an empty string is provided,
/// the name will be generated automatically.
pub fn add<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);
    context.install(Add::new(x, y, name)?)
}

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

/// Adds all input tensors element-wise.
pub fn add_n<S>(context: &mut Scope, values: Vec<Tensor>, name: S) -> Result<Tensor>
where
    S: AsRef<Path>,
{
    context.install(AddN::new(values, name)?)
}

add_new_op!(AddN,
    constructor: [
        fn new<S: AsRef<Path>>(values: Vec<Tensor>, name: S) -> Result<AddN<'a>> {
            let output_type = values[0].dtype;
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

/// Casts a tensor to a new type.
///
/// The operation casts x to dtype.
pub fn cast<Tx, S>(context: &mut Scope, x: Tx, ty: DataType, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    context.install(Cast::new(x, &[ty], name)?)
}

/// Casts a tensor to type float.
pub fn to_float<Tx, S>(context: &mut Scope, tensor: Tx, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let tensor = tensor.into_tensor(context);
    context.install(Cast::new(tensor, &[DataType::Float], name)?)
}

add_new_op!(Cast, 
    constructor: [
        fn new<S: AsRef<Path>>(x: Tensor, dst_type: &'a [DataType], name: S) -> Result<Cast<'a>> {
            Ok(
                Cast {
                    ident: NodeIdent::new(),
                    elements: vec![x],
                    name: generate_name!(is_none: name),
                    attributes: vec![("DstT", false, dst_type.into())],
                    input_lists: Vec::with_capacity(0),
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
    let x = context.constant(&[0_i32, 1], [2].as_ref(), "x").unwrap();
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
///  ```python
///      # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
///      tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
///  ```
///
///  If `x` is real, it is returned unchanged.
///
///  ### Args:
///    * x: `Tensor` to conjugate.  Must have numeric type.
///    * name: A name for the operation (empty string slice for autogenerated name).
///
///  ### Returns
///    A `Tensor` that is the conjugate of `x` (with the same type).
///
///  ### Error
///    If `x` is not a numeric tensor.
pub fn conj<Tx, S>(context: &mut Scope, x: Tx, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    if x.dtype.is_complex() {
        let scope = &mut context.name_scope(name.as_ref(), Some("Conj".as_ref()));
        scope.install(Conj::new(x, name)?)
    } else if x.dtype.is_floating() || x.dtype.is_integer() {
        Ok(x)
    } else {
        Err(Error::from(format!(
            "Expected numeric tensor, got dtype {:?}",
            x.dtype
        )))
    }
}

add_new_op!(Conj, 
    constructor: [ add_new_op!(UNARY CONSTRUCTOR: Conj, Init: []); ],
    digest: [DEFAULT_DIGEST: Conj, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

///// Division /////

/// Divide `x` by `y` element-wise.
pub fn divide<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);
    context.install(Div::new(x, y, name)?)
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

/// Returns the truth value of (x == y) element-wise.
///
/// ### Args:
///    * `x`: A Tensor. Must be one of the following types: half, float32, float64,
///     uint8, int8, int16, int32, int64, complex64, quint8, qint8, qint32, string,
///     bool, complex128.
///    * `y`: A Tensor. Must have the same type as x.
///    * `name`: A name for the operation (empty string slice for autogenerated name).
///
/// ### Returns
/// A Tensor of type bool.
pub fn equal<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);
    context.install(Equal::new(x, y, name)?)
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
    let x = context.constant(&[1_i32, 2], [2].as_ref(), "x").unwrap();
    let y = context.constant(&[1_i32, 2], [2].as_ref(), "y").unwrap();
    let op = equal(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [true, true]});
}

///// Exp /////

/// Computes exponential of x element-wise. y=ex.
///
/// ### Args:
/// * x: A Tensor. Must be one of the following types: half, float32, float64, complex64, complex128.
/// * name: A name for the operation (empty string slice for autogenerated name).
pub fn exp<Tx, S>(context: &mut Scope, x: Tx, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    context.install(Exp::new(x, name)?)
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

///// Floor /////

/// Returns element-wise largest integer not greater than x.
pub fn floor<Tx, S>(context: &mut Scope, x: Tx, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    context.install(Floor::new(x, name)?)
}

add_new_op!(Floor, 
    constructor: [
        add_new_op!(UNARY CONSTRUCTOR: Floor, Init: []);
    ],
    digest: [DEFAULT_DIGEST: Floor, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

///// FloorMod /////

/// Returns element-wise remainder of division.
///
/// When x < 0 xor y < 0 is true, this follows Python semantics in that the result here is
/// consistent with a flooring divide. E.g. floor(x / y) * y + mod(x, y) = x.
///
/// NOTE: FloorMod supports broadcasting.
pub fn floor_mod<Tx, Ty, S>(scope: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(scope);
    let y = y.into_tensor(scope);
    scope.install(FloorMod::new(x, y, name)?)
}

add_new_op!(FloorMod, 
    constructor: [
        add_new_op!(BIN CONSTRUCTOR: FloorMod, Init: []);
    ],
    digest: [DEFAULT_DIGEST: FloorMod, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

///// FloorDiv /////

/// Returns x // y element-wise.
///
/// NOTE: FloorDiv supports broadcasting.
pub fn floor_div<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);
    context.install(FloorDiv::new(x, y, name)?)
}

add_new_op!(FloorDiv, 
    constructor: [
        add_new_op!(BIN CONSTRUCTOR: FloorDiv, Init: []);
    ],
    digest: [DEFAULT_DIGEST: FloorDiv, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

///// Greater /////

/// Returns the truth value of (x > y) element-wise.
///
/// __NOTE:__ Greater supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
///
/// ### Args:
/// * x: A Tensor. Must be one of the following types: float32, float64, int32, int64, uint8, int16, int8, uint16, half.
/// * y: A Tensor. Must have the same type as x.
/// * name: A name for the operation (empty string slice for autogenerated name).
///
/// ### Returns:
/// * A Tensor of type bool.
pub fn greater<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);
    context.install(Greater::new(x, y, name)?)
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
    let x = context.constant(&[1_i32, 2], [2].as_ref(), "x").unwrap();
    let y = context.constant(&[0_i32, 3], [2].as_ref(), "y").unwrap();
    let op = greater(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [true, false]});
}

///// Maximum /////

///  Returns the max of x and y (i.e. x > y ? x : y) element-wise.
///
///  *NOTE*: `Maximum` supports broadcasting.
///
///  ### Args:
///    * x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
///    * y: A `Tensor`. Must have the same type as `x`.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    * A `Tensor`. Has the same type as `x`.
pub fn maximum<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);
    context.install(Maximum::new(x, y, name)?)
}

add_new_op!(Maximum,
    constructor: [add_new_op!(BIN CONSTRUCTOR: Maximum, Init: []);],
    digest: [DEFAULT_DIGEST: Maximum, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

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
///   ### Args:
///     * a: `Tensor` of type `float16`, `float32`, `float64`, `int32`, `complex64`,
///       `complex128` and rank > 1.
///     * b: `Tensor` with same type and rank as `a`.
///     * transpose_a: If `true`, `a` is transposed before multiplication.
///     * transpose_b: If `true`, `b` is transposed before multiplication.
///     * adjoint_a: If `true`, `a` is conjugated and transposed before
///       multiplication.
///     * adjoint_b: If `true`, `b` is conjugated and transposed before
///       multiplication.
///     * a_is_sparse: If `true`, `a` is treated as a sparse matrix.
///     * b_is_sparse: If `true`, `b` is treated as a sparse matrix.
///     * name: A name for the operation (empty string slice for autogenerated name).
///
///   ### Returns
///     A `Tensor` of the same type as `a` and `b` where each inner-most matrix is
///     the product of the corresponding matrices in `a` and `b`, e.g. if all
///     transpose or adjoint attributes are `False`:
///
///     `output`[..., i, j] = sum_k (`a`[..., i, k] * `b`[..., k, j]),
///     for all indices i, j.
///
///     __Note:__ This is matrix product, not element-wise product.
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
) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let scope = &mut context.name_scope(name.as_ref(), Some("MatMul".as_ref()));
    if transpose_a && adjoint_a {
        return Err(Error::from(
            "Only one of transpose_a and adjoint_a can be True.".to_owned(),
        ));
    }
    if transpose_b && adjoint_b {
        return Err(Error::from(
            "Only one of transpose_b and adjoint_b can be True.".to_owned(),
        ));
    }

    let mut a = a.into_tensor(context);
    let mut b = b.into_tensor(context);
    let a_shape = a.get_shape(scope);
    let b_shape = b.get_shape(scope);
    if (!a_is_sparse && !b_is_sparse)
        && ((a_shape.dims().is_none() || a_shape.dims().unwrap() > 2)
            && (b_shape.dims().is_none() || b_shape.dims().unwrap() > 2))
    {
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
        DataType::BFloat16 | DataType::Float => true,
        _ => false,
    };
    let sparse_matmul_b = match b.dtype {
        DataType::BFloat16 | DataType::Float => true,
        _ => false,
    };

    let use_sparse_matmul = if [a.dtype, b.dtype]
        .iter()
        .any(|x| DataType::BFloat16 == *x) { 
            true 
    } else { 
        sparse_matmul_a && sparse_matmul_b && (a_is_sparse || b_is_sparse) 
    };

    if use_sparse_matmul {
        //sparse_matmul(scope, a, b, transpose_a, transpose_b, a_is_sparse, b_is_sparse, "");
        scope.install(
            SparseMatMul::new(a, b, "")?
                .transpose_a(&[transpose_a])
                .transpose_b(&[transpose_b])
                .a_is_sparse(&[a_is_sparse])
                .b_is_sparse(&[b_is_sparse]),
        )
    } else {
        if a.dtype != b.dtype {
            return Err(Error::from(
                "Matrix a and matrix b must be of the same type.".to_owned(),
            ));
        }
        scope.install(
            MatMul::new(a, b, "")?
                .transpose_a(&[transpose_a])
                .transpose_b(&[transpose_b]),
        )
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

///// Neg /////

///   Computes numerical negative value element-wise.
///
///   I.e., `y = -x`.
///
///   ### Args:
///     *x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
///     *name: A name for the operation (optional).
///
///   ### Returns:
///     A `Tensor`. Has the same type as `x`.
pub fn negative<Tx, S>(context: &mut Scope, x: Tx, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    context.install(Neg::new(x, name)?)
}

add_new_op!(Neg,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: Neg, Init: []);],
    digest: [DEFAULT_DIGEST: Neg, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

/// Multiply matrix "a" by matrix "b".
///
/// The inputs must be two-dimensional matrices and the inner dimension of "a" must
/// match the outer dimension of "b". This op is optimized for the case where at
/// least one of "a" or "b" is sparse. The breakeven for using this versus a dense
/// matrix multiply on one platform was 30% zero values in the sparse matrix.
///
/// The gradient computation of this operation will only take advantage of sparsity
/// in the input gradient when that gradient comes from a Relu.
///
/// transpose_a: If true, "a" is transposed before multiplication.
/// transpose_b: If true, "b" is transposed before multiplication.
/// a_is_sparse: If true, `a` is treated as a sparse matrix.
/// b_is_sparse: If true, `b` is treated as a sparse matrix.
add_new_op!(SparseMatMul,
    constructor: [add_new_op!(BIN CONSTRUCTOR: SparseMatMul, Init: []);],
    digest: [DEFAULT_DIGEST: SparseMatMul, INPUT0],
    extra_funcs: [
        fn transpose_a(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("transpose_a", false, Attribute::Bool(val)));
            self
        }

        fn transpose_b(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("transpose_b", false, Attribute::Bool(val)));
            self
        }

        fn a_is_sparse(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("transpose_b", false, Attribute::Bool(val)));
            self
        }

        fn b_is_sparse(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("transpose_b", false, Attribute::Bool(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);

///// Multiply /////

/// Returns x * y element-wise.
pub fn multiply<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);
    context.install(Mul::new(x, y, name)?)
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

/// Returns the truth value of (x < y) element-wise.
///
/// __NOTE:__ Greater supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
///
/// ### Args:
/// * x: A Tensor. Must be one of the following types: float32, float64, int32, int64, uint8, int16, int8, uint16, half.
/// * y: A Tensor. Must have the same type as x.
/// * name: A name for the operation (empty string slice for autogenerated name).
///
/// ### Returns:
/// * A Tensor of type bool.
pub fn less<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);
    context.install(Less::new(x, y, name)?)
}

add_new_op!(Less,
    constructor: [
        add_new_op!(BIN CONSTRUCTOR: Less, Init: [output_type: DataType::Bool]);
    ],
    digest: [DEFAULT_DIGEST: Less, DTYPE_ATTR],
    extra_funcs: [], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_less() {
    let mut context = Scope::new();
    let x = context.constant(&[2_i32, 2], [2].as_ref(), "x").unwrap();
    let y = context.constant(&[1_i32, 3], [2].as_ref(), "y").unwrap();
    let op = less(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [false, true]});
}

///// Log /////

/// Computes natural logarithm of x element-wise.
///
/// ### Args:
/// * x: A Tensor. Must be one of the following types: half, float32, float64, complex64, complex128.
/// * name: A name for the operation (empty string slice for autogenerated name).
pub fn log<Tx, S>(context: &mut Scope, x: Tx, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    context.install(Log::new(x, name)?)
}

add_new_op!(Log, 
    constructor: [add_new_op!(UNARY CONSTRUCTOR: Log, Init: []);],
    digest: [DEFAULT_DIGEST: Log, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_log() {
    let mut context = Scope::new();
    let e = context
        .constant(&[::std::f64::consts::E], &[] as &[i32], "e")
        .unwrap();
    let op = log(&mut context, e, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Double] == [1.]});
}

///// Logical Not /////

/// Returns the truth value of NOT x element-wise.
///
/// ### Args:
/// * x: A Tensor of type bool.
/// * name: A name for the operation (empty string slice for autogenerated name).
///
/// ### Returns
/// * A Tensor of type bool.
pub fn logical_not<Tx, S>(context: &mut Scope, tensor: Tx, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let tensor = tensor.into_tensor(context);
    if tensor.dtype != DataType::Bool {
        return Err(Error::from(ErrorKind::Stub));
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
    let x = context.constant(&[true, false], [2].as_ref(), "x").unwrap();
    let op = logical_not(&mut context, x, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [false, true]});
}

///// Pow /////

/// Computes the power of one value to another.
///
/// Given a tensor `x` and a tensor `y`, this operation computes x^y for
/// corresponding elements in `x` and `y`. For example:
///
/// ```python
/// # tensor 'x' is [[2, 2]], [3, 3]]
/// # tensor 'y' is [[8, 16], [2, 3]]
/// pow(x, y) ==> [[256, 65536], [9, 27]]
/// ```
///
/// ### Args:
/// * x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
/// * y: A `Tensor`. Must have the same type as `x`.
/// * name: A name for the operation (empty string slice for autogenerated name).
///
/// ### Returns
/// * A `Tensor`. Has the same type as `x`.
pub fn pow<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);
    context.install(Pow::new(x, y, name)?)
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

/// Computes the "logical and" of elements across dimensions of a tensor.
///
/// Reduces input_tensor along the dimensions given in axis.
/// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each entry in axis.
/// If keep_dims is true, the reduced dimensions are retained with length 1.
///
/// If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
///
/// For example:
/// ```python
/// # 'x' is [[True,  True]
/// #         [False, False]]
/// tf.reduce_all(x) ==> False
/// tf.reduce_all(x, 0) ==> [False, False]
/// tf.reduce_all(x, 1) ==> [True, False]
/// ```
///
/// ### Args:
/// * input_tensor: The boolean tensor to reduce.
/// * axis: The dimensions to reduce. If empty, reduces all dimensions.
/// * keep_dims: If true, retains reduced dimensions with length 1.
/// * name: A name for the operation (empty string slice for autogenerated name).
pub fn reduce_all<Tx, TeS, S>(
    context: &mut Scope,
    tensor: Tx,
    axis: &[TeS],
    keep_dims: bool,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    TeS: ShapeSize,
    S: AsRef<Path>,
{
    let dims = [axis.len() as i64];
    let reduce = context.constant(axis, dims.as_ref(), name.as_ref())?;
    let tensor = tensor.into_tensor(context);
    context.install(All::new(tensor, reduce.into(), name)?.keep_dims(&[keep_dims]))
}

add_new_op!(All,
    constructor: [add_new_op!(BIN CONSTRUCTOR: All, Init: []);],
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
    let x = context
        .constant(&[true, true, true, false, true, true], [2, 3].as_ref(), "x")
        .unwrap();

    let op = reduce_all(&mut context, x, [1].as_ref(), false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [true, false]});
    test_suite!(results; assert_len: {[0;Bool] == 2});

    let op = reduce_all(&mut context, x, [0].as_ref(), false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [false, true, true]});
    test_suite!(results; assert_len: {[0;Bool] == 3});

    let op = reduce_all(&mut context, x, [0, 1].as_ref(), false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [false]});
    test_suite!(results; assert_len: {[0;Bool] == 1});
}

///// Reduce LogSumExp /////

/// Computes log(sum(exp(elements across dimensions of a tensor))).
///
/// Reduces input_tensor along the dimensions given in axis.
/// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each entry in axis.
/// If keep_dims is true, the reduced dimensions are retained with length 1.
///
/// If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
///
/// This function is more numerically stable than log(sum(exp(input))).
/// It avoids overflows caused by taking the exp of large inputs and underflows caused by taking the log of small inputs.
///
/// For example:
/// ```python
/// # 'x' is [[0, 0, 0]]
/// #         [0, 0, 0]]
/// tf.reduce_logsumexp(x) ==> log(6)
/// tf.reduce_logsumexp(x, 0) ==> [log(2), log(2), log(2)]
/// tf.reduce_logsumexp(x, 1) ==> [log(3), log(3)]
/// tf.reduce_logsumexp(x, 1, keep_dims=True) ==> [[log(3)], [log(3)]]
/// tf.reduce_logsumexp(x, [0, 1]) ==> log(6)
/// ```
///
/// ### Args:
/// * input: The tensor to reduce. Should have numeric type.
/// * axis: The dimensions to reduce. If empty reduces all dimensions.
/// * keep_dims: If true, retains reduced dimensions with length 1.
/// * name: A name for the operation (empty string slice for autogenerated name).
pub fn reduce_logsumexp<Tx, TeS, S>(
    context: &mut Scope,
    input: Tx,
    axis: &[TeS],
    keep_dims: bool,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    TeS: ShapeSize,
    S: AsRef<Path>,
{
    let scope = &mut context.name_scope(name.as_ref(), Some("ReduceLogSumExp".as_ref()));
    let input = input.into_tensor(scope);
    let axis_t = axis.into_tensor(scope);

    let reduce_max = reduce_max(scope, input, axis_t, true, "")?;
    let my_max = array_ops::stop_gradient(scope, reduce_max, "")?;

    let sub = sub(scope, input, my_max, "")?;
    let exp = exp(scope, sub, "")?;
    let reduce_sum = reduce_sum(scope, exp, axis_t, true, "")?;
    let log = log(scope, reduce_sum, "")?;

    let dims: Vec<i64>;
    let mut result = add(scope, log, my_max, "")?;
    if !keep_dims {
        let axis = if axis.is_empty() { None } else { Some(axis) };
        result = squeeze(scope, result, axis, "")?;
    }
    Ok(result)
}

#[test]
#[cfg(test)]
fn test_reduce_logsumexp() {
    let mut context = Scope::new();
    let x = context
        .constant(&[0_f64, 0., 0., 0., 0., 0.], [2, 3].as_ref(), "x")
        .unwrap();

    let op1 = reduce_logsumexp(&mut context, x, [0_i32, 1].as_ref(), false, "").unwrap();
    let results = test_suite!(run_op: [op1]; context, input: {});
    test_suite!(results; assert_len: {[0;Double] == 1});
    test_suite!(results; assert: {[0;Double] == [6_f64.ln()]});

    let op2 = reduce_logsumexp(&mut context, x, [1].as_ref(), true, "").unwrap();
    let results = test_suite!(run_op: [op2]; context, input: {});
    test_suite!(results; assert_len: {[0;Double] == 2});
    test_suite!(results; assert: {[0;Double] == [3_f64.ln(), 3_f64.ln()]});
}

///// Reduce Sum /////

/// Computes the sum of elements across dimensions of a tensor.
///
/// Reduces input_tensor along the dimensions given in axis.
/// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each entry in axis.
/// If keep_dims is true, the reduced dimensions are retained with length 1.
///
/// If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
///
/// For example:
/// ```python
/// # 'x' is [[1, 1, 1]
/// #         [1, 1, 1]]
/// tf.reduce_sum(x) ==> 6
/// tf.reduce_sum(x, 0) ==> [2, 2, 2]
/// tf.reduce_sum(x, 1) ==> [3, 3]
/// tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
/// tf.reduce_sum(x, [0, 1]) ==> 6
/// ```
///
/// ### Args:
/// * input: The boolean tensor to reduce.
/// * axis: The dimensions to reduce. If empty, reduces all dimensions.
/// * keep_dims: If true, retains reduced dimensions with length 1.
/// * name: A name for the operation (empty string slice for autogenerated name).
pub fn reduce_sum<Tx, Ty, S>(
    context: &mut Scope,
    input: Tx,
    axis: Ty,
    keep_dims: bool,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let input = input.into_tensor(context);
    let axis = axis.into_tensor(context);
    context.install(Sum::new(input, axis, name)?.keep_dims(&[keep_dims]))
}

add_new_op!(Sum, 
    constructor: [add_new_op!(BIN CONSTRUCTOR: Sum, Init: []);], 
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
    let x = context
        .constant(&[1_i32, 2, 3, 4], [2, 2].as_ref(), "x")
        .unwrap();

    let op = reduce_sum(&mut context, x, [0, 1].as_ref(), false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert_len: {[0;Int32] == 1});
    test_suite!(results; assert: {[0;Int32] == [10]});
}

///// Reduce Mean /////

/// Computes the mean of elements across dimensions of a tensor.
///
/// Reduces input_tensor along the dimensions given in axis.
/// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each entry in axis.
/// If keep_dims is true, the reduced dimensions are retained with length 1.
///
/// If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
///
/// For example:
/// ```python
/// # 'x' is [[1., 1.]
/// #         [2., 2.]]
/// tf.reduce_mean(x) ==> 1.5
/// tf.reduce_mean(x, 0) ==> [1.5, 1.5]
/// tf.reduce_mean(x, 1) ==> [1.,  2.]
/// ```
///
/// ### Args:
/// * input: The tensor to reduce. Should have numeric type.
/// * axis: The dimensions to reduce. If empty, reduces all dimensions.
/// * keep_dims: If true, retains reduced dimensions with length 1.
/// * name: A name for the operation (empty string slice for autogenerated name).
pub fn reduce_mean<TeS, Tx, S>(
    context: &mut Scope,
    input: Tx,
    axis: &[TeS],
    keep_dims: bool,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
    TeS: ShapeSize,
{
    let dims = [axis.len() as i64];
    let reduce = context.constant(axis, dims.as_ref(), name.as_ref())?;
    let input = input.into_tensor(context);
    context.install(Mean::new(input, reduce.into(), name)?.keep_dims(&[keep_dims]))
}

add_new_op!(Mean, 
    constructor: [add_new_op!(BIN CONSTRUCTOR: Mean, Init: []);],
    digest: [DEFAULT_DIGEST: Mean, INPUT0],
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
fn test_reduce_mean() {
    let mut context = Scope::new();
    let x = context
        .constant(&[1.0_f32, 1., 2., 2.], [2, 2].as_ref(), "x")
        .unwrap();

    let op = reduce_mean(&mut context, x, [0, 1].as_ref(), false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert_len: {[0;Float] == 1});
    test_suite!(results; assert: {[0;Float] == [1.5]});
}

///// Range /////

///  Creates a sequence of numbers.
///
///  Creates a sequence of numbers that begins at `start` and extends by
///  increments of `delta` up to but not including `limit`.
///
///  The dtype of the resulting tensor is inferred from the start tensor.
///
///  Like the Python builtin `range`, `start` defaults to 0, so that
///  `range(n) = range(0, n)`.
///
///  For example:
///
///  ```python
///  # 'start' is 3
///  # 'limit' is 18
///  # 'delta' is 3
///  tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
///
///  # 'start' is 3
///  # 'limit' is 1
///  # 'delta' is -0.5
///  tf.range(start, limit, delta) ==> [3, 2.5, 2, 1.5]
///
///  # 'limit' is 5
///  tf.range(limit) ==> [0, 1, 2, 3, 4]
///  ```
///
///  ### Args:
///    * start: A 0-D `Tensor` (scalar). First entry in the range.
///    * limit: A 0-D `Tensor` (scalar). Upper limit of sequence, exclusive.
///    * delta: A 0-D `Tensor` (scalar). Number that increments `start`.
///    * name: A name for the operation (if empty defaults to "Range").
///
///  ### Returns
///    * An 1-D `Tensor` of type of `start`.
pub fn range<Ts, Tl, Td, S>(
    context: &mut Scope,
    start: Ts,
    limit: Tl,
    delta: Td,
    name: S,
) -> Result<Tensor>
where
    Ts: TensorOps,
    Tl: TensorOps,
    Td: TensorOps,
    S: AsRef<Path>,
{
    let scope = &mut context.name_scope(name.as_ref(), Some("Range".as_ref()));
    let start = start.into_tensor(scope);
    let limit = start.into_tensor(scope);
    let delta = start.into_tensor(scope);
    scope.install(Range::new(start, limit, delta, name)?)
}

add_new_op!(Range,
    constructor: [
        fn new<S: AsRef<Path>>(start: Tensor, limit: Tensor, delta: Tensor, name: S) 
            -> Result<Range<'a>> 
        {
            Ok(
                Range {
                    ident: NodeIdent::new(),
                    elements: vec![start, limit, delta],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: Vec::with_capacity(0),
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: Range, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

///// Reduce Max /////

/// Computes the maximum of elements across dimensions of a tensor.
///
/// Reduces input_tensor along the dimensions given in axis.
/// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each entry in axis.
/// If keep_dims is true, the reduced dimensions are retained with length 1.
///
/// If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
///
/// ### Args:
/// * input: The boolean tensor to reduce.
/// * axis: The dimensions to reduce. If empty, reduces all dimensions.
/// * keep_dims: If true, retains reduced dimensions with length 1.
/// * name: A name for the operation (empty string slice for autogenerated name).
pub fn reduce_max<Tx, Ty, S>(
    context: &mut Scope,
    input: Tx,
    axis: Ty,
    keep_dims: bool,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let input = input.into_tensor(context);
    let axis = axis.into_tensor(context);
    context.install(Max::new(input, axis, name)?.keep_dims(&[keep_dims]))
}

add_new_op!(Max, 
    constructor: [
        fn new<S: AsRef<Path>>(input: Tensor, axis: Tensor, name: S) -> Result<Max<'a>> {
            if axis.dtype != DataType::Int32 &&
               axis.dtype != DataType::Int64 {
                return Err(Error::from(ErrorKind::Stub));
            }
            Ok(
                Max {
                    ident: NodeIdent::new(),
                    elements: vec![input, axis],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: Vec::with_capacity(0),
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
    let x = context
        .constant(&[1_i32, 2, 3, 4], [2, 2].as_ref(), "x")
        .unwrap();

    let op = reduce_max(&mut context, x, [0, 1].as_ref(), false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert_len: {[0;Int32] == 1});
    test_suite!(results; assert: {[0;Int32] == [4]});
}

///// Reduce Min /////

/// Computes the minimum of elements across dimensions of a tensor.
///
/// Reduces input_tensor along the dimensions given in axis.
/// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each entry in axis.
/// If keep_dims is true, the reduced dimensions are retained with length 1.
///
/// If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
///
/// ### Args:
/// * input: The boolean tensor to reduce.
/// * axis: The dimensions to reduce. If empty, reduces all dimensions.
/// * keep_dims: If true, retains reduced dimensions with length 1.
/// * name: A name for the operation (empty string slice for autogenerated name).
pub fn reduce_min<TeS, Tx, S>(
    context: &mut Scope,
    tensor: Tx,
    axis: &[TeS],
    keep_dims: bool,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
    TeS: ShapeSize,
{
    let dims = [axis.len() as i64];
    let reduce = context.constant(axis, dims.as_ref(), name.as_ref())?;
    let tensor = tensor.into_tensor(context);
    context.install(Min::new(tensor, reduce.into(), name)?.keep_dims(&[keep_dims]))
}

add_new_op!(Min, 
    constructor: [
        fn new<S: AsRef<Path>>(input: Tensor, axis: Tensor, name: S) -> Result<Min<'a>> {
            if axis.dtype != DataType::Int32 &&
               axis.dtype != DataType::Int64 {
                return Err(Error::from(ErrorKind::Stub));
            }
            Ok(
                Min {
                    ident: NodeIdent::new(),
                    elements: vec![input, axis],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: Vec::with_capacity(0),
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
fn test_reduce_min() {
    let mut context = Scope::new();
    let x = context
        .constant(&[1_i32, 2, 3, 4], [2, 2].as_ref(), "x")
        .unwrap();

    let op = reduce_min(&mut context, x, [0, 1].as_ref(), false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert_len: {[0;Int32] == 1});
    test_suite!(results; assert: {[0;Int32] == [1]});
}

///// Tanh /////

///  Computes hyperbolic tangent of `x` element-wise.
///
///  ### Args:
///    * x: A Tensor with type `float`, `double`, `int32`,
///      `complex64`, `int64`, or `qint32`.
///    * name: A name for the operation (empty string slice for autogenerated name).
///
///  ### Returns
///    A Tensor respectively with the same type as `x` if
///    `x.dtype != qint32` otherwise the return type is `quint8`.
pub fn tanh<Tx, S>(context: &mut Scope, tensor: Tx, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let scope = &mut context.name_scope(name.as_ref(), Some("Tanh".as_ref()));
    let tensor = tensor.into_tensor(context);
    scope.install(Tanh::new(tensor, name)?)
}

add_new_op!(Tanh,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: Tanh, Init: []);],
    digest: [DEFAULT_DIGEST: Tanh, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

///// Minimum /////

/// Returns the min of x and y (i.e. `x < y ? x : y`) element-wise.
///
/// __NOTE__: `Minimum` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
///
/// ### Args:
/// * x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
/// * y: A `Tensor`. Must have the same type as `x`.
/// * name: A name for the operation (empty string slice for autogenerated name).
///
/// ### Returns
/// A `Tensor`. Has the same type as `x`.
pub fn minimum<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);
    context.install(Minimum::new(x, y, name)?)
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

///// Rsqrt /////

/// Computes reciprocal of square root of x element-wise.
///
/// I.e., `y = 1 / sqrt{x}`.
///
/// ### Args:
///   * x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
///   * name: A name for the operation (optional).
///
/// ### Returns:
///   A `Tensor`. Has the same type as `x`.
pub fn rsqrt<Tx, S>(context: &mut Scope, x: Tx, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    context.install(Rsqrt::new(x, name)?)
}

add_new_op!(Rsqrt,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: Rsqrt, Init: []);],
    digest: [DEFAULT_DIGEST: Rsqrt, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

///// Select /////

///  Selects elements from `x` or `y`, depending on `condition`.
///
///  The `x`, and `y` tensors must all have the same shape, and the
///  output will also have that shape.
///
///  The `condition` tensor must be a scalar if `x` and `y` are scalars.
///  If `x` and `y` are vectors or higher rank, then `condition` must be either a
///  scalar, a vector with size matching the first dimension of `x`, or must have
///  the same shape as `x`.
///
///  The `condition` tensor acts as a mask that chooses, based on the value at each
///  element, whether the corresponding element / row in the output should be
///  taken from `x` (if true) or `y` (if false).
///
///  If `condition` is a vector and `x` and `y` are higher rank matrices, then
///  it chooses which row (outer dimension) to copy from `x` and `y`.
///  If `condition` has the same shape as `x` and `y`, then it chooses which
///  element to copy from `x` and `y`.
///
///  For example:
///
///  ```python
///  # 'condition' tensor is [[True,  False]
///  #                        [False, True]]
///  # 't' is [[1, 2],
///  #         [3, 4]]
///  # 'e' is [[5, 6],
///  #         [7, 8]]
///  select(condition, t, e)  # => [[1, 6], [7, 4]]
///
///
///  # 'condition' tensor is [True, False]
///  # 't' is [[1, 2],
///  #         [3, 4]]
///  # 'e' is [[5, 6],
///  #         [7, 8]]
///  select(condition, t, e) ==> [[1, 2],
///                               [7, 8]]
///
///  ```
///
///  ### Args:
///    * condition: A `Tensor` of type `bool`.
///    * x:  A `Tensor` which may have the same shape as `condition`.
///      If `condition` is rank 1, `x` may have higher rank,
///      but its first dimension must match the size of `condition`.
///    * y:  A `Tensor` with the same type and shape as `x`.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    A `Tensor` with the same type and shape as `t` and `e`.
pub(crate) fn select<Tc, Tx, Ty, S>(
    context: &mut Scope,
    condition: Tc,
    x: Tx,
    y: Ty,
    name: S,
) -> Result<Tensor>
where
    Tc: TensorOps,
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);
    let condition = condition.into_tensor(context);
    context.install(Select::new(x, y, condition, name)?)
}

add_new_op!(Select,
    constructor: [
        fn new<S: AsRef<Path>>(x: Tensor, y: Tensor, cond: Tensor, name: S) -> Result<Select<'a>> {
            let output_type = x.dtype;
            Ok(
                Select {
                    ident: NodeIdent::new(),
                    name: generate_name!(is_none: name),
                    attributes: Vec::with_capacity(0),
                    elements: vec![x, y, cond],
                    input_lists: Vec::with_capacity(0),
                    output_type,
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: Select, INPUT0],
    extra_funcs: [], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);

///// Sub /////

/// Returns x - y element-wise.
///
/// __NOTE__: sub supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
///
/// ### Args:
/// * x: A Tensor. Must be one of the following types: half, float32, float64, int32, int64, complex64, complex128.
/// * y: A Tensor. Must have the same type as x.
/// * name: A name for the operation (empty string slice for autogenerated name).
pub fn sub<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);
    context.install(Sub::new(x, y, name)?)
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

///// Square /////

///  Computes square of x element-wise.
///
///  I.e., `(y = x * x = x^2)`.
///
///  ### Args:
///    * x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    A `Tensor`. Has the same type as `x`.
pub fn square<Tx, S>(context: &mut Scope, x: Tx, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    context.install(Square::new(x, name)?)
}

add_new_op!(Square,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: Square, Init: []);],
    digest: [DEFAULT_DIGEST: Square, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

///// Squared Difference /////

///   Returns (x - y)(x - y) element-wise.
///
///   __NOTE__: `SquaredDifference` supports broadcasting. More about broadcasting
///   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
///   ### Args:
///     x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
///     y: A `Tensor`. Must have the same type as `x`.
///     name: A name for the operation (optional).
///
///   ### Returns:
///     A `Tensor`. Has the same type as `x`.
pub fn squared_difference<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);
    context.install(SquaredDifference::new(x, y, name)?)
}

add_new_op!(SquaredDifference,
    constructor: [add_new_op!(BIN CONSTRUCTOR: SquaredDifference, Init: []);],
    digest: [DEFAULT_DIGEST: SquaredDifference, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

/// Computes the sum along segments of a tensor.
///
/// Computes a tensor such that
/// `(output[i] = sum_{j...} data[j...]` where the sum is over tuples `j...` such
/// that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
/// need not be sorted and need not cover all values in the full
/// range of valid values.
///
/// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
///
/// `num_segments` should equal the number of distinct segment IDs.
///
/// Returns a `Tensor` that has the same type as `data`.
/// Has same shape as data, except for the first `segment_ids.rank`
/// dimensions, which are replaced with a single dimension which has size
/// `num_segments`.
pub fn unsorted_segment_sum<Tx, Ty, Tz, S>(
    context: &mut Scope,
    data: Tx,
    segment_ids: Ty,
    num_segments: Tz,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    Tz: TensorOps,
    S: AsRef<Path>,
{
    let data = data.into_tensor(context);
    let segment_ids = segment_ids.into_tensor(context);
    let num_segments = num_segments.into_tensor(context);
    context.install(UnsortedSegmentSum::new(
        data,
        segment_ids,
        num_segments,
        name,
    )?)
}

add_new_op!(UnsortedSegmentSum,
    constructor: [
        fn new<S: AsRef<Path>>(
            data: Tensor, 
            segment_ids: Tensor, 
            num_segments: Tensor, name: S
        ) -> Result<UnsortedSegmentSum<'a>> {
            let output_type = data.dtype;
            Ok(
                UnsortedSegmentSum {
                    ident: NodeIdent::new(),
                    name: generate_name!(is_none: name),
                    attributes: Vec::with_capacity(0),
                    elements: vec![data, segment_ids, num_segments],
                    input_lists: Vec::with_capacity(0),
                    output_type: output_type,
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: UnsortedSegmentSum, DTYPE_ATTR],
    extra_funcs: [], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);

//! Math Operations.
use super::*;


///// Add /////

pub fn add<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>,
          Ty: Into<Tensor>,
          S: AsRef<Path>
{
    context.install(Add::new(x.into(), y.into(), name)?)
}

/// Returns x + y element-wise.
///
/// Both x and y must be tensors of the same type. Name argument is optional,
/// if an empty string is provided, the name will be generated automatically.
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
    let x = context.constant("x", &[2_i32], &[] as &[i32]).unwrap();
    let y = context.constant("y", &[2_i32], &[] as &[i32]).unwrap();
    let op = add(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [4_i32]});
}


///// AddN /////

pub fn add_n<S>(context: &mut Scope, values: Vec<Tensor>, name: S) -> Result<Tensor, ::Error>
    where S: AsRef<Path>
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
                    ident: Ident::new(),
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
    let x = context.constant("x", &[2_i32], &[] as &[i32]).unwrap();
    let y = context.constant("y", &[2_i32], &[] as &[i32]).unwrap();
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
    where Tx: Into<Tensor>,
          S: AsRef<Path>
{
    context.install(Cast::new(tensor.into(), &[ty], name)?)
}

pub fn to_float<Tx, S>(context: &mut Scope, tensor: Tx, name: S) -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>,
          S: AsRef<Path>
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
                    ident: Ident::new(),
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
    let x = context.constant("x", &[0_i32, 1], &[2]).unwrap();
    let op = cast(&mut context, x, DataType::Double, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Double] == [0_f64, 1.]});
}

///// Division /////

pub fn divide<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>,
          Ty: Into<Tensor>,
          S: AsRef<Path>
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
    let x = context.constant("x", &[4_i32], &[] as &[i32]).unwrap();
    let y = context.constant("y", &[2_i32], &[] as &[i32]).unwrap();
    let op = divide(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [2_i32]});
}


///// Equal /////

pub fn equal<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>,
          Ty: Into<Tensor>,
          S: AsRef<Path>
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
    let x = context.constant("x", &[1_i32, 2], &[2]).unwrap();
    let y = context.constant("y", &[1_i32, 2], &[2]).unwrap();
    let op = equal(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [true, true]});
}


///// Exp /////

pub fn exp<Tx, S>(context: &mut Scope, x: Tx, name: S) -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>,
          S: AsRef<Path>
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
    let e = context.constant("e", &[1_f64], &[] as &[i32]).unwrap();
    let op = exp(&mut context, e, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Double] == [::std::f64::consts::E]});
}


///// Greater /////

pub fn greater<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>,
          Ty: Into<Tensor>,
          S: AsRef<Path>
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
    let x = context.constant("x", &[1_i32, 2], &[2]).unwrap();
    let y = context.constant("y", &[0_i32, 3], &[2]).unwrap();
    let op = greater(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [true, false]});
}


///// Multiply /////

pub fn multiply<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>,
          Ty: Into<Tensor>,
          S: AsRef<Path>
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
    let x = context.constant("x", &[4_i32], &[] as &[i32]).unwrap();
    let y = context.constant("y", &[2_i32], &[] as &[i32]).unwrap();
    let op = multiply(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [8_i32]});
}


///// Less /////

pub fn less<Tx, Ty, S>(context: &mut Scope, x: Tx, y: Ty, name: S) -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>,
          Ty: Into<Tensor>,
          S: AsRef<Path>
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
    let x = context.constant("x", &[2_i32, 2], &[2]).unwrap();
    let y = context.constant("y", &[1_i32, 3], &[2]).unwrap();
    let op = less(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Bool] == [false, true]});
}


///// Log /////

pub fn log<Tx, S>(context: &mut Scope, x: Tx, name: S) -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>,
          S: AsRef<Path>
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
    let e = context.constant("e", &[::std::f64::consts::E], &[] as &[i32]).unwrap();
    let op = log(&mut context, e, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Double] == [1.]});
}


///// Logical Not /////

pub fn logical_not<Tx, S>(context: &mut Scope, tensor: Tx, name: S) -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>,
          S: AsRef<Path>
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
    let x = context.constant("x", &[true, false], &[2]).unwrap();
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
    where Tx: Into<Tensor>,
          Ty: Into<Tensor>,
          S: AsRef<Path>
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
    let x = context.constant("x", &[2_i32], &[] as &[i32]).unwrap();
    let y = context.constant("y", &[2_i32], &[] as &[i32]).unwrap();
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
    where Tx: Into<Tensor>,
          S: AsRef<Path>
{
    if axis.len() == 0 {
        // TODO: infer reduction to scalar
    }
    let dims = &[axis.len() as i64];
    let reduce = context.constant(name.as_ref(), axis, dims)?;
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
                    ident: Ident::new(),
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
    let x = context.constant("x", &[true, true, true, false, true, true], &[2, 3]).unwrap();

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
    where Tx: Into<Tensor>,
          S: AsRef<Path>,
          TeS: ShapeSize
{
    let name = if name_cmp!(name, "") {
        Path::new("ReduceLogSumExp")
    } else {
        name.as_ref()
    };
    let scope = &mut context.name_scope(name);
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
    let x = context.constant("x", &[0_f64, 0., 0., 0., 0., 0.], &[2, 3]).unwrap();

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
    where Tx: Into<Tensor>,
          S: AsRef<Path>,
          TeS: ShapeSize
{
    if axis.len() == 0 {
        // TODO: infer reduction to scalar
    }
    let dims = &[axis.len() as i64];
    let reduce = context.constant(name.as_ref(), axis, dims)?;
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
                    ident: Ident::new(),
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
    let x = context.constant("x", &[1_i32, 2, 3, 4], &[2, 2]).unwrap();

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
    where Tx: Into<Tensor>,
          S: AsRef<Path>,
          TeS: ShapeSize
{
    if axis.len() == 0 {
        // TODO: infer reduction to scalar
    }
    let dims = &[axis.len() as i64];
    let reduce = context.constant(name.as_ref(), axis, dims)?;
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
                    ident: Ident::new(),
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
    let x = context.constant("x", &[1_i32, 2, 3, 4], &[2, 2]).unwrap();

    let op = reduce_max(&mut context, x, &[0, 1], false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert_len: {[0;Int32] == 1});
    test_suite!(results; assert: {[0;Int32] == [4]});
}


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
    where Tx: Into<Tensor>,
          Ty: Into<Tensor>,
          S: AsRef<Path>
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
    let x = context.constant("x", &[4_i32, 3], &[2] as &[i32]).unwrap();
    let y = context.constant("y", &[2_i32, 6], &[2] as &[i32]).unwrap();

    let op = minimum(&mut context, x, y, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [4_i32, 6]});
}


///// Stop Gradient /////

pub fn stop_gradient<Tx, S>(context: &mut Scope, tensor: Tx, name: S) -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>,
          S: AsRef<Path>
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
    where Tx: Into<Tensor>,
          Ty: Into<Tensor>,
          S: AsRef<Path>
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
    let x = context.constant("x", &[4_i32], &[] as &[i32]).unwrap();
    let y = context.constant("y", &[2_i32], &[] as &[i32]).unwrap();
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
    where Tx: Into<Tensor>,
          Ty: Into<Tensor>,
          Tz: Into<Tensor>,
          S: AsRef<Path>
{
    unimplemented!()
}

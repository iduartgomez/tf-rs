//! Array Operations.
#[allow(unused_imports)]
use tf::Shape as TensorShape;

use super::*;

///// Concat /////

pub fn concat<S, TeS>(
    context: &mut Scope,
    values: Vec<Tensor>,
    axis: TeS,
    name: S,
) -> Result<Tensor, ::Error>
where
    S: AsRef<Path>,
    TeS: ShapeSize,
{
    let axis = context.constant(&[axis], &[] as &[i32], "")?;
    context.install(Concat::new(values, axis.into(), name)?)
}

type Concat<'a> = ConcatV2<'a>;

add_new_op!(ConcatV2, 
    constructor: [
        fn new<S: AsRef<Path>>(values: Vec<Tensor>, axis: Tensor, name: S,) -> Result<Concat<'a>, ::Error> {
            let output_type = values[0].dtype;
            for x in &values {
                if &x.dtype != &output_type {
                    return Err(::Error::Stub);
                }
            }

            Ok(
                Concat {
                    ident: NodeIdent::new(),
                    input_lists: vec![(0, values)],
                    elements: vec![axis],
                    name: generate_name!(is_none: name),
                    attributes: Vec::with_capacity(0),
                    output_type,
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: ConcatV2, DTYPE_ATTR],
    extra_funcs: [], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_concat() {
    let mut context = Scope::new();
    let t1 = context.constant(&[1_i32, 2, 3, 4, 5, 6], &[2, 3], "t1").unwrap().into();
    let t2 = context.constant(&[7_i32, 8, 9, 10, 11, 12], &[2, 3], "t2").unwrap().into();
    let op1 = concat(&mut context, vec![t1, t2], 0, "").unwrap();
    let op2 = concat(&mut context, vec![t1, t2], 1, "").unwrap();
    test_suite!(run_op: [op1, op2]; context, input: {});

    let (src_op1, idx1) = context.get_src_op(op1);
    let (src_op2, idx2) = context.get_src_op(op2);
    let g = context.unwrap_graph().unwrap();
    assert_eq!(
        g.tensor_shape(test_suite!(out: src_op1, idx1)).unwrap(),
        TensorShape::from(Some(vec![Some(4), Some(3)]))
    );
    assert_eq!(
        g.tensor_shape(test_suite!(out: src_op2, idx2)).unwrap(),
        TensorShape::from(Some(vec![Some(2), Some(6)]))
    );
}


///// ExpandDims /////

///   Inserts a dimension of 1 into a tensor's shape.
///
///   Given a tensor `input`, this operation inserts a dimension of 1 at the
///   dimension index `axis` of `input`'s shape. The dimension index `axis` starts
///   at zero; if you specify a negative number for `axis` it is counted backward
///   from the end.
///
///   This operation is useful if you want to add a batch dimension to a single
///   element. For example, if you have a single image of shape `[height, width,
///   channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
///   which will make the shape `[1, height, width, channels]`.
///
///   Other examples:
///
///   ```python
///   # 't' is a tensor of shape [2]
///   shape(expand_dims(t, 0)) ==> [1, 2]
///   shape(expand_dims(t, 1)) ==> [2, 1]
///   shape(expand_dims(t, -1)) ==> [2, 1]
///
///   # 't2' is a tensor of shape [2, 3, 5]
///   shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
///   shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
///   shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
///   ```
///
///   This operation requires that:
///
///   `-1-input.dims() <= dim <= input.dims()`
///
///   This operation is related to `squeeze()`, which removes dimensions of
///   size 1.
///
///   Args:
///     input: A `Tensor`.
///     axis: 0-D (scalar). Specifies the dimension index at which to
///       expand the shape of `input`.
///     name: The name of the output `Tensor`.
///     dim: 0-D (scalar). Equivalent to `axis`, to be deprecated.
///
///   Returns:
///     A `Tensor` with the same data as `input`, but its shape has an additional
///     dimension of size 1 added.
///
///   Raises:
///     ValueError: if both `dim` and `axis` are specified.
pub fn expand_dims<Tx, S, TeS>(
    context: &mut Scope,
    tensor: Tx,
    axis: TeS,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
    TeS: ShapeSize,
{
    let m = context.constant(&[axis], &[] as &[TeS], "")?;
    context.install(ExpandDims::new(tensor.into(), m.into(), name)?)
}

add_new_op!(ExpandDims,
    constructor: [add_new_op!(BIN CONSTRUCTOR: ExpandDims, Init: []);],
    digest: [DEFAULT_DIGEST: ExpandDims, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);


///// Gather /////

pub fn gather<Tx, Ty, S>(
    context: &mut Scope,
    params: Tx,
    indices: Ty,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    let indices = indices.into();
    if indices.dtype != DataType::Int32 && indices.dtype != DataType::Int64 {
        return Err(::Error::Stub);
    }
    context.install(Gather::new(params.into(), indices, name)?)
}

add_new_op!(Gather,
    constructor: [add_new_op!(BIN CONSTRUCTOR: Gather, Init: []);],
    digest: [DEFAULT_DIGEST: Gather, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_gather() {
    let mut context = Scope::new();
    let x = context.constant(&[0_i32, 1, 2, 3, 4, 5], &[6], "x").unwrap();
    let indices = context.constant(&[2_i32, 0, 2, 5], &[4], "gather").unwrap();
    let op = gather(&mut context, x, indices, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [2_i32, 0, 2, 5]});
}


///// Rank /////

///  Returns the rank of a tensor.
///
///  This operation returns an integer representing the rank of `input`.
///
///  For example:
///
///  ```python
///  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
///  # shape of tensor 't' is [2, 2, 3]
///  rank(t) ==> 3
///  ```
///
///  **Note**: The rank of a tensor is not the same as the rank of a matrix. The
///  rank of a tensor is the number of indices required to uniquely select each
///  element of the tensor. Rank is also known as "order", "degree", or "ndims."
///
///  Args:
///    input: A `Tensor` or `SparseTensor`.
///    name: A name for the operation (optional).
///
///  Returns:
///    A `Tensor` of type `int32`.
pub fn rank<Tx, S>(context: &mut Scope, input_tensor: Tx, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
{
    let scope = &mut context.name_scope(name.as_ref(), Some("Rank".as_ref()));
    let input_tensor = input_tensor.into();
    // optimize: encode the rank as a constant when possible.
    if let Some(ndim) = input_tensor.get_shape(scope).dims() {
        Ok(scope.constant(&[ndim as i32], &[] as &[i32], "")?.into())
    } else {
        context.install(Rank::new(input_tensor.into(), "")?)
    }
}

add_new_op!(Rank,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: Rank, 
        Init: [output_type: DataType::Int32]);
    ],
    digest: [DEFAULT_DIGEST: Rank, DTYPE_ATTR],
    extra_funcs: [], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);


///// Reshape /////

pub fn reshape<Tx, Ty, S>(
    context: &mut Scope,
    tensor: Tx,
    shape: Ty,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    /*
    let shape = {
        let dims: &[i64] = &[shape.len() as i64];
        context.constant("", shape, dims)?
    };
    */
    let shape = shape.into_tensor(context, "");
    context.install(Reshape::new(tensor.into(), shape, name)?)
}

add_new_op!(Reshape,
    constructor: [add_new_op!(BIN CONSTRUCTOR: Reshape, Init: []);],
    digest: [DEFAULT_DIGEST: Reshape, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_reshape() {
    let mut context = Scope::new();
    let x = context.constant(&[1_i32, 2, 3, 4, 5, 6, 7, 8, 9], &[9], "x").unwrap();
    let y = context.constant(&[1_i32, 2, 3, 4, 5, 6, 7, 8, 9], &[3, 3], "y").unwrap();

    let shape = context.constant(&[3, 3], &[2], "").unwrap();
    let op1 = reshape(&mut context, x, shape, "").unwrap();
    let (src_op1, idx1) = context.get_src_op(op1);

    let shape = context.constant(&[-1], &[1], "").unwrap();
    let op2 = reshape(&mut context, y, shape, "").unwrap();
    let (src_op2, idx2) = context.get_src_op(op2);

    let g = context.unwrap_graph().unwrap();
    assert_eq!(
        g.tensor_shape(test_suite!(out: src_op1, idx1)).unwrap(),
        TensorShape::from(Some(vec![Some(3), Some(3)]))
    );
    assert_eq!(
        g.tensor_shape(test_suite!(out: src_op2, idx2)).unwrap(),
        TensorShape::from(Some(vec![Some(9)]))
    );
}


///// Shape /////

pub fn shape<Tx, S>(
    context: &mut Scope,
    tensor: Tx,
    out_type: Option<DataType>,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
{
    let out_type = if let Some(val) = out_type {
        vec![val]
    } else {
        vec![]
    };
    context.install(Shape::new(tensor.into(), &out_type, name)?)
}

/// Returns the shape of a tensor.
///
/// This operation returns a 1-D integer tensor representing the shape of `input`.
add_new_op!(Shape,
    constructor: [
        fn new<S: AsRef<Path>>(tensor: Tensor, output_type: &'a [DataType], name: S) 
            -> Result<Shape<'a>, ::Error> 
        {
            let out;
            let attributes = if let Some(dtype) = output_type.get(0) {
                match *dtype {
                    DataType::Int64 => out = DataType::Int64,
                    DataType::Int32 => out = DataType::Int32,
                    _ => return Err(::Error::Stub),
                }
                vec![("out_type", false, Attribute::Type(output_type))]
            } else if output_type.len() > 0 {
                return Err(::Error::Stub);
            } else {
                out = DataType::Int32;
                Vec::with_capacity(0)
            };

            Ok(
                Shape {
                    ident: NodeIdent::new(),
                    elements: vec![tensor],
                    name: generate_name!(is_none: name),
                    input_lists: Vec::with_capacity(0),
                    attributes,
                    output_type: out,
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: ShapeOp, DTYPE_ATTR],
    extra_funcs: [], 
    extra_attr: [ output_type: DataType ],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_shape() {
    let mut context = Scope::new();
    let x = context.constant(&[1_i32, 2, 3, 4, 5, 6, 7, 8, 9], &[3, 3], "x").unwrap();

    let op = shape(&mut context, x, Some(DataType::Int64), "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int64] == [3, 3]});
}


///// Size /////

pub fn size<Tx, S>(context: &mut Scope, tensor: Tx, name: S) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(Size::new(tensor.into(), name)?)
}

add_new_op!(Size, 
    constructor: [add_new_op!(UNARY CONSTRUCTOR: Size, Init: []);],
    digest: [DEFAULT_DIGEST: Size, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_size() {
    let mut context = Scope::new();
    let x = context.constant(&[1_i32, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], &[2, 2, 3], "x").unwrap();
    let op = size(&mut context, x, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [12]});
}


///// Squeeze /////

pub fn squeeze<TeS, Tx, S>(
    context: &mut Scope,
    tensor: Tx,
    axis: Option<&[TeS]>,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    S: AsRef<Path>,
    TeS: ShapeSize,
{
    let dims: Vec<i64>;
    let mut squeeze = Squeeze::new(tensor.into(), name)?;
    if let Some(axis) = axis {
        dims = shape_as_i64(axis);
        squeeze = squeeze.squeeze_dims(&dims);
    }
    context.install(squeeze)
}

add_new_op!(Squeeze,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: Squeeze, Init: []);],
    digest: [DEFAULT_DIGEST: Squeeze, INPUT0],
    extra_funcs: [
        fn squeeze_dims(mut self, squeeze_dims: &'a [i64]) -> Self {
            self.attributes.push(
                ("squeeze_dims", 
                true, 
                Attribute::Int(squeeze_dims)
            ));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);


///// Slice /////

///  Extracts a slice from a tensor.
///
///  This operation extracts a slice of size `size` from a tensor `input` starting
///  at the location specified by `begin`. The slice `size` is represented as a
///  tensor shape, where `size[i]` is the number of elements of the 'i'th dimension
///  of `input` that you want to slice. The starting location (`begin`) for the
///  slice is represented as an offset in each dimension of `input`. In other
///  words, `begin[i]` is the offset into the 'i'th dimension of `input` that you
///  want to slice from.
///
///  `begin` is zero-based; `size` is one-based. If `size[i]` is -1,
///  all remaining elements in dimension i are included in the
///  slice. In other words, this is equivalent to setting:
///
///  `size[i] = input.dim_size(i) - begin[i]`
///
///  This operation requires that:
///
///  `0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n]`
///
///  For example:
///
///  ```python
///  # 'input' is [[[1, 1, 1], [2, 2, 2]],
///  #             [[3, 3, 3], [4, 4, 4]],
///  #             [[5, 5, 5], [6, 6, 6]]]
///  tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
///  tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
///                                              [4, 4, 4]]]
///  tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
///                                             [[5, 5, 5]]]
///  ```
///
///  Args:
///    input_: A `Tensor`.
///    begin: An `int32` or `int64` `Tensor`.
///    size: An `int32` or `int64` `Tensor`.
///    name: A name for the operation (optional).
///
///  Returns:
///    A `Tensor` the same type as `input`.
pub fn slice<Tx, Tb, Ts, S>(
    context: &mut Scope,
    input: Tx,
    begin: Tb,
    size: Ts,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    Tb: TensorOps,
    Ts: TensorOps,
    S: AsRef<Path>,
{
    let begin = begin.into_tensor(context, "");
    let size = size.into_tensor(context, "");
    let input = input.into();
    context.install(Slice::new(input, begin, size, name)?)
}

add_new_op!(Slice,
    constructor: [
        fn new<S: AsRef<Path>>(input: Tensor, begin: Tensor, size: Tensor, name: S) 
            -> Result<Slice<'a>, ::Error> 
        {
            Ok(
                Slice {
                    ident: NodeIdent::new(),
                    elements: vec![input, begin, size],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: vec![],
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: Transpose, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);


///// Transpose /////

///  Transposes `a`. Permutes the dimensions according to `perm`.
///
///  The returned tensor's dimension i will correspond to the input dimension
///  `perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
///  the rank of the input tensor. Hence by default, this operation performs a
///  regular matrix transpose on 2-D input Tensors.
///
///  For example:
///
///  ```python
///  # 'x' is [[1 2 3]
///  #         [4 5 6]]
///  tf.transpose(x) ==> [[1 4]
///                       [2 5]
///                       [3 6]]
///
///  # Equivalently
///  tf.transpose(x, perm=[1, 0]) ==> [[1 4]
///                                    [2 5]
///                                    [3 6]]
///
///  # 'perm' is more useful for n-dimensional tensors, for n > 2
///  # 'x' is   [[[1  2  3]
///  #            [4  5  6]]
///  #           [[7  8  9]
///  #            [10 11 12]]]
///  # Take the transpose of the matrices in dimension-0
///  tf.transpose(x, perm=[0, 2, 1]) ==> [[[1  4]
///                                        [2  5]
///                                        [3  6]]
///
///                                       [[7 10]
///                                        [8 11]
///                                        [9 12]]]
///  ```
///
///  Args:
///    a: A `Tensor`.
///    perm: A permutation of the dimensions of `a`.
///    name: A name for the operation (optional).
///
///  Returns:
///    A transposed `Tensor`.
pub fn transpose<S, TeS, Tx>(
    context: &mut Scope,
    a: Tx,
    perm: Option<TeS>,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tx: Into<Tensor>,
    TeS: TensorOps,
    S: AsRef<Path>,
{
    let scope = &mut context.name_scope(name.as_ref(), Some("transpose".as_ref()));
    let a = a.into();
    if let Some(perm) = perm {
        //let perm = scope.constant(perm, &[1] as &[i32], "")?.into();
        let perm = perm.into_tensor(scope, "");
        scope.install(Transpose::new(a, perm, "")?)
    } else {
        let rank = rank(scope, a, "")?;
        let p0 = scope.constant(&[1_i32], &[] as &[i32], "")?;
        let p1 = sub(scope, rank, p0, "")?;
        let p2 = range(scope, 0_i32, rank, 1_i32, "")?;
        let perm = sub(scope, p1, p2, "")?;

        let ret = scope.install(Transpose::new(a, perm, "")?)?;
        let input_shape = a.get_shape(scope);
        Ok(ret)
    }
}

add_new_op!(Transpose,
    constructor: [
        add_new_op!(BIN CONSTRUCTOR: Transpose, Init: []);
    ],
    digest: [DEFAULT_DIGEST: Transpose, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);


///// Where /////

pub fn where_cond<Tc, S>(
    context: &mut Scope,
    cond: Tc,
    x: Option<Tensor>,
    y: Option<Tensor>,
    name: S,
) -> Result<Tensor, ::Error>
where
    Tc: Into<Tensor>,
    S: AsRef<Path>,
{
    let cond = cond.into();
    if cond.dtype != DataType::Bool {
        return Err(::Error::Stub);
    }
    if (x.is_none() && y.is_some()) || (x.is_some() && y.is_none()) {
        return Err(::Error::Stub);
    } else if x.is_some() || y.is_some() {
        unimplemented!()
    } else {
        context.install(Where::new(cond.into(), name)?)
    }
}

add_new_op!(Where,
    constructor: [
        add_new_op!(UNARY CONSTRUCTOR: Where, Init: [output_type: DataType::Int64]);
    ],
    digest: [DEFAULT_DIGEST: Where, DTYPE_ATTR],
    extra_funcs: [], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);


///// Zeros /////

/// Creates a tensor with all elements set to zero.
///
/// This operation returns a tensor of type _dtype_ with shape _shape_ and all
/// elements set to zero.
pub fn zeros<S, TeS>(
    context: &mut Scope,
    shape: TeS,
    dtype: DataType,
    name: S,
) -> Result<Tensor, ::Error>
where
    S: AsRef<Path>,
    TeS: Into<Tensor>,
{

    unimplemented!()
}


///// Lower level support ops /////

pub(crate) fn constant<'a, T, I>(
    graph: &mut Graph,
    name: &str,
    value: TypedTensor<T>,
    control_inputs: I,
) -> Result<OperationData, Status>
where
    T: TensorType,
    I: IntoIterator<Item = &'a OperationData>,
{
    let mut c = graph.new_operation("Const", name)?;
    c.set_attr_tensor("value", value)?;
    c.set_attr_type("dtype", T::data_type())?;
    ::framework::add_control_input(&mut c, control_inputs);
    c.finish()
}

pub(crate) fn identity<'a, I>(
    graph: &mut Graph,
    name: &str,
    input: (OperationData, i32),
    control_inputs: I,
) -> Result<OperationData, Status>
where
    I: IntoIterator<Item = &'a OperationData>,
{
    let mut copy = graph.new_operation("Identity", name)?;
    copy.add_input(
        Output {
            operation: input.0,
            index: input.1,
        },
    );
    super::add_control_input(&mut copy, control_inputs);
    copy.finish()
}

pub(crate) fn placeholder(
    graph: &mut Graph,
    name: &str,
    dtype: DataType,
) -> Result<OperationData, Status> {
    let mut p = graph.new_operation("Placeholder", name)?;
    p.set_attr_type("dtype", dtype)?;
    p.finish()
}

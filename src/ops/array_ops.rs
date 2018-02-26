//! Array Operations.

use super::*;
#[allow(unused_imports)]
use framework::TensorContent;

///// Concat /////

/// Concatenates tensors along one dimension.
///
/// Concatenates the list of tensors values along dimension axis.
/// If `values[i].shape = [D0, D1, ... Daxis(i), ...Dn]`, the concatenated result
/// has shape where `Raxis = sum(Daxis(i))`
///
/// That is, the data from the input tensors is joined along the axis dimension.
///
/// The number of dimensions of the input tensors must match, and all dimensions
/// except `axis` must be equal.
pub fn concat<S, TeS>(
    context: &mut Scope,
    values: Vec<Tensor>,
    axis: TeS,
    name: S,
) -> Result<Tensor>
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
        fn new<S: AsRef<Path>>(values: Vec<Tensor>, axis: Tensor, name: S,) -> Result<Concat<'a>> {
            let output_type = values[0].dtype;
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
    use TensorShape;

    let mut context = Scope::new();
    let t1 = context
        .constant(&[1_i32, 2, 3, 4, 5, 6], [2, 3].as_ref(), "t1")
        .unwrap()
        .into();
    let t2 = context
        .constant(&[7_i32, 8, 9, 10, 11, 12], [2, 3].as_ref(), "t2")
        .unwrap()
        .into();
    let op1 = concat(&mut context, vec![t1, t2], 0, "").unwrap();
    let op2 = concat(&mut context, vec![t1, t2], 1, "").unwrap();
    test_suite!(run_op: [op1, op2]; context, input: {});

    let src_op1 = context.get_src_op(op1);
    let src_op2 = context.get_src_op(op2);
    let g = context.unwrap_graph().unwrap();
    assert_eq!(
        g.tensor_shape(test_suite!(out: src_op1, 0)).unwrap(),
        TensorShape::from(Some(vec![Some(4), Some(3)]))
    );
    assert_eq!(
        g.tensor_shape(test_suite!(out: src_op2, 0)).unwrap(),
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
///
///     * input: A `Tensor`.
///     * axis: 0-D (scalar). Specifies the dimension index at which to
///       expand the shape of `input`.
///     * name: The name of the output `Tensor`.
///     * dim: 0-D (scalar). Equivalent to `axis`, to be deprecated.
///
///   ### Returns:
///     A `Tensor` with the same data as `input`, but its shape has an additional
///     dimension of size 1 added.
///
///   ### Error
///     ValueError: if both `dim` and `axis` are specified.
pub fn expand_dims<Tx, S, TeS>(
    context: &mut Scope,
    tensor: Tx,
    axis: TeS,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
    TeS: ShapeSize,
{
    let tensor = tensor.into_tensor(context);
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

///// Fill /////

/// Creates a tensor filled with a scalar value.
///
/// This operation creates a tensor of shape dims and fills it with value.
///
///
/// * dims: A Tensor of type int32. 1-D. Represents the shape of the output tensor.
/// * value: A Tensor. 0-D (scalar). Value to fill the returned tensor.
/// * name: A name for the operation (optional).
///
/// ### Returns:
/// A Tensor. Has the same type as value.
pub fn fill<Tx, Ty, S>(context: &mut Scope, dims: Tx, value: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let dims = dims.into_tensor(context);
    let val = value.into_tensor(context);
    context.install(Fill::new(dims, val, name)?)
}

add_new_op!(Fill,
    constructor: [add_new_op!(BIN CONSTRUCTOR: Fill, Init: []);],
    digest: [DEFAULT_DIGEST: Fill, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

///// Gather /////

/// Gather slices from params axis axis according to indices.
///
/// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
/// Produces an output tensor with shape
/// `params.shape[:axis] + indices.shape + params.shape[axis + 1:]` where:
///
/// ```Python
/// # Scalar indices (output is rank(params) - 1).
/// output[a_0, ..., a_n, b_0, ..., b_n] =
///     params[a_0, ..., a_n, indices, b_0, ..., b_n]
///
/// # Vector indices (output is rank(params)).
/// output[a_0, ..., a_n, i, b_0, ..., b_n] =
///     params[a_0, ..., a_n, indices[i], b_0, ..., b_n]
///
/// # Higher rank indices (output is rank(params) + rank(indices) - 1).
/// output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
///     params[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
/// ```
///
///
/// * params: The tensor from which to gather values. Must be at least rank axis + 1.
/// * indices: A `Tensor`. Must be one of the following types: int32, int64. Index tensor.
///       Must be in range [0, params.shape[axis]).
/// * axis: A `Tensor`. Must be one of the following types: int32, int64.
///       The axis in params to gather indices from. Defaults to the first dimension.
///       Supports negative indexes.
/// * name: A name for the operation (optional).
///
/// ### Returns:
/// A `Tensor` with the same type as `params`. Values from `params` gathered from indices given
/// by `indices`, with shape `params.shape[:axis] + indices.shape + params.shape[axis + 1:]`.
pub fn gather<Tx, Ty, S>(context: &mut Scope, params: Tx, indices: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let indices = indices.into_tensor(context);
    let params = params.into_tensor(context);
    if indices.dtype != DataType::Int32 && indices.dtype != DataType::Int64 {
        return Err(Error::from(ErrorKind::Stub));
    }
    context.install(Gather::new(params, indices, name)?)
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
    let x = context
        .constant(&[0_i32, 1, 2, 3, 4, 5], [6].as_ref(), "x")
        .unwrap();
    let indices = context
        .constant(&[2_i32, 0, 2, 5], [4].as_ref(), "gather")
        .unwrap();
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
///
///    * input: A `Tensor` or `SparseTensor`.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    * A `Tensor` of type `int32`.
pub fn rank<Tx, S>(context: &mut Scope, input_tensor: Tx, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let scope = &mut context.name_scope(name.as_ref(), Some("Rank".as_ref()));
    let input_tensor = input_tensor.into_tensor(context);
    // optimize: encode the rank as a constant when possible.
    if let Some(ndim) = input_tensor.get_shape(scope).dims() {
        Ok(scope.constant(&[ndim as i32], &[] as &[i32], "")?.into())
    } else {
        context.install(Rank::new(input_tensor, "")?)
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

/// Reshapes a tensor.
///
/// Given `tensor`, this operation returns a tensor that has the same values
/// as `tensor` with shape `shape`.
///
/// If one component of `shape` is the special value -1, the size of that dimension is computed
/// so that the total size remains constant. In particular, a shape of `[-1]` flattens into 1-D.
/// At most one component of `shape` can be -1.
///
/// If `shape` is 1-D or higher, then the operation returns a tensor with shape `shape` filled
/// with the values of `tensor`. In this case, the number of elements implied by `shape` must
/// be the same as the number of elements in `tensor`.
///
///
///  * tensor: A Tensor.
///  * shape: A Tensor. Must be one of the following types: int32, int64.
///    Defines the shape of the output tensor.
///  * name: A name for the operation (optional).
pub fn reshape<Tx, Ty, S>(context: &mut Scope, tensor: Tx, shape: Ty, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    /*
    let shape = {
        let dims: &[i64] = &[shape.len() as i64];
        context.constant("", shape, dims)?
    };
    */
    let shape = shape.into_tensor(context);
    let tensor = tensor.into_tensor(context);
    context.install(Reshape::new(tensor, shape, name)?)
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
    use TensorShape;

    let mut context = Scope::new();
    let x = context
        .constant(&[1_i32, 2, 3, 4, 5, 6, 7, 8, 9], [9].as_ref(), "x")
        .unwrap();
    let y = context
        .constant(&[1_i32, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3].as_ref(), "y")
        .unwrap();

    let shape = context.constant(&[3, 3], [2].as_ref(), "").unwrap();
    let op1 = reshape(&mut context, x, shape, "").unwrap();
    let src_op1 = context.get_src_op(op1);

    let shape = context.constant(&[-1], [1].as_ref(), "").unwrap();
    let op2 = reshape(&mut context, y, shape, "").unwrap();
    let src_op2 = context.get_src_op(op2);

    let g = context.unwrap_graph().unwrap();
    assert_eq!(
        g.tensor_shape(test_suite!(out: src_op1, 0)).unwrap(),
        TensorShape::from(Some(vec![Some(3), Some(3)]))
    );
    assert_eq!(
        g.tensor_shape(test_suite!(out: src_op2, 0)).unwrap(),
        TensorShape::from(Some(vec![Some(9)]))
    );
}

///// Shape /////

/// Returns the shape of a tensor.
///
/// This operation returns a 1-D integer tensor representing the shape of `input`.
pub fn shape<Tx, S>(
    context: &mut Scope,
    tensor: Tx,
    out_type: Option<DataType>,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let out_type = if let Some(val) = out_type {
        vec![val]
    } else {
        vec![]
    };
    let tensor = tensor.into_tensor(context);
    context.install(Shape::new(tensor, &out_type, name)?)
}

add_new_op!(Shape,
    constructor: [
        fn new<S: AsRef<Path>>(tensor: Tensor, output_type: &'a [DataType], name: S) 
            -> Result<Shape<'a>> 
        {
            let out;
            let attributes = if let Some(dtype) = output_type.get(0) {
                match *dtype {
                    DataType::Int64 => out = DataType::Int64,
                    DataType::Int32 => out = DataType::Int32,
                    _ => return Err(Error::from(ErrorKind::Stub)),
                }
                vec![("out_type", false, Attribute::Type(output_type))]
            } else if output_type.len() > 0 {
                return Err(Error::from(ErrorKind::Stub));
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
    let x = context
        .constant(&[1_i32, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3].as_ref(), "x")
        .unwrap();

    let op = shape(&mut context, x, Some(DataType::Int64), "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int64] == [3, 3]});
}

///// Size /////

/// Returns the size of a tensor.
///
/// This operation returns an int32 representing the number of elements in input.
pub fn size<Tx, S>(context: &mut Scope, input: Tx, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let input = input.into_tensor(context);
    context.install(Size::new(input, name)?)
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
    let x = context
        .constant(
            &[1_i32, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            [2, 2, 3].as_ref(),
            "x",
        )
        .unwrap();
    let op = size(&mut context, x, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [12]});
}

///// Squeeze /////

/// Removes dimensions of size 1 from the shape of a tensor.
///
/// Given a tensor `input`, this operation returns a tensor of the same type with
/// all dimensions of size 1 removed. If you don't want to remove all size 1 dimensions,
/// you can remove specific size 1 dimensions by specifying `axis`.
///
///
/// * input: A Tensor. The input to squeeze.
/// * axis: An optional list of ints. If specified, only squeezes the dimensions listed.
///    The dimension index starts at 0. It is an error to squeeze a dimension that is not 1.
/// * name: A name for the operation (optional).
///
/// ### Returns:
/// * A Tensor with the same type as input. Contains the same data as input,
///   but has one or more dimensions of size 1 removed.
pub fn squeeze<Tx, Sh, S>(
    context: &mut Scope,
    input: Tx,
    axis: Option<Sh>,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
    Sh: ShapeOps,
{
    let dims: Vec<i64>;
    let input = input.into_tensor(context);
    let mut squeeze = Squeeze::new(input, name)?;
    if let Some(axis) = axis {
        dims = axis.definition_i64()
            .ok_or(Error::from(ErrorKind::UndefinedTensorShape))?;
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
///
///    * input: A `Tensor`.
///    * begin: An `int32` or `int64` tensor.
///    * size: An `int32` or `int64` tensor.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    A `Tensor` with the same type as `input`.
pub fn slice<Tx, Tb, Ts, S>(
    context: &mut Scope,
    input: Tx,
    begin: Tb,
    size: Ts,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    Tb: TensorOps,
    Ts: TensorOps,
    S: AsRef<Path>,
{
    let begin = begin.into_tensor(context);
    let size = size.into_tensor(context);
    let input = input.into_tensor(context);
    context.install(Slice::new(input, begin, size, name)?)
}

add_new_op!(Slice,
    constructor: [
        fn new<S: AsRef<Path>>(input: Tensor, begin: Tensor, size: Tensor, name: S) 
            -> Result<Slice<'a>> 
        {
            Ok(
                Slice {
                    ident: NodeIdent::new(),
                    elements: vec![input, begin, size],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: Vec::with_capacity(0),
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: Transpose, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

///// Stack /////

///  Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.
///
///  Packs the list of tensors in `values` into a tensor with rank one higher than
///  each tensor in `values`, by packing them along the `axis` dimension.
///  Given a list of length `N` of tensors of shape `(A, B, C)`;
///
///  if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
///  if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
///  Etc.
///
///  For example:
///
///  ```python
///  x = tf.constant([1, 4])
///  y = tf.constant([2, 5])
///  z = tf.constant([3, 6])
///  tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
///  tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
///  ```
///
///  This is the opposite of unstack.  The numpy equivalent is
///
///  ```python
///  tf.stack([x, y, z]) = np.stack([x, y, z])
///  ```
///
///  ### Args:
///    * values: A list of `Tensor` objects with the same shape and type.
///    * axis: An `int`. The axis to stack along. Defaults to the first dimension.
///      Negative values wrap around, so the valid range is `[-(R+1), R+1)`.
///    * name: A name for this operation (optional).
///
///  ### Returns:
///    * output: A stacked `Tensor` with the same type as `values`.
pub fn stack<Tx, TeS, S>(context: &mut Scope, input: Vec<Tx>, axis: TeS, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    TeS: ShapeSize,
    S: AsRef<Path>,
{
    let input: Vec<_> = input.into_iter().map(|x| x.into_tensor(context)).collect();
    let n = input.len() as i64;
    let axis = axis.as_i64();
    context.install(Pack::new(input, name)?.axis(&[axis]).input_len(&[n]))
}

add_new_op!(
    Pack,
    constructor: [
        fn new<S: AsRef<Path>>(values: Vec<Tensor>, name: S,) -> Result<Pack<'a>> {
            let output_type = values[0].dtype;
            Ok(
                Pack {
                    ident: NodeIdent::new(),
                    input_lists: vec![(0, values)],
                    elements: Vec::with_capacity(0),
                    name: generate_name!(is_none: name),
                    attributes: Vec::with_capacity(1),
                    output_type,
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: Pack, DTYPE_ATTR],
    extra_funcs: [
        fn axis(mut self, val: &'a [i64]) -> Self {
            self.attributes.push(("axis", false, Attribute::Int(val)));
            self
        }

        fn input_len(mut self, val: &'a [i64]) -> Self {
            self.attributes.push(("N", false, Attribute::Int(val)));
            self
        }
    ], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_stack() {
    let mut scope = &mut Scope::new();
    let x = [1, 4].as_ref();
    let y = [2, 5].as_ref();
    let z = [3, 6].as_ref();

    let op = stack(scope, vec![x, y, z], 0, "").unwrap();
    let results = test_suite!(run_op: [op]; scope, input: {});
    test_suite!(results; assert: {[0;Int32] == [1, 4, 2, 5, 3, 6]});
}

///// Stop Gradient /////

///  Stops gradient computation.
///
///  When executed in a graph, this op outputs its input tensor as-is.
///
///  When building ops to compute gradients, this op prevents the contribution of
///  its inputs to be taken into account.  Normally, the gradient generator adds ops
///  to a graph to compute the derivatives of a specified 'loss' by recursively
///  finding out inputs that contributed to its computation.  If you insert this op
///  in the graph it inputs are masked from the gradient generator.  They are not
///  taken into account for computing gradients.
///
///  This is useful any time you want to compute a value with TensorFlow but need
///  to pretend that the value was a constant. Some examples include:
///
///  *  The _EM_ algorithm where the _M-step_ should not involve backpropagation
///     through the output of the _E-step_.
///  *  Contrastive divergence training of Boltzmann machines where, when
///     differentiating the energy function, the training must not backpropagate
///     through the graph that generated the samples from the model.
///  *  Adversarial training, where no backprop should happen through the adversarial
///     example generation process.
///
///  ### Args:
///    * input: A `Tensor`.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    A `Tensor`. Has the same type as `input`.
pub fn stop_gradient<Tx, S>(context: &mut Scope, input: Tx, name: S) -> Result<Tensor>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let input = input.into_tensor(context);
    context.install(StopGradient::new(input, name)?)
}

add_new_op!(StopGradient,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: StopGradient, Init: []);],
    digest: [DEFAULT_DIGEST: StopGradient, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

///  Extracts a strided slice of a tensor (generalized python array indexing).
///
///  To a first order, this operation extracts a slice of size `end - begin`
///  from a tensor `input`
///  starting at the location specified by `begin`. The slice continues by adding
///  `stride` to the `begin` index until all dimensions are not less than `end`.
///  Note that components of stride can be negative, which causes a reverse
///  slice.
///
///  This operation can be thought of an encoding of a numpy style sliced
///  range. Given a python slice input[<spec0>, <spec1>, ..., <specn>]
///  this function will be called as follows.
///
///  `begin`, `end`, and `strides` will be all length n. n is in general
///  not the same dimensionality as `input`.
///
///  For the ith spec,
///  `begin_mask`, `end_mask`, `ellipsis_mask`, `new_axis_mask`,
///  and `shrink_axis_mask` will have the ith bit corresponding to
///  the ith spec.
///
///  If the ith bit of `begin_mask` is non-zero, `begin[i]` is ignored and
///  the fullest possible range in that dimension is used instead.
///  `end_mask` works analogously, except with the end range.
///
///  `foo[5:,:,:3]` on a 7x8x9 tensor is equivalent to `foo[5:7,0:8,0:3]`.
///  `foo[::-1]` reverses a tensor with shape 8.
///
///
///  If the ith bit of `ellipsis_mask` is non-zero, as many unspecified dimensions
///  as needed will be inserted between other dimensions. Only one
///  non-zero bit is allowed in `ellipsis_mask`.
///
///  For example `foo[3:5,...,4:5]` on a shape 10x3x3x10 tensor is
///  equivalent to `foo[3:5,:,:,4:5]` and
///  `foo[3:5,...]` is equivalent to `foo[3:5,:,:,:]`.
///
///  If the ith bit of `new_axis_mask` is one, then `begin`,
///  `end`, and `stride` are ignored and a new length 1 dimension is
///  added at this point in the output tensor.
///
///  For example `foo[3:5,4]` on a 10x8 tensor produces a shape 2 tensor
///  whereas `foo[3:5,4:5]` produces a shape 2x1 tensor with shrink_mask
///  being 1<<1 == 2.
///
///  If the ith bit of `shrink_axis_mask` is one, then `begin`,
///  `end[i]`, and `stride[i]` are used to do a slice in the appropriate
///  dimension, but the output tensor will be reduced in dimensionality
///  by one. This is only valid if the ith entry of slice[i]==1.
///
///  NOTE: `begin` and `end` are zero-indexed`.
///  `strides` entries must be non-zero.
///
///
///  ```python
///  # 'input' is [[[1, 1, 1], [2, 2, 2]],
///  #             [[3, 3, 3], [4, 4, 4]],
///  #             [[5, 5, 5], [6, 6, 6]]]
///  tf.strided_slice(input, [1, 0, 0], [2, 1, 3], [1, 1, 1]) ==> [[[3, 3, 3]]]
///  tf.strided_slice(input, [1, 0, 0], [2, 2, 3], [1, 1, 1]) ==> [[[3, 3, 3],
///                                                                 [4, 4, 4]]]
///  tf.strided_slice(input, [1, -1, 0], [2, -3, 3], [1, -1, 1]) ==>[[[4, 4, 4],
///                                                                   [3, 3, 3]]]
///  ```
///
///  ### Args:
///    * input_: A `Tensor`.
///    * begin: An `int32` or `int64` `Tensor`.
///    * end: An `int32` or `int64` `Tensor`.
///    * strides: An `int32` or `int64` `Tensor`.
///    * begin_mask: An `int32` mask.
///    * end_mask: An `int32` mask.
///    * ellipsis_mask: An `int32` mask.
///    * new_axis_mask: An `int32` mask.
///    * shrink_axis_mask: An `int32` mask.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    * A `Tensor` the same type as `input`.
pub fn strided_slice<Ti, Tb, Te, Ts, S>(
    context: &mut Scope,
    input: Ti,
    begin: Tb,
    end: Te,
    strides: Ts,
    begin_mask: Option<i32>,
    end_mask: Option<i32>,
    ellipsis_mask: Option<i32>,
    new_axis_mask: Option<i32>,
    shrink_axis_mask: Option<i32>,
    name: S,
) -> Result<Tensor>
where
    Ti: TensorOps,
    Tb: TensorOps,
    Te: TensorOps,
    Ts: TensorOps,
    S: AsRef<Path>,
{
    let input = input.into_tensor(context);
    let begin = begin.into_tensor(context);
    let end = end.into_tensor(context);
    let strides = strides.into_tensor(context);
    let begin_mask = if let Some(val) = begin_mask { val } else { 0 };
    let end_mask = if let Some(val) = end_mask { val } else { 0 };
    let ellipsis_mask = if let Some(val) = ellipsis_mask {
        val
    } else {
        0
    };
    let new_axis_mask = if let Some(val) = new_axis_mask {
        val
    } else {
        0
    };
    let shrink_axis_mask = if let Some(val) = shrink_axis_mask {
        val
    } else {
        0
    };
    context.install(StridedSlice::new(input, begin, end, strides, name)?)
}

add_new_op!(StridedSlice,
    constructor: [
        pub(crate) fn new<S: AsRef<Path>>(
            input: Tensor, 
            begin: Tensor, 
            end: Tensor, 
            strides: Tensor, 
            name: S
        ) 
            -> Result<StridedSlice<'a>> 
        {
            Ok(
                StridedSlice {
                    ident: NodeIdent::new(),
                    elements: vec![input, begin, end, strides],
                    name: generate_name!(is_none: name),
                    input_lists: Vec::with_capacity(0),
                    attributes: vec![],
                },
            )
        } 
    ],
    digest: [DEFAULT_DIGEST: StridedSlice, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_stride() {
    let mut context = &mut Scope::new();
    let op1 = strided_slice(
        context,
        [0_i32, 1, 2, 3].as_ref(),
        [-1_i32].as_ref(),
        [::std::i32::MAX].as_ref(),
        [1_i32].as_ref(),
        None,
        None,
        None,
        None,
        None,
        "",
    ).unwrap();
    let results = test_suite!(run_op: [op1]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [3]});
}

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
///
///    * a: A `Tensor`.
///    * perm: A permutation of the dimensions of `a`.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    * A transposed `Tensor`.
pub fn transpose<S, TeS, Tx>(
    context: &mut Scope,
    a: Tx,
    perm: Option<TeS>,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    TeS: TensorOps,
    S: AsRef<Path>,
{
    let scope = &mut context.name_scope(name.as_ref(), Some("transpose".as_ref()));
    let a = a.into_tensor(context);
    if let Some(perm) = perm {
        //let perm = scope.constant(perm, &[1] as &[i32], "")?.into();
        let perm = perm.into_tensor(scope);
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
    constructor: [add_new_op!(BIN CONSTRUCTOR: Transpose, Init: []);],
    digest: [DEFAULT_DIGEST: Transpose, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

///// Unique /////

///   Finds unique elements in a 1-D tensor.
///
///  This operation returns a tensor `y` containing all of the unique elements of `x`
///  sorted in the same order that they occur in `x`. This operation also returns a
///  tensor `idx` the same size as `x` that contains the index of each value of `x`
///  in the unique output `y`. In other words:
///
///  `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
///
///  For example:
///
///  ```python
///  # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
///  y, idx = unique(x)
///  y ==> [1, 2, 4, 7, 8]
///  idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
///  ```
pub fn unique<S, Tx>(
    context: &mut Scope,
    x: Tx,
    out_idx: Option<DataType>,
    name: S,
) -> Result<(Tensor, Tensor)>
where
    Tx: TensorOps,
    S: AsRef<Path>,
{
    let out_idx = if let Some(dtype) = out_idx {
        [dtype]
    } else {
        [DataType::Int32]
    };

    let x = x.into_tensor(context);
    context.install(Unique::new(x, out_idx[0], name)?.out_idx(&out_idx))
}

add_new_op!(Unique,
    constructor: [
        fn new<S: AsRef<Path>>(x: Tensor, output_type: DataType, name: S,) -> Result<Unique<'a>> {
            Ok(
                Unique {
                    ident: NodeIdent::new(),
                    input_lists: Vec::with_capacity(0),
                    elements: vec![x],
                    name: generate_name!(is_none: name),
                    attributes: Vec::with_capacity(1),
                    output_type,
                },
            )
        }
    ],
    digest: [DIGEST_BIN_OUT: Unique, INPUT0, DTYPE_ATTR],
    extra_funcs: [
        fn out_idx(mut self, val: &'a [DataType]) -> Self {
            self.attributes.push(("out_idx", false, Attribute::Type(val)));
            self
        }
    ], 
    extra_attr: [output_type: DataType],
    output: [(Tensor, Tensor)],
);

///// Where /////

/// Return the elements, either from x or y, depending on the condition.
///
/// If both x and y are None, then this operation returns the coordinates of true elements of condition.
/// The coordinates are returned in a 2-D tensor where the first dimension (rows) represents
/// the number of true elements, and the second dimension (columns) represents the coordinates
/// of the true elements. Keep in mind, the shape of the output tensor can vary depending on
/// how many true values there are in input. Indices are output in row-major order.
///
/// If both non-None, `x` and `y` must have the same shape. The `condition` tensor must be a scalar
/// if `x` and `y` are scalar. If `x` and `y` are vectors of higher rank, then condition must be either
/// a vector with size matching the first dimension of `x`, or must have the same shape as `x`.
///
/// The `condition` tensor acts as a mask that chooses, based on the value at each element,
/// whether the corresponding element / row in the output should be taken from `x` (if true)
/// or `y` (if false).
///
/// If `condition` is a vector and `x` and `y` are higher rank matrices, then it chooses which row
/// (outer dimension) to copy from `x` and `y`. If `condition` has the same shape as `x` and `y`,
/// then it chooses which element to copy from `x` and `y`.
///
/// ### Args:
/// * `condition`: A `Tensor` of type `bool`.
/// * `x`: A Tensor which may have the same shape as condition. If condition is rank 1,
///   `x` may have higher rank, but its first dimension must match the size of condition.
/// * `y`: A tensor with the same shape and type as `x`.
/// * `name`: A name of the operation.
///
/// ### Returns:
/// * A `Tensor` with the same type and shape as `x`, `y` if they are non-None.
///   A `Tensor` with shape `(num_true, dim_size(condition))`.
pub fn where_cond<Tc, S>(
    context: &mut Scope,
    cond: Tc,
    x: Option<Tensor>,
    y: Option<Tensor>,
    name: S,
) -> Result<Tensor>
where
    Tc: TensorOps,
    S: AsRef<Path>,
{
    let cond = cond.into_tensor(context);
    if cond.dtype != DataType::Bool {
        return Err(Error::from(ErrorKind::Stub));
    }
    if (x.is_none() && y.is_some()) || (x.is_some() && y.is_none()) {
        return Err(Error::from(ErrorKind::Stub));
    } else if x.is_some() || y.is_some() {
        math_ops::select(context, cond, x.unwrap(), y.unwrap(), name)
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

#[cfg(test)]
#[test]
fn test_where_cond() {
    use ops::math_ops::greater;
    let mut context = Scope::new();
    let x = context.constant(&[4_i32, 2, 4], [3].as_ref(), "x").unwrap();
    let y = context.constant(&[2_i32, 4, 2], [3].as_ref(), "y").unwrap();
    let cond = greater(&mut context, x, y, "").unwrap();
    let op = where_cond(&mut context, cond, None, None, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int64] == [0_i64, 2]});
    test_suite!(results; assert_len: {[0;Int64] == 2});
    //println!("{:?}", Vec::from(&*results.pop().unwrap().unwrap_i64()));
}

///// Zeros /////

/// Creates a tensor with all elements set to zero.
///
/// This operation returns a tensor of type _dtype_ with shape _shape_ and all
/// elements set to zero.
pub fn zeros<S, Sh>(context: &mut Scope, shape: Sh, dtype: DataType, name: S) -> Result<Tensor>
where
    S: AsRef<Path>,
    Sh: ShapeOps,
{
    let def = shape
        .to_shape()
        .definition_i64()
        .ok_or(Error::from(ErrorKind::UndefinedTensorShape))?;
    let shape = context.constant(&def, [def.len() as i32].as_ref(), "")?;
    let zero = match dtype {
        DataType::Bool => context.constant(&[false], &[] as &[i32], "")?,
        DataType::Double => context.constant(&[0_f64], &[] as &[i32], "")?,
        DataType::Float => context.constant(&[0_f32], &[] as &[i32], "")?,
        DataType::Int32 => context.constant(&[0_i32], &[] as &[i32], "")?,
        DataType::UInt8 => context.constant(&[0_u8], &[] as &[i32], "")?,
        DataType::Int16 => context.constant(&[0_i16], &[] as &[i32], "")?,
        DataType::Int8 => context.constant(&[0_i8], &[] as &[i32], "")?,
        DataType::Int64 => context.constant(&[0_i64], &[] as &[i32], "")?,
        DataType::String => context.constant(&["".to_string()], &[] as &[i32], "")?,
        DataType::QUInt8 => context.constant(&[::QUInt8::from(0)], &[] as &[i32], "")?,
        DataType::QInt16 => context.constant(&[::QInt16::from(0)], &[] as &[i32], "")?,
        DataType::QUInt16 => context.constant(&[::QUInt16::from(0)], &[] as &[i32], "")?,
        DataType::QInt32 => context.constant(&[::QInt32::from(0)], &[] as &[i32], "")?,
        DataType::BFloat16 => context.constant(&[::BFloat16::from(0.)], &[] as &[i32], "")?,
        DataType::Complex64 => context.constant(&[::Complex32::new(0., 0.)], &[] as &[i32], "")?,
        DataType::Complex128 => context.constant(&[::Complex64::new(0., 0.)], &[] as &[i32], "")?,
        _ => return Err(Error::from(ErrorKind::Stub)),
    };
    context.install(Fill::new(shape.into(), zero.into(), name)?)
}

///// Lower level support ops /////

pub(crate) fn constant<'a, T, I>(
    graph: &mut Graph,
    name: &str,
    value: TensorData<T>,
    control_inputs: I,
) -> Result<OperationData>
where
    T: TensorType,
    I: IntoIterator<Item = &'a OperationData>,
{
    let mut c = graph.new_operation("Const", name)?;
    c.set_attr_tensor("value", value)?;
    c.set_attr_type("dtype", T::data_type())?;
    ::framework::add_control_input(&mut c, control_inputs);
    Ok(c.finish()?)
}

pub(crate) fn identity<'a, I>(
    graph: &mut Graph,
    name: &str,
    input: (OperationData, i32),
    control_inputs: I,
) -> Result<OperationData>
where
    I: IntoIterator<Item = &'a OperationData>,
{
    let mut copy = graph.new_operation("Identity", name)?;
    copy.add_input(Output {
        operation: input.0,
        index: input.1,
    });
    super::add_control_input(&mut copy, control_inputs);
    Ok(copy.finish()?)
}

pub(crate) fn placeholder(graph: &mut Graph, name: &str, dtype: DataType) -> Result<OperationData> {
    let mut p = graph.new_operation("Placeholder", name)?;
    p.set_attr_type("dtype", dtype)?;
    Ok(p.finish()?)
}

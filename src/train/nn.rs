//! Neural network support.

use std::path::{Path, PathBuf};

use super::*;
use ops::{ControlFlow, array_ops, math_ops};

/*
pub fn in_top_k<C, Tx, Ty>(context: &mut C,
                            predictions: Tx,
                            targets: Ty,
                            k: u32)
                            -> Result<Tensor>
    where Tx: Into<Tensor>,
            Ty: Into<Tensor>
{
}

pub fn l2_loss<C, Tx>(context: &mut C, tensor: Tx) -> Result<Tensor>
    where Tx: Into<Tensor>
{
}

pub fn sparse_softmax_cross_entropy_with_logits<C, Tx, Ty>(context: &mut C,
                                                            tensor: Tx,
                                                            logits: Ty)
                                                            -> Result<Tensor>
    where Tx: Into<Tensor>,
            Ty: Into<Tensor>
{
}
*/

///  # Batch normalization.
///
///  As described in http://arxiv.org/abs/1502.03167.
///  Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
///  `scale` `gamma` to it, as well as an `offset` `beta`:
///
///  `((gamma * (x - mu)) \ sigma ) + beta`
///
///  `mean`, `variance`, `offset` and `scale` are all expected to be of one of two
///  shapes:
///
///    * In all generality, they can have the same number of dimensions as the
///      input `x`, with identical sizes as `x` for the dimensions that are not
///      normalized over (the 'depth' dimension(s)), and dimension 1 for the
///      others which are being normalized over.
///      `mean` and `variance` in this case would typically be the outputs of
///      `tf.nn.moments(..., keep_dims=True)` during training, or running averages
///      thereof during inference.
///    * In the common case where the 'depth' dimension is the last dimension in
///      the input tensor `x`, they may be one dimensional tensors of the same
///      size as the 'depth' dimension.
///      This is the case for example for the common `[batch, depth]` layout of
///      fully-connected layers, and `[batch, height, width, depth]` for
///      convolutions.
///      `mean` and `variance` in this case would typically be the outputs of
///      `tf.nn.moments(..., keep_dims=False)` during training, or running averages
///      thereof during inference.
///
///  ### Args:
///    * x: Input `Tensor` of arbitrary dimensionality.
///    * mean: A mean `Tensor`.
///    * variance: A variance `Tensor`.
///    * offset: An offset `Tensor`, often denoted `beta` in equations, or
///      None. If present, will be added to the normalized tensor.
///    * scale: A scale `Tensor`, often denoted `gamma` in equations, or
///      `None`. If present, the scale is applied to the normalized tensor.
///    * variance_epsilon: A small float number to avoid dividing by 0.
///    * name: A name for this operation (optional).
///
///  ### Returns:
///    The normalized, scaled, offset tensor.
pub fn batch_normalization<Tx, Tm, Tv, S>(
    scope: &mut Scope,
    x: Tx,
    mean: Tm,
    variance: Tv,
    offset: Option<Tensor>,
    scale: Option<Tensor>,
    variance_epsilon: f32,
    name: S,
) -> Result<Tensor> 
where
    Tx: Into<Tensor>,
    Tm: Into<Tensor>,
    Tv: Into<Tensor>,
    S: AsRef<Path>,
{
    let scope = &mut scope.name_scope(name.as_ref().to_str().unwrap(), Some("batchnorm"));
    let mut inv = {
        let a = variance_epsilon.into_tensor(scope, "");
        let sum = math_ops::add(scope, variance, a, "")?;
        math_ops::rsqrt(scope, sum, "")?
    };
    if let Some(scale) = scale {
        inv = math_ops::multiply(scope, scale, inv, "")?;
    }
    if let Some(offset) = offset {
        let b = {
            let b = math_ops::multiply(scope, mean, inv, "")?;
            math_ops::sub(scope, offset, b, "")?
        };
        let a = math_ops::multiply(scope, x, inv, "")?;
        math_ops::add(scope, a, b, "")
    } else {
        let m = math_ops::negative(scope, mean, "")?;
        let b = math_ops::multiply(scope, m, inv, "")?;
        let a = math_ops::multiply(scope, x, inv, "")?;
        math_ops::add(scope, a, b, "")
    }
}


///// BiasAdd /////

///  Adds `bias` to `value`.
///
///  This is (mostly) a special case of `tf.add` where `bias` is restricted to 1-D.
///  Broadcasting is supported, so `value` may have any number of dimensions.
///  Unlike `tf.add`, the type of `bias` is allowed to differ from `value` in the
///  case where both types are quantized.
///
///  Args:
///    value: A `Tensor` with type `float`, `double`, `int64`, `int32`, `uint8`,
///      `int16`, `int8`, `complex64`, or `complex128`.
///    bias: A 1-D `Tensor` with size matching the last dimension of `value`.
///      Must be the same type as `value` unless `value` is a quantized type,
///      in which case a different quantized type may be used.
///    data_format: A string. 'NHWC' and 'NCHW' are supported.
///    name: A name for the operation (optional).
///
///  Returns:
///    A `Tensor` with the same type as `value`.
pub fn bias_add<Tx, B, S>(
    context: &mut Scope,
    value: Tx,
    bias: B,
    data_format: Option<&str>,
    name: S,
) -> Result<Tensor>
where
    Tx: Into<Tensor>,
    B: Into<Tensor>,
    S: AsRef<Path>,
{
    context.name_scope(name.as_ref(), Some("BiasAdd".as_ref()));
    let value = value.into();
    let bias = bias.into();
    let d_id: &mut [&str] = &mut [""];
    let mut bias_add = BiasAdd::new(value, bias, name)?;
    if let Some(data_format) = data_format {
        d_id[0] = validate_convnet_data_dormat(data_format)?;
        bias_add = bias_add.data_format(&d_id);
    }
    context.install(bias_add)
}

add_new_op!(BiasAdd,
    constructor: [add_new_op!(BIN CONSTRUCTOR: BiasAdd, Init: []);],
    digest: [DEFAULT_DIGEST: BiasAdd, INPUT0],
    extra_funcs: [
        fn data_format(mut self, val: &'a [&'a str]) -> Self {
            self.attributes.push(("data_format", false, Attribute::String(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);


///// LogSoftmax /////

///  Computes log softmax activations.
///
///  For each batch `i` and class `j` we have
///      logsoftmax = logits - log(reduce_sum(exp(logits), dim))
///
///  Args:
///    logits: A non-empty `Tensor`. Must be one of the following types: `half`,
///      `float32`, `float64`.
///    dim: The dimension softmax would be performed on. The default is -1 which
///      indicates the last dimension.
///    name: A name for the operation (optional).
///
///  Returns:
///    A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
///    Error if `logits` is empty or `dim` is beyond the last dimension of `logits`.
pub fn log_softmax<L, S, TeS>(context: &mut Scope, logits: L, dim: TeS, name: S) -> Result<Tensor>
where
    L: Into<Tensor>,
    S: AsRef<Path>,
    TeS: ShapeSize,
{
    softmax_helper(context, logits.into(), true, dim.as_i32(), name.as_ref())
}

add_new_op!(LogSoftmax, 
    constructor: [
        add_new_op!(UNARY CONSTRUCTOR: LogSoftmax, Init: []);
    ],
    digest: [DEFAULT_DIGEST: LogSoftmax, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);


///// Relu /////

///  Computes rectified linear: `max(features, 0)`.
///
///  Args:
///    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
///    name: A name for the operation (optional).
///
///  Returns:
///    A `Tensor`. Has the same type as `features`.
pub fn relu<F, S>(scope: &mut Scope, features: F, name: S) -> Result<Tensor>
where
    F: Into<Tensor>,
    S: AsRef<Path>,
{
    scope.install(Relu::new(features.into(), name)?)
}

add_new_op!(Relu, 
    constructor: [
        add_new_op!(UNARY CONSTRUCTOR: Relu, Init: []);
    ],
    digest: [DEFAULT_DIGEST: Relu, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);


///// Softmax /////

///  Computes softmax activations.
///
///  For each batch `i` and class `j` we have
///      softmax = exp(logits) / reduce_sum(exp(logits), dim)
///
///  Args:
///    logits: A non-empty `Tensor`. Must be one of the following types: `half`,
///      `float32`, `float64`.
///    dim: The dimension softmax would be performed on. The default is -1 which
///      indicates the last dimension.
///    name: A name for the operation (optional).
///
///  Returns:
///    A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
///    Error: if `logits` is empty or `dim` is beyond the last dimension of `logits`.
pub fn softmax<L, S, TeS>(context: &mut Scope, logits: L, dim: TeS, name: S) -> Result<Tensor>
where
    L: Into<Tensor>,
    S: AsRef<Path>,
    TeS: ShapeSize,
{
    softmax_helper(context, logits.into(), false, dim.as_i32(), name.as_ref())
}

add_new_op!(Softmax, 
    constructor: [
        add_new_op!(UNARY CONSTRUCTOR: Softmax, Init: []);
    ],
    digest: [DEFAULT_DIGEST: Softmax, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);


///  Helper function for softmax and log_softmax.
///
///  It reshapes and transposes the input logits into a 2-D Tensor and then invokes
///  the tf.nn._softmax or tf.nn._log_softmax function. The output would be
///  transposed and reshaped back.
///
///  Args:
///    logits: A non-empty `Tensor`. Must be one of the following types: `half`,
///      `float32`, `float64`.
///    compute_op: Either gen_nn_ops._softmax or gen_nn_ops._log_softmax
///    dim: The dimension softmax would be performed on. The default is -1 which
///      indicates the last dimension.
///    name: A name for the operation (optional).
///
///  Returns:
///    A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
///    Error if `logits` is empty or `dim` is beyond the last
///      dimension of `logits`.
fn softmax_helper(
    context: &mut Scope,
    mut logits: Tensor,
    is_log_softmax: bool,
    dim: i32,
    name: &Path,
) -> Result<Tensor> {

    fn swap_axis(
        scope: &mut Scope,
        logits: Tensor,
        dim_index: i32,
        last_index: Tensor,
        name: &Path,
    ) -> Result<Tensor> {
        let r0 = ops::range(scope, 0_i32, dim_index, 1_i32, name)?;
        let r1 = ops::range(scope, 0_i32, dim_index + 1, 1_i32, name)?;
        let r2 = Constant::new(scope, &[dim_index], &[] as &[i32]).into();
        let c = array_ops::concat(scope, vec![r0, last_index, r1, r2], 0, "")?;
        array_ops::transpose(scope, logits, Some(c), name)
    }

    // We need its original shape for shape inference.
    let shape = logits.get_shape(context);
    let ndims = if let Some(n) = shape.dims() {
        n as i32
    } else {
        return Err(Error::from(
            "shape of logits tensor must be defined for softmax operation.",
        ));
    };
    let is_last_dim = dim == -1 || dim == ndims - 1;
    if (ndims == 2) && is_last_dim {
        if is_log_softmax {
            return context.install(LogSoftmax::new(logits, name)?);
        } else {
            return context.install(Softmax::new(logits, name)?);
        }
    }

    // If dim is the last dimension, simply reshape the logits to a matrix and
    // apply the internal softmax.

    // Swap logits' dimension of dim and its last dimension.
    let input_rank = array_ops::rank(context, logits, "")?;

    let s = {
        let n = context.constant(&[1], &[] as &[i32], "")?;
        ops::math_ops::sub(context, input_rank, n, "")?
    };
    logits = swap_axis(context, logits, dim, s, "".as_ref())?;
    let shape_after_swap = array_ops::shape(context, logits, None, "")?;

    // Reshape logits into a matrix.
    logits = flatten_outer_dims(context, logits)?;

    // Do the actual softmax on its last dimension.
    let mut output = if is_log_softmax {
        context.install(LogSoftmax::new(logits, name)?)?
    } else {
        context.install(Softmax::new(logits, name)?)?
    };

    // Transform back the output tensor.
    output = array_ops::reshape(context, output, shape_after_swap, "")?;
    output = swap_axis(context, output, dim, s, name)?;

    // Make shape inference work since reshape and transpose may erase its static shape.
    output = output.set_shape(context, shape)?;
    Ok(output)
}

/// Flattens logits' outer dimensions and keep its last dimension.
fn flatten_outer_dims(scope: &mut Scope, logits: Tensor) -> Result<Tensor> {
    let r = array_ops::rank(scope, logits, "")?;
    let last_dim_size = {
        let s0 = array_ops::shape(scope, logits, None, "")?;
        let s1 = math_ops::sub(scope, r, 1_i32, "")?;
        array_ops::slice(scope, s0, s1, 1_i32, "")?
    };
    let mut output = {
        let c0 = Constant::new(scope, &[1_i32], &[] as &[i32]).into();
        let c = array_ops::concat(scope, vec![c0, last_dim_size], 0, "")?;
        array_ops::reshape(scope, logits, c, "")?
    };

    // Set output shaoe if known.
    let shape: Option<Vec<Option<i64>>> = logits.get_shape(scope).into();
    if let Some(shape) = shape {
        //let shape.
        let mut product = 1;
        let mut product_valid = true;
        for d in &shape[..shape.len()] {
            if let Some(d) = *d {
                product *= d;
            } else {
                product_valid = false;
            }
        }
        if product_valid {
            let output_shape = [product, shape.last().unwrap().unwrap()];
            output = array_ops::reshape(scope, output, &output_shape as &[i64], "")?;
        }
    }
    Ok(output)
}

///  Calculate the mean and variance of `x`.
///
///  The mean and variance are calculated by aggregating the contents of `x`
///  across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
///  and variance of a vector.
///
///  __Note:__ for numerical stability, when shift=None, the true mean
///  would be computed and used as shift.
///
///  When using these moments for batch normalization (see
///  `tf.nn.batch_normalization`):
///   * for so-called "global normalization", used with convolutional filters with
///     shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
///   * for simple batch normalization pass `axes=[0]` (batch only).
///
///  ### Args:
///    *x: A `Tensor`.
///    *axes: Array of ints.  Axes along which to compute mean and
///      variance.
///    *shift: A `Tensor` containing the value by which to shift the data for
///      numerical stability, or `None` in which case the true mean of the data is
///      used as shift. A shift close to the true mean provides the most
///      numerically stable results.
///    *name: Name used to scope the operations that compute the moments.
///    *keep_dims: produce moments with the same dimensionality as the input.
///
///  ### Returns:
///    Two `Tensor` objects: `mean` and `variance`.
pub fn moments<S, Tx, TeS>(
    scope: &mut Scope,
    x: Tx,
    axes: &[TeS],
    shift: Option<Tensor>,
    keep_dims: bool,
    name: S,
) -> Result<(Tensor, Tensor)>
where
    S: AsRef<Path>,
    Tx: Into<Tensor>,
    TeS: ShapeSize,
{
    let scope = &mut scope.name_scope(name.as_ref().to_str().unwrap(), Some("moments"));
    let x = x.into();
    // The dynamic range of fp16 is too limited to support the collection of
    // sufficient statistics. As a workaround we simply perform the operations
    // on 32-bit floats before converting the mean and variance back to fp16
    let y = if let DataType::BFloat16 = x.dtype {
        math_ops::cast(scope, x, DataType::Float, "")?
    } else {
        x
    };
    let shift = if let Some(s) = shift {
        math_ops::cast(scope, x, y.dtype, "")?
    } else {
        // Compute true mean while keeping the dims for proper broadcasting.
        let rm = math_ops::reduce_mean(scope, y, axes, true, "")?;
        array_ops::stop_gradient(scope, rm, "")?
    };
    let shifted_mean = {
        let s = math_ops::sub(scope, y, shift, "")?;
        math_ops::reduce_mean(scope, s, axes, true, "shifted_mean")?
    };

    let mut variance = {
        let a = math_ops::squared_difference(scope, y, shift, "")?;
        let rm = math_ops::reduce_mean(scope, a, axes, true, "")?;
        let sm = math_ops::square(scope, shifted_mean, "")?;
        math_ops::sub(scope, rm, sm, "variance")?
    };
    let mut mean = math_ops::add(scope, shifted_mean, shift, "mean")?;
    if !keep_dims {
        mean = array_ops::squeeze(scope, mean, Some(axes), "")?;
        variance = array_ops::squeeze(scope, mean, Some(axes), "")?;
    }
    if x.dtype == DataType::BFloat16 {
        Ok((
            math_ops::cast(scope, mean, DataType::BFloat16, "")?,
            math_ops::cast(scope, mean, DataType::BFloat16, "")?,
        ))
    } else {
        Ok((mean, variance))
    }
}

use std::path::{Path, PathBuf};

use super::*;
use ops::ControlFlow;

/*
pub fn in_top_k<C, Tx, Ty>(context: &mut C,
                            predictions: Tx,
                            targets: Ty,
                            k: u32)
                            -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>,
            Ty: Into<Tensor>
{
    unimplemented!()
}

pub fn log_softmax<C, Tx>(context: &mut C, tensor: Tx) -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>
{
    unimplemented!()
}

pub fn l2_loss<C, Tx>(context: &mut C, tensor: Tx) -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>
{
    unimplemented!()
}

pub fn sparse_softmax_cross_entropy_with_logits<C, Tx, Ty>(context: &mut C,
                                                            tensor: Tx,
                                                            logits: Ty)
                                                            -> Result<Tensor, ::Error>
    where Tx: Into<Tensor>,
            Ty: Into<Tensor>
{
    unimplemented!()
}
*/

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
) -> Result<Tensor, ::Error>
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


///  Computes rectified linear: `max(features, 0)`.
///
///  Args:
///    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
///    name: A name for the operation (optional).
///
///  Returns:
///    A `Tensor`. Has the same type as `features`.
pub fn relu<F, S>(scope: &mut Scope, features: F, name: S) -> Result<Tensor, ::Error>
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

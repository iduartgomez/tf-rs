use super::*;

///  Clips tensor values to a maximum L2-norm.
///
///  Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
///  normalizes `t` so that its L2-norm is less than or equal to `clip_norm`,
///  along the dimensions given in `axes`. Specifically, in the default case
///  where all dimensions are used for calculation, if the L2-norm of `t` is
///  already less than or equal to `clip_norm`, then `t` is not modified. If
///  the L2-norm is greater than `clip_norm`, then this operation returns a
///  tensor of the same type and shape as `t` with its values set to:
///
///  `t * clip_norm / l2norm(t)`
///
///  In this case, the L2-norm of the output tensor is `clip_norm`.
///
///  As another example, if `t` is a matrix and `axes == [1]`, then each row
///  of the output will have L2-norm equal to `clip_norm`. If `axes == [0]`
///  instead, each column of the output will be clipped.
///
///  This operation is typically used to clip gradients before applying them with
///  an optimizer.
///
///  ### Args:
///    *t: A `Tensor`.
///    *clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
///    *axes: A 1-D (vector) `Tensor` of type int32 containing the dimensions
///      to use for computing the L2-norm. If `None` (the default), uses all
///      dimensions.
///    *name: A name for the operation (optional).
///
///  ### Returns:
///    A clipped `Tensor`.
pub fn clip_by_norm<Tz, Tx, Ty, S>(
    scope: &mut Scope,
    t: Tx,
    clip_norm: Ty,
    axes: Option<Tz>,
    name: S,
) -> Result<Tensor>
where
    Ty: TensorOps,
    Tx: TensorOps,
    Tz: TensorOps,
    S: AsRef<Path>,
{
    let scope = &mut scope.name_scope(name.as_ref().to_str().unwrap(), Some("clip_by_norm"));
    let t = t.into_tensor(scope);
    let clip_norm = clip_norm.into_tensor(scope);
    let axes = if let Some(axes) = axes {
        axes.into_tensor(scope)
    } else {
        (&[] as &[i32]).into_tensor(scope)
    };

    // Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    let l2norm_inv = {
        let a = math_ops::multiply(scope, t, t, "")?;
        let r = math_ops::reduce_sum(scope, a, axes, true, "")?;
        math_ops::rsqrt(scope, r, "")?
    };
    let intermediate = math_ops::multiply(scope, t, clip_norm, "")?;

    // Assert that the shape is compatible with the initial shape,
    // to prevent unintentional broadcasting.
    if !t.get_shape(scope).is_compatible_with(&intermediate.get_shape(scope)) {
        return Err(Error::from(ErrorKind::Stub))
    }
    {
        let c = dtype_to_const!(scope; t.dtype; &[1.0]; &[] as &[i32]; "")?;
        let d = math_ops::divide(scope, c, clip_norm, "")?;
        let min = math_ops::minimum(scope, l2norm_inv, d, "")?;
        let m = math_ops::multiply(scope, intermediate, min, "")?;
        scope.identity(m, name)
    }
}

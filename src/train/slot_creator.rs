//! Standard functions for creating slots.
//!
//! A slot is a `Variable` created with the same shape as a primary variable or
//! `Tensor`. A slot is always scoped in the namespace of the primary object and
//! typically has the same device and type.
//!
//! Slots are typically used as accumulators to track values associated with
//! the primary object:
//!
//! ```python
//! # Optimizers can create a slot for each variable to track accumulators
//! accumulators = {var : create_zeros_slot(var, "momentum") for var in vs}
//! for var in vs:
//!   apply_momentum(var, accumulators[var], lr, grad, momentum_tensor)
//!
//! # Slots can also be used for moving averages
//! mavg = create_slot(var, var.initialized_value(), "exponential_moving_avg")
//! update_mavg = mavg.assign_sub((mavg - var) * (1 - decay))
//! ```

use super::*;
use super::ops::array_ops;
use super::ops::init_ops;

///   Create a slot initialized to the given value.
///
///   The type of the slot is determined by the given value.
///
///   ### Args:
///     * primary: The primary `Variable` or `Tensor`.
///     * val: A `Tensor` specifying the initial value of the slot.
///     * name: Name to use for the slot variable.
///     * colocate_with_primary: Boolean.  If True the slot is located
///       on the same device as `primary`.
///
///   ### Returns:
///     A `Variable` object.
pub fn create_slot<S>(
    context: &mut Scope,
    primary: Tensor,
    val: Tensor,
    name: S,
    _colocate_with_primary: bool,
) -> Result<Variable>
where
    S: AsRef<str>,
{
    // Scope the slot name in the namespace of the primary variable.
    // Set "primary.op.name + '/' + name" as default name, so the scope name of
    // optimizer can be shared when reuse is True. Meanwhile when reuse is False
    // and the same name has been previously used, the scope name will add '_N'
    // as suffix for unique identifications.
    let validate_shape = val.get_shape(context).is_fully_defined();
    let primary_op_name = format!("{}/{}", primary.get_name(context), name.as_ref());
    let scope = &mut context.variable_scope("", Some(primary_op_name.as_str()), None)?;
    // TODO: if colocate_with_primary:
    //           with ops.colocate_with(primary):
    scope.get_variable_with_initializer(val, validate_shape, "")
}

pub fn create_zeros_slot<S, Tx>(
    scope: &mut Scope,
    primary: Tx,
    name: S,
    colocate_with_primary: bool,
) -> Result<Variable>
where
    S: AsRef<str>,
    Tx: TensorOps,
{
    let primary = primary.into_tensor(scope);
    let scope = &mut scope.name_scope("zeros", None);
    let slot_shape = array_ops::shape(scope, primary, None, "").unwrap();
    let slot_shape = slot_shape.get_shape(scope);
    if slot_shape.is_fully_defined() {
        let initializer = init_ops::zeros_initializer(scope, slot_shape, primary.dtype)?;
        create_slot(
            scope,
            primary,
            initializer.into(),
            name,
            colocate_with_primary,
        )
    } else {
        let val = array_ops::zeros(scope, slot_shape, primary.dtype, "")?;
        create_slot(scope, primary, val, name, colocate_with_primary)
    }
}

///  Creates a slot initialized using an `Initializer`.
///
///  The type of the slot is determined by the given value.
///
///  ### Args:
///    * primary: The primary `Variable` or `Tensor`.
///    * initializer: An `Initializer`.  The initial value of the slot.
///    * shape: Shape of the initial value of the slot.
///    * dtype: Type of the value of the slot.
///    * name: Name to use for the slot variable.
///    * colocate_with_primary: Boolean.  If True the slot is located
///      on the same device as `primary`.
///
///  ### Returns:
///    A `Variable` object.
pub fn create_slot_with_initializer<Tp, Op, Sh, S>(
    context: &mut Scope,
    primary: Tp,
    initializer: Op,
    shape: Sh,
    dtype: DataType,
    name: S,
    colocate_with_primary: bool,
) -> Result<Variable>
where
    Tp: TensorOps,
    Op: GetOp,
    Sh: ShapeOps,
    S: AsRef<str>,
{
    // Scope the slot name in the namespace of the primary variable.
    // Set "primary.op.name + '/' + name" as default name, so the scope name of
    // optimizer can be shared when reuse is True. Meanwhile when reuse is False
    // and the same name has been previously used, the scope name will add '_N'
    // as suffix for unique identifications.
    let validate_shape = shape.is_fully_defined();
    let primary = primary.into_tensor(context);
    let primary_op_name = format!("{}/{}", primary.get_name(context), name.as_ref());
    let scope = &mut context.variable_scope("", Some(primary_op_name.as_str()), None)?;
    // TODO: if colocate_with_primary:
    //           with ops.colocate_with(primary):
    context.get_variable_with_initializer(initializer, validate_shape, "")
}

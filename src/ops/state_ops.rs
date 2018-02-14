//! State operations.

use super::*;

// FIXME: op+assign on variables has no deterministic behaviour, marked as private for now

///// Assign /////

/// Update 'ref' by assigning 'value' to it.
///
/// This operation outputs a Tensor that holds the new value of 'ref' after the value has been assigned.
/// This makes it easier to chain operations that need to use the reset value.
pub fn assign<Tx, Ty, S>(
    context: &mut Scope,
    ref_tensor: Tx,
    value: Ty,
    use_locking: bool,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let ref_tensor = ref_tensor.into_tensor(context);
    let value = value.into_tensor(context);
    context.install(Assign::new(ref_tensor, value, name)?.use_locking(&[use_locking]))
}

add_new_op!(Assign, 
    constructor: [
        add_new_op!(BIN CONSTRUCTOR: Assign, Init: []);
    ],
    digest: [DEFAULT_DIGEST: Assign, INPUT0],
    extra_funcs: [
        /// Default is false, must be an slice of len == 1.
        fn use_locking(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("use_locking", false, Attribute::Bool(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_assign() {
    let mut context = Scope::new();
    let x = context
        .get_variable(Some(DataType::Int32), Some(&[] as &[i32]), "x")
        .unwrap();
    let y = context.constant(&[2_i32], &[] as &[i32], "y").unwrap();
    let op = assign(&mut context, x, y, true, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [2_i32]});
}

///// AssignAdd /////

///  Update 'ref_tensor' by adding 'value' to it.
///
///  This operation outputs "ref_tensor" after the update is done.
///  This makes it easier to chain operations that need to use the reset value.
///
///  ### Args:
///    * ref_tensor: A mutable `Tensor`. Must be one of the following types:
///      `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`,
///      `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
///      Should be from a `Variable` node.
///    * value: A `Tensor`. Must have the same type as `ref`.
///      The value to be added to the variable.
///    * use_locking: An optional `bool`. Defaults to `False`.
///      If True, the addition will be protected by a lock;
///      otherwise the behavior is undefined, but may exhibit less contention.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    Same as "ref".  Returned as a convenience for operations that want
///    to use the new value after the variable has been updated.
pub fn assign_add<Tx, Ty, S>(
    scope: &mut Scope,
    ref_tensor: Tx,
    value: Ty,
    use_locking: bool,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let ref_tensor = ref_tensor.into_tensor(scope);
    let value = value.into_tensor(scope);
    scope.install(AssignAdd::new(ref_tensor, value, name)?.use_locking(&[use_locking]))
}

add_new_op!(AssignAdd, 
    constructor: [
        add_new_op!(BIN CONSTRUCTOR: AssignAdd, Init: []);
    ],
    digest: [DEFAULT_DIGEST: AssignAdd, INPUT0],
    extra_funcs: [
        /// Default is false, must be an slice of len == 1.
        fn use_locking(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("use_locking", false, Attribute::Bool(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_assign_add() {
    let mut context = Scope::new();
    let x = context
        .get_variable(Some(DataType::Int32), Some(&[1] as &[i32]), "x")
        .unwrap();
    let y = context.constant(&[3_i32], &[1] as &[i32], "y").unwrap();
    let op = assign_add(&mut context, x, y, false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [3_i32]});
}

///// AssignSub /////

pub fn assign_sub<Tx, Ty, S>(
    context: &mut Scope,
    ref_tensor: Tx,
    value: Ty,
    use_locking: bool,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let ref_tensor = ref_tensor.into_tensor(context);
    let value = value.into_tensor(context);
    context.install(AssignSub::new(ref_tensor, value, name)?.use_locking(&[use_locking]))
}

add_new_op!(AssignSub, 
    constructor: [
        add_new_op!(BIN CONSTRUCTOR: AssignSub, Init: []);
    ],
    digest: [DEFAULT_DIGEST: AssignSub, INPUT0],
    extra_funcs: [
        /// Default is false, must be an slice of len == 1.
        fn use_locking(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("use_locking", false, Attribute::Bool(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_assign_sub() {
    let mut context = Scope::new();
    let x = context
        .get_variable(Some(DataType::Int32), Some(&[1] as &[i32]), "x")
        .unwrap();
    let y = context.constant(&[3_i32], &[1] as &[i32], "y").unwrap();
    let op = assign_sub(&mut context, x, y, false, "").unwrap();
    let results = test_suite!(run_op: [op]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [-3_i32]});
}

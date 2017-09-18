use super::*;

pub fn constant_initializer<S, T>(
    context: &mut Scope,
    name: S,
    value: &[T],
    shape: &[u64],
) -> Result<Constant, ::Error>
    where S: AsRef<Path>,
          T: TensorType
{
    context.constant(name, value, shape)
}

#[test]
#[cfg(test)]
fn test_constant_initializer_explicit() {
    let mut context = Scope::new();

    let init = constant_initializer(&mut context, "", &[3_i32, 3], &[2]).unwrap();
    let var = context.get_variable_with_initializer("", init, None).unwrap();

    let results = test_suite!(run_op: [var]; context, input: {});
    test_suite!(results; assert: {[0;Int32] == [3_i32, 3]});
}


///// Lower level support ops /////

/// The output is a handle to the underlying mutable tensor.
///
/// Needs to be initialized before it can be used.
pub(crate) fn variable_<'a, I>(
    graph: &mut Graph,
    name: &str,
    dtype: DataType,
    shape: &Shape,
    control_inputs: I,
) -> Result<OperationData, Status>
    where I: IntoIterator<Item = &'a OperationData>
{
    let mut var = graph.new_operation("VariableV2", name)?;
    var.set_attr_type("dtype", dtype)?;
    var.set_attr_shape("shape", shape)?;
    super::add_control_input(&mut var, control_inputs);
    var.finish()
}

/// The inputs are a handle to the underlying tensor and the data source operation.
pub(crate) fn assign_(
    graph: &mut Graph,
    name: &str,
    reference: OperationData,
    data: (OperationData, i32),
    validate_shape: bool,
) -> Result<OperationData, Status> {
    let mut var = graph.new_operation("Assign", name)?;
    var.set_attr_bool("validate_shape", validate_shape)?;
    var.add_input(
        Output {
            operation: reference,
            index: 0,
        },
    );
    var.add_input(
        Output {
            operation: data.0,
            index: data.1,
        },
    );
    var.finish()
}

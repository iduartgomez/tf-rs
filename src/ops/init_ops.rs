use super::*;
use tf::Shape;

pub fn constant_initializer<TeS, T>(
    context: &mut Scope,
    value: T,
    shape: &[TeS],
) -> Result<Constant, ::Error>
where
    T: TensorType,
    TeS: ShapeSize,
{
    let total = shape.iter().fold(1_i64, |acc, &x| acc * x.as_i64()) as usize;
    let values = vec![value; total];
    context.constant("", &values, shape)
}

#[test]
#[cfg(test)]
fn test_constant_initializer_explicit() {
    let mut context = Scope::new();

    let init = constant_initializer(&mut context, 3_i32, &[2, 2, 2]).unwrap();
    let var = context.get_variable_with_initializer("", init, true).unwrap();

    let results = test_suite!(run_op: [var]; context, input: {});
    test_suite!(results; assert_len: {[0;Int32] == 8});
    test_suite!(results; assert: {[0;Int32] == [3_i32, 3, 3, 3, 3, 3, 3, 3]});
}

pub fn random_normal_initializer<TeS, F>(
    context: &mut Scope,
    mean: F,
    stddev: F,
    seed: Option<i32>,
    shape: &[TeS],
) -> Result<Tensor, ::Error>
where
    F: Float,
    TeS: ShapeSize,
{
    let scope = &mut context.name_scope("random_normal", None);

    let shape_tensor = scope.constant("", shape, &[shape.len() as i64])?;
    let mean_tensor = scope.constant("mean", &[mean], &[] as &[i64])?;
    let stddev_tensor = scope.constant("name", &[stddev], &[] as &[i64])?;

    let rnd = {
        let (seed, seed2) = scope.get_seed(seed);
        let seed = &[seed as i64];
        let seed2 = &[seed2 as i64];
        let dtype = &[F::data_type()];
        let op = RandomStandardNormal::new(shape_tensor.into(), "")?
            .set_seed(seed)
            .set_seed2(seed2)
            .set_dtype(dtype);
        scope.install(op)?
    };

    let mul = multiply(scope, rnd, stddev_tensor, "")?;
    add(scope, mul, mean_tensor, "")
}

/// Outputs random values from a normal distribution.
///
/// The generated values will have mean 0 and standard deviation 1.
///
/// The generated values will have mean 0 and standard deviation 1.
///
/// shape: The shape of the output tensor.
/// dtype: The type of the output.
/// seed: If either `seed` or `seed2` are set to be non-zero, the random number
///   generator is seeded by the given seed.  Otherwise, it is seeded by a
///   random seed.
/// seed2: A second seed to avoid seed collision.
///
/// output: A tensor of the specified shape filled with random normal values.
add_new_op!(RandomStandardNormal,
    constructor: [add_new_op!(
        UNARY CONSTRUCTOR: RandomStandardNormal, Init: [output_type: DataType::Float]
    );],
    digest: [DEFAULT_DIGEST: RandomStandardNormal, DTYPE_ATTR],
    extra_funcs: [
        /// Default is 0.
        fn set_seed(mut self, val: &'a [i64]) -> Self {
            self.attributes.push(("seed", false, Attribute::Int(val)));
            self
        }

        /// Default is 0.
        fn set_seed2(mut self, val: &'a [i64]) -> Self {
            self.attributes.push(("seed2", false, Attribute::Int(val)));
            self
        }

        /// Output tensor dtype.
        fn set_dtype(mut self, val: &'a [DataType]) -> Self {
            self.output_type = val[0];
            self.attributes.push(("dtype", false, Attribute::Type(val)));
            self
        }
    ], 
    extra_attr: [
        output_type: DataType
    ],
    output: [Tensor],
);

#[test]
#[cfg(test)]
fn test_random_normal_initializer() {
    let mut context = Scope::new();

    let init = random_normal_initializer(&mut context, 0.0_f32, 1.0, None, &[2, 2]).unwrap();
    let var = context.get_variable_with_initializer("", init, true).unwrap();

    let results = test_suite!(run_op: [var]; context, input: {});
    test_suite!(results; assert_len: {[0;Float] == 4});
}

/// Initializer that generates tensors initialized to 0.
pub fn zeros_initializer<TeS>(context: &mut Scope, shape: &[TeS]) -> Result<Constant, ::Error>
where
    TeS: ShapeSize,
{
    unimplemented!()
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
where
    I: IntoIterator<Item = &'a OperationData>,
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
    var.add_input(Output {
        operation: reference,
        index: 0,
    });
    var.add_input(Output {
        operation: data.0,
        index: data.1,
    });
    var.finish()
}

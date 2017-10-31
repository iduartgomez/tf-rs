use super::*;

use tf::Shape;

/// Initializer that generates tensors with constant values.
/// 
/// The resulting tensor is populated with values of type dtype, as specified by arguments value 
/// following the desired shape of the new tensor (see examples below).
/// 
/// ### Args:
/// * value: All elements of the initialized variable will be set to the corresponding value 
///          in the value argument.
/// * shape: Shape of the initializer.
pub fn constant_initializer<TeS, T>(
    context: &mut Scope,
    value: T,
    shape: &[TeS],
) -> Result<Constant>
where
    T: TensorType,
    TeS: ShapeSize,
{
    let total = shape.iter().fold(1_i64, |acc, &x| acc * x.as_i64()) as usize;
    let values = vec![value; total];
    context.constant(&values, shape, "")
}

#[test]
#[cfg(test)]
fn test_constant_initializer_explicit() {
    let mut context = Scope::new();

    let init = constant_initializer(&mut context, 3_i32, &[2, 2, 2]).unwrap();
    let var = context.get_variable_with_initializer(init, true, "").unwrap();

    let results = test_suite!(run_op: [var]; context, input: {});
    test_suite!(results; assert_len: {[0;Int32] == 8});
    test_suite!(results; assert: {[0;Int32] == [3_i32, 3, 3, 3, 3, 3, 3, 3]});
}

/// Initializer that generates tensors with a normal distribution.
///
/// ### Args:
/// * mean: A scalar float. Mean of the random values to generate.
/// * stddev: A scalar float. Standard deviation of the random values to generate.
/// * seed: Optional value used to create random seeds. 
///         See [set_random_seed](../prelude/struct.Scope.html#method.set_random_seed) for behavior.
/// * shape: Shape of the initializer.  
pub fn random_normal_initializer<TeS, F>(
    context: &mut Scope,
    mean: F,
    stddev: F,
    seed: Option<i32>,
    shape: &[TeS],
) -> Result<Tensor>
where
    F: Float,
    TeS: ShapeSize,
{
    let scope = &mut context.name_scope("random_normal", None);

    let shape_tensor = scope.constant(shape, &[shape.len() as i64], "")?;
    let mean_tensor = scope.constant(&[mean], &[] as &[i64], "mean")?;
    let stddev_tensor = scope.constant(&[stddev], &[] as &[i64], "name")?;

    let rnd = {
        let mut seed_ = [0_i64];
        let mut seed2_ = [0_i64];
        let (seed, seed2) = scope.get_seed(seed);
        let dtype = &[F::data_type()];
        let mut op = RandomStandardNormal::new(shape_tensor.into(), "")?.set_dtype(dtype);
        if let Some(seed) = seed {
            seed_[0] = seed as i64;
            op.set_seed(&seed_);
        };
        if let Some(seed) = seed2 {
            seed2_[0] = seed as i64;
            op.set_seed2(&seed2_);
        };
        scope.install(op)?
    };

    let mul = multiply(scope, rnd, stddev_tensor, "")?;
    add(scope, mul, mean_tensor, "")
}

add_new_op!(RandomStandardNormal,
    constructor: [add_new_op!(
        UNARY CONSTRUCTOR: RandomStandardNormal, Init: [output_type: DataType::Float]
    );],
    digest: [DEFAULT_DIGEST: RandomStandardNormal, DTYPE_ATTR],
    extra_funcs: [
        /// Default is 0.
        fn set_seed(&mut self, val: &'a [i64]) {
            self.attributes.push(("seed", false, Attribute::Int(val)));
        }

        /// Default is 0.
        fn set_seed2(&mut self, val: &'a [i64]) {
            self.attributes.push(("seed2", false, Attribute::Int(val)));
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

#[ignore]
#[test]
#[cfg(test)]
fn test_random_normal_initializer() {
    let mut context = Scope::new();

    let init = random_normal_initializer(&mut context, 0.0_f32, 1.0, None, &[2, 2]).unwrap();
    let var = context.get_variable_with_initializer(init, true, "").unwrap();

    let results = test_suite!(run_op: [var]; context, input: {});
    test_suite!(results; assert_len: {[0;Float] == 4});
}

/// Initializer that generates tensors initialized to 0.
pub fn zeros_initializer<'a, TeS>(
    context: &mut Scope,
    shape: &'a [TeS],
    dtype: DataType,
) -> Result<Constant>
where
    TeS: ShapeSize + ::std::iter::Product<&'a TeS>,
{
    // TODO: rewrite this function so it's less inefficient, avoid extra allocation of "vals"
    let elem_num: TeS = shape.iter().product();
    let elem_num = elem_num.as_u64() as usize;
    match dtype {
        DataType::Bool => {
            let vals = vec![false; elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::Double => {
            let vals = vec![0_f64; elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::Float => {
            let vals = vec![0_f32; elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::Int32 => {
            let vals = vec![0_i32; elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::UInt8 => {
            let vals = vec![0_u8; elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::Int16 => {
            let vals = vec![0_i16; elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::Int8 => {
            let vals = vec![0_i8; elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::Int64 => {
            let vals = vec![0_i64; elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::String => {
            let vals = vec!["".to_string(); elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::QUInt8 => {
            let vals = vec![::QUInt8::from(0); elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::QInt16 => {
            let vals = vec![::QInt16::from(0); elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::QUInt16 => {
            let vals = vec![::QUInt16::from(0); elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::QInt32 => {
            let vals = vec![::QInt32::from(0); elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::BFloat16 => {
            let vals = vec![::BFloat16::from(0.); elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::Complex64 => {
            let vals = vec![::Complex32::new(0., 0.); elem_num];
            context.constant(&vals, shape, "")
        }
        DataType::Complex128 => {
            let vals = vec![::Complex64::new(0., 0.); elem_num];
            context.constant(&vals, shape, "")
        }
        _ => Err(Error::from(ErrorKind::Stub)),
    }
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
) -> Result<OperationData>
where
    I: IntoIterator<Item = &'a OperationData>,
{
    let mut var = graph.new_operation("VariableV2", name)?;
    var.set_attr_type("dtype", dtype)?;
    var.set_attr_shape("shape", shape)?;
    super::add_control_input(&mut var, control_inputs);
    Ok(var.finish()?)
}

/// The inputs are a handle to the underlying tensor and the data source operation.
pub(crate) fn assign_(
    graph: &mut Graph,
    name: &str,
    reference: OperationData,
    data: (OperationData, i32),
    validate_shape: bool,
) -> Result<OperationData> {
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
    Ok(var.finish()?)
}

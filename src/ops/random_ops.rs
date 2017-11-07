//! Random operations
use super::*;

/// Outputs random values from a uniform distribution.
///
/// The generated values follow a uniform distribution in the range [minval, maxval).
/// The lower bound minval is included in the range, while the upper bound maxval is excluded.
///
/// For floats, the default range is [0, 1). For ints, at least maxval must be specified explicitly.
///
/// In the integer case, the random integers are slightly biased unless maxval - minval
/// is an exact power of two. The bias is small for values of maxval - minval significantly
/// smaller than the range of the output (either 2**32 or 2**64).
///
/// ### Args:
/// * shape: A 1-D integer Tensor or array. The shape of the output tensor.
/// * minval: A 0-D Tensor or value of type dtype. The lower bound on the range of random values to generate.
/// * maxval: A 0-D Tensor or value of type dtype. The upper bound on the range of random values to generate
/// * dtype: The type of the output: float16, float32, float64, int32, or int64.
/// * seed: Used to create a random seed for the distribution. See tf.set_random_seed for behavior.
/// * name: A name for the operation (optional, empty str for auto-generated).
///
/// ### Returns:
/// * A tensor of the specified shape filled with random uniform values.
pub fn random_uniform<S, Ts, Tmin, Tmax>(
    scope: &mut Scope,
    shape: Ts,
    minval: Tmin,
    maxval: Tmax,
    dtype: Option<DataType>,
    seed: Option<i64>,
    name: S,
) -> Result<Tensor>
where
    S: AsRef<Path>,
    Ts: TensorOps,
    Tmin: TensorOps,
    Tmax: TensorOps,
{
    let scope = &mut scope.name_scope(name.as_ref().to_str().unwrap(), Some("random_uniform"));
    let shape = shape.into_tensor(scope);
    let minval = minval.into_tensor(scope);
    let maxval = maxval.into_tensor(scope);

    if minval.dtype != maxval.dtype {
        return Err(Error::from(ErrorKind::Stub));
    }

    let mut seed_ = [0_i64];
    let mut seed2_ = [0_i64];
    let (seed, seed2) = scope.get_seed(seed);

    let dtype = if let Some(dtype) = dtype {
        dtype
    } else {
        DataType::Float
    };
    let dtype = &[dtype];

    if dtype[0].is_integer() {
        let mut op = RandomUniformInt::new(shape, minval, maxval, name)?;
        if let Some(seed) = seed {
            seed_[0] = seed;
            op.set_seed(&seed_);
        };
        if let Some(seed) = seed2 {
            seed2_[0] = seed;
            op.set_seed2(&seed2_);
        };
        scope.install(op)
    } else {
        let mut op = RandomUniform::new(shape, dtype, name)?;
        if let Some(seed) = seed {
            seed_[0] = seed;
            op.set_seed(&seed_);
        };
        if let Some(seed) = seed2 {
            seed2_[0] = seed;
            op.set_seed2(&seed2_);
        };
        let rnd = scope.install(op)?;
        let b = math_ops::sub(scope, maxval, minval, "")?;
        let a = math_ops::multiply(scope, rnd, b, "")?;
        math_ops::add(scope, a, minval, "")
    }
}

#[test]
#[cfg(test)]
fn test_random_uniform() {
    let mut context = Scope::new();
    let init = random_uniform(
        &mut context,
        [2, 2].as_ref(),
        0,
        100,
        Some(DataType::Int32),
        None,
        "",
    ).unwrap();
    let results = test_suite!(run_op: [init]; context, input: {});
    test_suite!(results; assert_len: {[0;Int32] == 4});

    let init = random_uniform(
        &mut context,
        [2, 2].as_ref(),
        0.,
        1.,
        None,
        None,
        "",
    ).unwrap();
    let results = test_suite!(run_op: [init]; context, input: {});
    test_suite!(results; assert_len: {[0;Float] == 4});
}


///  Outputs random values from a uniform distribution.
///
///  The generated values follow a uniform distribution in the range `[0, 1)`. The
///  lower bound 0 is included in the range, while the upper bound 1 is excluded.
///
///  ### Args:
///    * shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
///      The shape of the output tensor.
///    * dtype: A `tf.DType` from: `tf.half, tf.float32, tf.float64`.
///      The type of the output.
///    * seed: An optional `int`. Defaults to `0`.
///      If either `seed` or `seed2` are set to be non-zero, the random number
///      generator is seeded by the given seed.  Otherwise, it is seeded by a
///      random seed.
///    * seed2: An optional `int`. Defaults to `0`.
///      A second seed to avoid seed collision.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    A `Tensor` of type `dtype`.
///    A tensor of the specified shape filled with uniform random values.
add_new_op!(RandomUniform,
    constructor: [
        fn new<S: AsRef<Path>>(shape: Tensor, dtype: &'a [DataType], name: S) -> Result<RandomUniform<'a>> {
            Ok(
                RandomUniform {
                    ident: NodeIdent::new(),
                    elements: vec![shape],
                    name: generate_name!(is_none: name),
                    attributes: vec![("dtype", false, dtype.into())],
                    input_lists: vec![],
                    output_type: dtype[0],
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: RandomUniform, DTYPE_ATTR],
    extra_funcs: [
        /// Default is 0.
        fn set_seed(&mut self, val: &'a [i64]) {
            self.attributes.push(("seed", false, Attribute::Int(val)));
        }

        /// Default is 0.
        fn set_seed2(&mut self, val: &'a [i64]) {
            self.attributes.push(("seed2", false, Attribute::Int(val)));
        }
    ], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);


///  Outputs random integers from a uniform distribution.
///
///  The generated values are uniform integers in the range `[minval, maxval)`.
///  The lower bound `minval` is included in the range, while the upper bound
///  `maxval` is excluded.
///
///  The random integers are slightly biased unless `maxval - minval` is an exact
///  power of two.  The bias is small for values of `maxval - minval` significantly
///  smaller than the range of the output (either `2^32` or `2^64`).
///
///  ### Args:
///    * shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
///      The shape of the output tensor.
///    * minval: A `Tensor`. Must be one of the following types: `int32`, `int64`.
///      0-D.  Inclusive lower bound on the generated integers.
///    * maxval: A `Tensor`. Must have the same type as `minval`.
///      0-D.  Exclusive upper bound on the generated integers.
///    * seed: An optional `int`. Defaults to `0`.
///      If either `seed` or `seed2` are set to be non-zero, the random number
///      generator is seeded by the given seed.  Otherwise, it is seeded by a
///      random seed.
///    * seed2: An optional `int`. Defaults to `0`.
///      A second seed to avoid seed collision.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    A `Tensor`. Has the same type as `minval`.
///    A tensor of the specified shape filled with uniform random integers.
add_new_op!(RandomUniformInt,
    constructor: [
        fn new<S: AsRef<Path>>(
            shape: Tensor, 
            minval: Tensor, 
            maxval: Tensor, 
            name: S
        ) 
            -> Result<RandomUniformInt<'a>> 
        {
            Ok(
                RandomUniformInt {
                    ident: NodeIdent::new(),
                    elements: vec![shape, minval, maxval],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: vec![],
                    output_type: minval.dtype,
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: RandomUniformInt, DTYPE_ATTR],
    extra_funcs: [
        /// Default is 0.
        fn set_seed(&mut self, val: &'a [i64]) {
            self.attributes.push(("seed", false, Attribute::Int(val)));
        }

        /// Default is 0.
        fn set_seed2(&mut self, val: &'a [i64]) {
            self.attributes.push(("seed2", false, Attribute::Int(val)));
        }
    ], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);

//! Additional ops and models ported from TensorFlow Python lib.

use std::path::{Path, PathBuf};

use tf::TensorType;

use super::{DataType, Graph, TypedTensor, OperationData, Output, Shape, Status};
pub use super::framework::*;

#[doc(hidden)]
pub trait Float: TensorType {}

impl Float for f32 {}
impl Float for f64 {}

#[doc(hidden)]
pub trait ShapeSize: TensorType + Copy {
    fn as_i64(self) -> i64;
    fn as_u64(self) -> u64;
}

impl ShapeSize for i32 {
    fn as_i64(self) -> i64 {
        self as i64
    }

    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl ShapeSize for i64 {
    fn as_i64(self) -> i64 {
        self
    }

    fn as_u64(self) -> u64 {
        self as u64
    }
}

////// Macros //////

macro_rules! generate_name {
    (is_none: $name:ident) => (
        if name_cmp!($name, "") {
            None
        } else {
            Some($name.as_ref().to_owned())
        }
    );
    (ret: $name:expr) => (
        if let Some(ref name) = *$name {
            Some(name.as_path())
        } else {
            None
        }
    )
}

macro_rules! impl_into_ident {
    ($name:ident) => {
        impl<'a> Into<Ident> for $name<'a> {
            fn into(self) -> Ident {
                self.ident
            }
        }
        
        impl<'a> Into<Ident> for &'a $name<'a> {
            fn into(self) -> Ident {
                self.ident
            }
        }
    }
}

/// Macro for creating new operations.
///
/// __BINARY__: Define a binary (takes two inputs, and returns one output) operation,
/// with _n_ => 0 extra attributes, and _m_ => 0 extra functions.
///
/// Provide a custom constructor or use the default one calling `BIN CONSTRUCTOR`
/// variant of this same macro.
///
/// __UNARY__: Define a unary (takes one input, returns one output) operation,
/// with _n_ => 0 extra attributes, and _m_ => 0 extra functions.
///
/// Provide a custom constructor or use the default one calling `BIN CONSTRUCTOR`
/// variant of this same macro.
macro_rules! add_new_op {
    (
        $name:tt, 
        constructor: [$($constructor:tt)*],
        digest: [$($digest:tt)*],
        extra_funcs: [$($funcs:tt)*], 
        extra_attr: [$($attr_name:ident: $attr_ty:ty),*],
        output: [$($output:tt)*],
    ) => {
        #[derive(Debug, Clone)]
        struct $name<'a> {
            ident: Ident,
            elements: Vec<Tensor>,
            name: Option<PathBuf>,
            /// attr_name, is_list, attr_value
            attributes: Vec<(&'a str, bool, Attribute<'a>)>,
            input_lists: Vec<(usize, Vec<Tensor>)>,
            $( $attr_name: $attr_ty, )*
        }

        impl<'a> $name<'a> {
            $($constructor)*
            $($funcs)*
        }

        impl<'a> Operation<'a> for $name<'a> {
            type Outputs = $($output)*;
            add_new_op!(CORE_FN: $name);
            add_new_op!($($digest)*);
        }
        impl_into_ident!($name);
    };
    // Generic constructor for unary ops.
    (UNARY CONSTRUCTOR: $name:ident, Init: [$($attr_name:ident: $attr_init:expr),*]) => {
        fn new<S: AsRef<Path>>(x: Tensor, name: S) -> Result<$name<'a>, ::Error> {
            Ok(
                $name {
                    ident: Ident::new(),
                    elements: vec![x],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: vec![],
                    $($attr_name: $attr_init),*
                },
            )
        }
    };
    // Generic constructor for binary ops.
    (BIN CONSTRUCTOR: $name:ident, Init: [$($attr_name:ident: $attr_init:expr),*]) => {
        fn new<S: AsRef<Path>>(x: Tensor, y: Tensor, name: S) -> Result<$name<'a>, ::Error> {
            Ok(
                $name {
                    ident: Ident::new(),
                    elements: vec![x, y],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: vec![],
                    $($attr_name: $attr_init),*
                },
            )
        }
    };

    // digest fn for context
    (DEFAULT_DIGEST: $name:tt, $infer_dtype:tt) => {
        #[doc(hidden)]
        fn digest(
            self,
            context: &mut Scope,
            op: OperationData,
        ) -> Result<Self::Outputs, ::Error> {
            let (ident, idtype, dtype) = add_new_op!(
                REGISTER_SELF: (self, context, op); $name, $infer_dtype);
            let tensor = Tensor {
                ident,
                idtype,
                dtype,
                idx: 0,
            };
            match context.control_context {
                ControlFlow::CondContext(ref mut cond) => {
                    cond.values.insert(ident);
                    cond.external_values.insert(ident, tensor); 
                }
                ControlFlow::WhileContext(ref mut cond) => {
                    cond.values.insert(ident);
                    cond.external_values.insert(ident, tensor); 
                }
                ControlFlow::None => {}
            }
            Ok(tensor)
        }
    };
    (DIGEST: $($digest:tt)*) => { $($digest)* };

    (REGISTER_SELF: ($SELF:ident, $context:ident, $op:ident); $name:tt, $infer_dtype:tt) => {{
        let ident = Ident::new();
        let dtype = add_new_op!($infer_dtype $SELF);
        let idtype = IdType::Operation(stringify!($name));
        let full_name = $context.resolve_tensor_name($SELF.get_op_name(), idtype, false)?;
        let shape = {
            let g = &*$context.graph.borrow();
            g.tensor_shape(
                    Output {
                        operation: $op.clone(),
                        index: 0,
                    },
                )?
        };
        {
            let reg = &mut *$context.registry.borrow_mut();
            $context.own_scope.ops.push((full_name.clone(), ident));
            reg.insert(
                ident,
                TensorData {
                    full_name,
                    dtype,
                    idtype,
                    data_origin: ($op, 0),
                    shape,
                },
            );
        }
        (ident, idtype, dtype)
    }};
    // extra funcs:
    (CORE_FN: $op_name:tt) => {
        fn get_op_type_name(&self) -> &'static str {
            stringify!($op_name)
        }

        fn get_op_name(&self) -> Option<&Path> {
            generate_name!(ret: &self.name)
        }

        fn fetch_inputs(&self) -> &[Tensor] {
            &self.elements
        }

        fn fetch_input_lists(&self) -> &[(usize, Vec<Tensor>)] {
            &self.input_lists
        }

        fn fetch_attributes<'s>(&'s self) 
            -> &'s [(&str, bool, Attribute<'a>)] 
        {
            &self.attributes
        }
    };
    // DataType inference:
    (INPUT0 $s:ident) => ($s.elements[0].dtype);
    (DTYPE_ATTR $s:ident) => ($s.output_type);
    (NONE $s:ident) => (DataType::Resource)
}

#[allow(unused_macros)]
macro_rules! test_suite {
    (run_op: [$($op:ident),+]; $context:ident, input: {$($arg:tt),*}) => {{
        use client::ClientSession;
        let mut session = ClientSession::new(&mut $context).unwrap();
        session.fetch(vec![$($op),+]);
        $( session.feed(vec![$arg]) )*
        session.run(None).unwrap()
    }};
    ($res:ident; assert: {$([$res_idx:expr; $ty:ident] == $cmp:expr),+}) => {{
        $(
            match $res[$res_idx] {
                TensorContent::$ty(ref val) => {
                    for (i, n) in (&$cmp).iter().enumerate() {
                        assert_eq!(val[i], *n);
                    }
                }
                _ => panic!("wrong type specified for this test")
            }
        )+
    }};
    ($res:ident; assert_len: {$([$res_idx:expr; $ty:ident] == $cmp:expr),+}) => {{
        $(
            match $res[$res_idx] {
                TensorContent::$ty(ref val) => {
                    assert_eq!(val.len(), $cmp)
                }
                _ => panic!("wrong type specified for this test")
            }
        )+
    }};
    (out: $op:expr, $idx:expr) => (
        Output {
            operation: $op,
            index: $idx,
        }
    )
}

pub(crate) mod array_ops;
pub use self::array_ops::*;

pub(crate) mod control_flow_ops;
pub use self::control_flow_ops::*;

pub(crate) mod init_ops;
pub use self::init_ops::*;

pub(crate) mod math_ops;
pub use self::math_ops::*;

pub(crate) mod state_ops;
pub use self::state_ops::*;


//////////////////////
//    Helper ops    //
//////////////////////

/*
#[derive(Debug, Clone)]
pub struct AvgUpdatingOp;

impl<'a> Operation<'a> for AvgUpdatingOp {
    type Outputs = ();
}

impl Into<Ident> for AvgUpdatingOp {
    fn into(self) -> Ident {
        unimplemented!()
    }
}

impl Into<Tensor> for AvgUpdatingOp {
    fn into(self) -> Tensor {
        unimplemented!()
    }
}
*/

/*
/// Create an op that groups multiple operations.
///
/// When this op finishes, all ops in input have finished. This op has no output.
pub struct GroupedOps;

impl GroupedOps {
    pub fn new<S>(name: S) -> GroupedOps
        where S: Into<String>
    {
        unimplemented!()
    }

    pub fn group<O>(self, op: O) -> Self
        where O: Into<Ident>
    {
        unimplemented!()
    }
}

impl<'a> Operation<'a> for GroupedOps {
    type Outputs = ();
}

impl Into<Ident> for GroupedOps {
    fn into(self) -> Ident {
        unimplemented!()
    }
}

impl Into<Tensor> for GroupedOps {
    fn into(self) -> Tensor {
        unimplemented!()
    }
}
*/

/////////////////////////////////
//   Pre-built train models    //
/////////////////////////////////

/*
pub mod nn {
    use super::*;

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
}

#[derive(Debug, Clone)]
pub(crate) struct Optimizer;

/*
beta1: Option<f64>, // 0.9
beta2: Option<f64>, // 0.999
epsilon: Option<f64>, // 1e-08
*/

impl Optimizer {
    pub fn new<V>(kind: Optimizers, learning_rate: V, use_locking: bool) -> Self
        where V: Into<Tensor>
    {
        unimplemented!()
    }

    pub fn configure(&mut self) -> &mut OptimizerInterface {
        unimplemented!()
    }

    pub fn composite_optimizer(optimizer1: Optimizer,
                               optimizer2: Optimizer,
                               switch: bool,
                               use_locking: bool)
                               -> Self {
        unimplemented!()
    }

    pub fn compute_gradients(&mut self,
                             loss: Tensor,
                             var_list: Vec<Tensor>)
                             -> Vec<(Option<Tensor>, Tensor)> {
        unimplemented!()
    }

    pub fn apply_gradients(&mut self,
                           grads_and_vars: Vec<(Tensor, Tensor)>,
                           global_step: Tensor)
                           -> Tensor {
        unimplemented!()
    }
}

pub(crate) trait OptimizerInterface {
    fn add_param(&mut self, name: &str, value: f64) {
        unimplemented!()
    }
}

pub enum Optimizers {
    GradientDescent,
    Adam,
    LazyAdam,
    Momentum,
    Composite,
}

// GradientDescentOptimizer
#[derive(Clone)]
struct GradientDescentOptimizer;

impl GradientDescentOptimizer {}


// AdamOptimizer
#[derive(Clone)]
struct AdamOptimizer;

impl AdamOptimizer {}


// LazyAdamOptimizer
#[derive(Clone)]
struct LazyAdamOptimizer;

impl LazyAdamOptimizer {}

// MomentumOptimizer
#[derive(Clone)]
struct MomentumOptimizer;

impl MomentumOptimizer {}


// CompositeOptimizer
#[derive(Clone)]
struct CompositeOptimizer;

impl CompositeOptimizer {}


#[derive(Debug, Clone)]
pub struct ExponentialDecay;

impl ExponentialDecay {
    pub fn new(base_rate: Tensor,
               step_var: Tensor,
               decay_steps: u32,
               decay_base: f64,
               staircase: bool)
               -> ExponentialDecay {
        ExponentialDecay
    }

    pub fn run(self) -> Result<Tensor, ::Error> {
        unimplemented!()
    }
}

/// Maintains moving averages of variables by employing an exponential decay.
///
/// When training a model, it is often beneficial to maintain moving averages of
/// the trained parameters.  Evaluations that use averaged parameters sometimes
/// produce significantly better results than the final trained values.
///
/// The `apply()` method adds shadow copies of trained variables and add ops that
/// maintain a moving average of the trained variables in their shadow copies.
/// It is used when building the training model.  The ops that maintain moving
/// averages are typically run after each training step.
/// The `average()` and `average_name()` methods give access to the shadow
/// variables and their names.  They are useful when building an evaluation
/// model, or when restoring a model from a checkpoint file.  They help use the
/// moving averages in place of the last trained values for evaluations.
///
/// The moving averages are computed using exponential decay.  You specify the
/// decay value when creating the `ExponentialMovingAverage` object.  The shadow
/// variables are initialized with the same initial values as the trained
/// variables.  When you run the ops to maintain the moving averages, each
/// shadow variable is updated with the formula:
///
/// `shadow_variable -= (1 - decay) * (shadow_variable - variable)`
///
/// This is mathematically equivalent to the classic formula below, but the use
/// of an `assign_sub` op (the `"-="` in the formula) allows concurrent lockless
/// updates to the variables:
///
/// `shadow_variable = decay * shadow_variable + (1 - decay) * variable`
///
/// Reasonable values for `decay` are close to 1.0, typically in the
/// multiple-nines range: 0.999, 0.9999, etc.
#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage;

impl ExponentialMovingAverage {
    /// Creates a new ExponentialMovingAverage object.
    ///
    /// The `apply()` method has to be called to create shadow variables and add
    /// ops to maintain moving averages.
    ///
    /// The optional `num_updates` parameter allows one to tweak the decay rate
    /// dynamically. It is typical to pass the count of training steps, usually
    /// kept in a variable that is incremented at each step, in which case the
    /// decay rate is lower at the start of training.  This makes moving averages
    /// move faster.  If passed, the actual decay rate used is:
    ///
    ///   `min(decay, (1 + num_updates) / (10 + num_updates))`
    ///
    /// Args:
    ///   decay: The decay to use.
    ///   num_updates: Optional count of number of updates applied to variables.
    ///   zero_debias: If `True`, zero debias moving-averages that are initialized
    ///     with tensors.
    ///   name: String. Optional prefix name to use for the name of ops added in
    ///     `apply()`.
    pub fn new() -> ExponentialMovingAverage {
        /*
        self._decay = decay
        self._num_updates = num_updates
        self._zero_debias = zero_debias
        self._name = name
        self._averages = {}
        */
        unimplemented!()
    }

    /// Maintains moving averages of variables.
    ///
    /// `var_list` must be a list of `Tensor` or `Tensor` objects.  This method
    /// creates shadow variables for all elements of `var_list`.  Shadow variables
    /// for `Tensor` objects are initialized to the variable's initial value.
    /// They will be added to the `GraphKeys.MOVING_AVERAGE_VARIABLES` collection.
    /// For `Tensor` objects, the shadow variables are initialized to 0 and zero
    /// debiased (see docstring in `assign_moving_average` for more details).
    ///
    /// shadow variables are created with `trainable=False` and added to the
    /// `GraphKeys.ALL_VARIABLES` collection.  They will be returned by calls to
    /// `tf.global_variables()`.
    ///
    /// Returns an op that updates all shadow variables as described above.
    ///
    /// Note that `apply()` can be called multiple times with different lists of
    /// variables.
    ///
    /// Args:
    ///   var_list: A list of Tensor or Tensor objects. The variables
    ///     and Tensors must be of types float16, float32, or float64.
    ///
    /// Returns:
    ///   An Operation that updates the moving averages.
    ///
    /// Raises:
    ///   TypeError: If the arguments are not all float16, float32, or float64.
    ///   ValueError: If the moving average of one of the variables is already
    ///     being computed.
    pub fn apply(&self, var_list: &[Tensor]) -> Result<AvgUpdatingOp, ::Error> {
        /*
        # TODO(touts): op_scope
        if var_list is None:
          var_list = variables.trainable_variables()
        zero_debias_true = set()  # set of vars to set `zero_debias=True`
        for var in var_list:
          if var.dtype.base_dtype not in [dtypes.float16, dtypes.float32,
                                          dtypes.float64]:
            raise TypeError("The variables must be half, float, or double: %s" %
                            var.name)
          if var in self._averages:
            raise ValueError("Moving average already computed for: %s" % var.name)

          # For variables: to lower communication bandwidth across devices we keep
          # the moving averages on the same device as the variables. For other
          # tensors, we rely on the existing device allocation mechanism.
          with ops.control_dependencies(None):
            if isinstance(var, variables.Tensor):
              avg = slot_creator.create_slot(var,
                                            var.initialized_value(),
                                            self._name,
                                            colocate_with_primary=True)
              # NOTE(mrry): We only add `tf.Tensor` objects to the
              # `MOVING_AVERAGE_VARIABLES` collection.
              ops.add_to_collection(ops.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
            else:
              avg = slot_creator.create_zeros_slot(
                  var,
                  self._name,
                  colocate_with_primary=(var.op.type in ["Tensor", "VariableV2"]))
              if self._zero_debias:
                zero_debias_true.add(avg)
          self._averages[var] = avg

        with ops.name_scope(self._name) as scope:
          decay = ops.convert_to_tensor(self._decay, name="decay")
          if self._num_updates is not None:
            num_updates = math_ops.cast(self._num_updates,
                                        dtypes.float32,
                                        name="num_updates")
            decay = math_ops.minimum(decay,
                                    (1.0 + num_updates) / (10.0 + num_updates))
          updates = []
          for var in var_list:
            zero_debias = self._averages[var] in zero_debias_true
            updates.append(assign_moving_average(
                self._averages[var], var, decay, zero_debias=zero_debias))
          return control_flow_ops.group(*updates, name=scope)
        */
        unimplemented!()
    }

    /// Returns the `Tensor` holding the average of `var`.
    ///
    /// Args:
    ///   var: A `Tensor` object.
    ///
    /// Returns:
    ///   A `Tensor` object or `None` if the moving average of `var`
    ///   is not maintained.
    fn average(&self) {
        // return self._averages.get(var, None)
        unimplemented!()
    }

    /// Returns the name of the `Tensor` holding the average for `var`.
    ///
    /// The typical scenario for `ExponentialMovingAverage` is to compute moving
    /// averages of variables during training, and restore the variables from the
    /// computed moving averages during evaluations.
    ///
    /// To restore variables, you have to know the name of the shadow variables.
    /// That name and the original variable can then be passed to a `Saver()` object
    /// to restore the variable from the moving average value with:
    ///   `saver = tf.train.Saver({ema.average_name(var): var})`
    ///
    /// `average_name()` can be called whether or not `apply()` has been called.
    ///
    /// Args:
    ///  var: A `Tensor` object.
    ///
    /// Returns:
    ///   A string: The name of the variable that will be used or was used
    ///   by the `ExponentialMovingAverage class` to hold the moving average of
    ///   `var`.
    fn average_name(&self) {
        /*
        if var in self._averages:
          return self._averages[var].op.name
        return ops.get_default_graph().unique_name(
            var.op.name + "/" + self._name, mark_as_used=False)
        */
    }

    /// Returns a map of names to `Variables` to restore.
    ///
    /// If a variable has a moving average, use the moving average variable name as
    /// the restore name; otherwise, use the variable name.
    ///
    /// For example,
    ///
    /// ```python
    ///   variables_to_restore = ema.variables_to_restore()
    ///   saver = tf.train.Saver(variables_to_restore)
    /// ```
    ///
    /// Below is an example of such mapping:
    ///
    /// ```bash
    ///   conv/batchnorm/gamma/ExponentialMovingAverage: conv/batchnorm/gamma,
    ///   conv_4/conv2d_params/ExponentialMovingAverage: conv_4/conv2d_params,
    ///   global_step: global_step
    /// ```
    /// Args:
    ///   moving_avg_variables: a list of variables that require to use of the
    ///     moving variable name to be restored. If None, it will default to
    ///     variables.moving_average_variables() + variables.trainable_variables()
    ///
    /// Returns:
    ///   A map from restore_names to variables. The restore_name can be the
    ///   moving_average version of the variable name if it exist, or the original
    ///   variable name.
    fn variables_to_restore(&self) {
        /*
        name_map = {}
        if moving_avg_variables is None:
          # Include trainable variables and variables which have been explicitly
          # added to the moving_average_variables collection.
          moving_avg_variables = variables.trainable_variables()
          moving_avg_variables += variables.moving_average_variables()
        # Remove duplicates
        moving_avg_variables = set(moving_avg_variables)
        # Collect all the variables with moving average,
        for v in moving_avg_variables:
          name_map[self.average_name(v)] = v
        # Make sure we restore variables without moving average as well.
        for v in list(set(variables.global_variables()) - moving_avg_variables):
          if v.op.name not in name_map:
            name_map[v.op.name] = v
        return name_map
        */
    }
}
*/

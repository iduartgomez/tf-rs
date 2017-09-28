use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

use super::*;
use ops::control_flow_ops::Group;
use ops::math_ops;
use ops::state_ops;

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
pub struct ExponentialMovingAverage {
    averages: HashMap<Tensor, Variable>,
    decay: Tensor,
    num_updates: Option<Tensor>,
    zero_debias: bool,
    name: String,
}

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
    pub fn new<Tx, Ty>(
        decay: Tx,
        num_updates: Option<Ty>,
        zero_debias: bool,
        name: &str,
    ) -> ExponentialMovingAverage
    where
        Tx: Into<Tensor>,
        Ty: Into<Tensor>,
    {
        let num_updates = if let Some(num_updates) = num_updates {
            Some(num_updates.into())
        } else {
            None
        };
        ExponentialMovingAverage {
            averages: HashMap::new(),
            decay: decay.into(),
            num_updates,
            zero_debias,
            name: name.to_owned(),
        }
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
    pub fn apply(&mut self, context: &mut Scope, var_list: &[Tensor]) -> Result<Group, ::Error> {
        let mut zero_debias_true: HashSet<NodeIdent> = HashSet::new(); // set of vars to set to `zero_debias=True`
        for var in var_list {
            match var.dtype {
                DataType::Float | DataType::Double => {}
                _ => return Err(::Error::Stub),
            }

            if self.averages.keys().find(|&&x| x.ident == var.ident).is_some() {
                return Err(::Error::Stub);
            }

            // For variables: to lower communication bandwidth across devices we keep
            // the moving averages on the same device as the variables. For other
            // tensors, we rely on the existing device allocation mechanism.
            let scope = &mut context.clear_control_dependencies();
            let avg;
            if var.is_ref() {
                let init = var.get_initializer(scope)?;
                avg = create_slot(scope, *var, init, &self.name, true)?;
            } else {
                avg = create_zeros_slot(scope, *var, &self.name, true)?;
                if self.zero_debias {
                    zero_debias_true.insert(avg.clone().into());
                }
            }
            self.averages.insert(*var, avg);
        }

        let scope = &mut context.name_scope(self.name.as_str(), None);
        let mut decay: Tensor = self.decay;
        if let Some(mut num_updates) = self.num_updates {
            num_updates = math_ops::cast(scope, num_updates, DataType::Float, "num_updates")?;
            let c0 = scope.constant(&[1.0_f32], &[] as &[i32], "")?;
            let c1 = scope.constant(&[10.0_f32], &[] as &[i32], "")?;
            let s0 = math_ops::add(scope, c0, num_updates, "")?;
            let s1 = math_ops::add(scope, c1, num_updates, "")?;
            let n = math_ops::divide(scope, s0, s1, "")?;
            decay = math_ops::minimum(scope, self.decay, n, "")?;
        }
        let mut updates = vec![];
        for var in var_list {
            let zero_debias = zero_debias_true.get(&self.averages[var].get_ident()).is_some();
            updates.push(
                assign_moving_average(
                    scope,
                    &self.averages[var],
                    var,
                    decay.into(),
                    zero_debias,
                    None,
                )?,
            )
        }
        Group::new(scope, &updates, "")
    }

    /// Returns the `Tensor` holding the average of `var`.
    ///
    /// Args:
    ///   var: A `Tensor` object.
    ///
    /// Returns:
    ///   A `Tensor` object or `None` if the moving average of `var`
    ///   is not maintained.
    pub fn average(&self, var: &Tensor) -> Option<&Variable> {
        self.averages.get(var)
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
    pub fn average_name(&self, context: &Scope, var: &Tensor) -> String {
        if let Some(ref var) = self.averages.get(var) {
            var.get_name(context)
        } else {
            /*
            return ops.get_default_graph().unique_name(
                var.op.name + "/" + self._name, mark_as_used=False)
            */
            format!("{}/{}", var.get_name(context), self.name)
        }
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
    ///     moving variable name to be restored. If empty, it will default to
    ///     context.moving_average_variables() + context.trainable_variables()
    ///
    /// Returns:
    ///   A map from restore_names to variables. The restore_name can be the
    ///   moving_average version of the variable name if it exist, or the original
    ///   variable name.
    pub fn variables_to_restore(
        &self,
        scope: &Scope,
        moving_avg_variables: &[Variable],
    ) -> HashMap<String, Tensor> {
        let mut name_map: HashMap<String, Tensor> = HashMap::new();

        if moving_avg_variables.len() == 0 {
            // Include trainable variables and variables which have been explicitly
            // added to the moving_average_variables collection.
            // TODO: moving_avg_variables = variables.trainable_variables()
            //       moving_avg_variables += variables.moving_average_variables()
        }

        // Remove duplicates
        let moving_avg_variables: HashSet<Tensor> =
            HashSet::from_iter(moving_avg_variables.iter().map(|x| x.clone().into()));
        // Collect all the variables with moving average.
        for v in &moving_avg_variables {
            name_map.insert(self.average_name(scope, v), v.clone());
        }
        /*
        TODO: // Make sure we restore variables without moving average as well.
        for v in list(set(variables.global_variables()) - moving_avg_variables):
          if v.op.name not in name_map:
            name_map[v.op.name] = v
        */
        name_map
    }
}

///   Compute the moving average of a variable.
///
///   The moving average of 'variable' updated with 'value' is:
///     variable * decay + value * (1 - decay)
///
///   The returned Operation sets 'variable' to the newly computed moving average.
///
///   The new value of 'variable' can be set with the 'AssignSub' op as:
///      variable -= (1 - decay) * (variable - value)
///
///   Since variables that are initialized to a `0` value will be `0` biased,
///   `zero_debias` optionally enables scaling by the mathematically correct
///   debiasing factor of
///     1 - decay ** num_updates
///   See `ADAM: A Method for Stochastic Optimization` Section 3 for more details
///   (https://arxiv.org/abs/1412.6980).
///
///   Args:
///     variable: A Variable.
///     value: A tensor with the same shape as 'variable'.
///     decay: A float Tensor or float value.  The moving average decay.
///     zero_debias: A python bool. If true, assume the variable is 0-initialized and
///       unbias it, as in https://arxiv.org/abs/1412.6980. See docstring in
///       `_zero_debias` for more details.
///     name: Optional name of the returned operation.
///
///   Returns:
///     A reference to the input 'variable' tensor with the newly computed
///     moving average.
fn assign_moving_average(
    scope: &mut Scope,
    variable: &Variable,
    value: &Tensor,
    decay: Tensor,
    zero_debias: bool,
    name: Option<&str>,
) -> Result<Tensor, ::Error> {
    let scope = &mut if let Some(name) = name {
                         scope.name_scope(name, None)
                     } else {
                         scope.name_scope("", Some("AssignMovingAvg"))
                     };
    // TODO: colocate_with(variable)
    let decay = {
        let n = scope.constant(&[1], &[] as &[i32], "")?;
        math_ops::sub(scope, n, decay, "decay")?
    };
    let update_delta = if zero_debias {
        _zero_debias(scope, variable, value, &decay)?
    } else {
        let sub = math_ops::sub(scope, *variable, *value, "")?;
        math_ops::multiply(scope, sub, decay, "")?
    };
    let name = scope.name().to_owned();
    state_ops::assign_sub(scope, *variable, update_delta, false, name)
}

///  Compute the delta required for a debiased Variable.
///
///  All exponential moving averages initialized with Tensors are initialized to 0,
///  and therefore are biased to 0. Variables initialized to 0 and used as EMAs are
///  similarly biased. This function creates the debias updated amount according to
///  a scale factor, as in https://arxiv.org/abs/1412.6980.
///
///  To demonstrate the bias the results from 0-initialization, take an EMA that
///  was initialized to `0` with decay `b`. After `t` timesteps of seeing the
///  constant `c`, the variable have the following value:
///
///  ```maths
///    EMA = 0*b^(t) + c*(1 - b)*b^(t-1) + c*(1 - b)*b^(t-2) + ...
///        = c*(1 - b^t)
///  ```
///
///  To have the true value `c`, we would divide by the scale factor `1 - b^t`.
///
///  In order to perform debiasing, we use two shadow variables. One keeps track of
///  the biased estimate, and the other keeps track of the number of updates that
///  have occurred.
///
///  Args:
///    unbiased_var: A Variable representing the current value of the unbiased EMA.
///    value: A Tensor representing the most recent value.
///    decay: A Tensor representing `1-decay` for the EMA.
///
///  Returns:
///    The amount that the unbiased variable should be updated. Computing this
///    tensor will also update the shadow variables appropriately.
fn _zero_debias(
    scope: &mut Scope,
    unbiased_var: &Variable,
    value: &Tensor,
    decay: &Tensor,
) -> Result<Tensor, ::Error> {
    let scope_name = unbiased_var.get_name(scope);
    let scope = &mut scope.variable_scope(&scope_name, None, None)?;
    // TODO: colocate_with(unbiased_var)
    let scope = &mut scope.clear_control_dependencies();

    let unbiased_var_shape = unbiased_var.get_shape(scope);
    let biased_var =
        scope.get_variable(Some(unbiased_var.dtype), Some(unbiased_var_shape), "biased")?;
    let local_step =
        scope.get_variable(Some(unbiased_var.dtype), Some(&[] as &[i64]), "local_step")?;

    // constants:
    let one = scope.constant(&[1_i32], &[] as &[i32], "")?;

    // Get an update ops for both shadow variables.
    let update_biased = {
        let sub = math_ops::sub(scope, biased_var, *value, "")?;
        let mul = math_ops::multiply(scope, sub, *decay, "")?;
        state_ops::assign_sub(scope, biased_var, mul, false, &scope_name)?
    };
    let update_local_step = {
        state_ops::assign_add(scope, local_step, one, false, "")?
    };

    // Compute the value of the delta to update the unbiased EMA. Make sure to
    // use the new values of the biased variable and the local step.
    let scope = &mut scope.control_dependencies(&[update_biased, update_local_step]);
    {
        // This function gets `1 - decay`, so use `1.0 - decay` in the exponent.
        let a = math_ops::sub(scope, one, *decay, "")?;
        let pow = math_ops::pow(scope, a, local_step, "")?;

        let b = math_ops::sub(scope, one, pow, "")?;
        let c = math_ops::sub(scope, *unbiased_var, biased_var, "")?;

        math_ops::divide(scope, c, b, "unbiased_ema_delta")
    }
}

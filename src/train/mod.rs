mod slot_creator;

use std::collections::{HashMap, HashSet};

use self::slot_creator::*;
use super::*;
use super::framework::*;

use ops::math_ops;

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
    decay: f32,
    num_updates: Option<u32>,
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
    pub fn new(
        decay: f32,
        num_updates: Option<u32>,
        zero_debias: bool,
        name: String,
    ) -> ExponentialMovingAverage {
        let num_updates;
        ExponentialMovingAverage {
            averages: HashMap::new(),
            decay,
            num_updates,
            zero_debias,
            name,
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
    pub fn apply(&mut self, context: &mut Scope, var_list: &[Tensor]) -> Result<(), ::Error> {
        let mut zero_debias_true: HashSet<Ident> = HashSet::new(); // set of vars to set to `zero_debias=True`
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

        let scope = &mut context.name_scope(&self.name);
        let decay = scope.constant("decay", &[self.decay], &[] as &[i32]);
        if let Some(num_updates) = self.num_updates {
            //num_updates = math_ops::cast(scope, num_updates, DataType::Float, "");
            //let n = (1.0 + num_updates) / (10.0 + num_updates);
            //let decay = math_ops::minimum(scope, decay, n);
        }
        let mut updates: Vec<i32> = vec![];
        for var in var_list {
            let zero_debias = zero_debias_true.get(self.averages[var].get_ident());
            updates.push(assign_moving_average(self.averages[var], var, decay, zero_debias),)
        }

        Ok(())
    }

    /// Returns the `Tensor` holding the average of `var`.
    ///
    /// Args:
    ///   var: A `Tensor` object.
    ///
    /// Returns:
    ///   A `Tensor` object or `None` if the moving average of `var`
    ///   is not maintained.
    fn average(&self, var: &Tensor) -> Option<&Variable> {
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
    fn average_name(&self, context: &Scope, var: &Tensor) -> String {
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
    ///     moving variable name to be restored. If None, it will default to
    ///     variables.moving_average_variables() + variables.trainable_variables()
    ///
    /// Returns:
    ///   A map from restore_names to variables. The restore_name can be the
    ///   moving_average version of the variable name if it exist, or the original
    ///   variable name.
    fn variables_to_restore(&self) {
        let name_map: HashMap<String, String> = HashMap::new();
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

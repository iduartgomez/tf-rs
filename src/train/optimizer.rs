use std::path::Path;

use super::*;
use framework::Operation;

trait Optimizer {
    /// Apply gradients to variables.
    ///
    /// This is the second part of minimize(). It returns an Operation that applies gradients.
    ///
    /// ### Args:
    /// * grads_and_vars: List of (gradient, variable) pairs as returned by compute_gradients().
    /// * global_step: Optional Variable to increment by one after the variables have been updated.
    /// * name: Optional name for the returned operation.
    ///         Default to the name passed to the Optimizer constructor.
    ///
    /// ### Returns:
    /// * An `Operation` that applies the specified gradients. If `global_step` was not None,
    ///   that operation also increments `global_step`.
    fn apply_gradients<S: AsRef<Path>>(
        &mut self,
        scope: Scope,
        grads_and_vars: Vec<(Option<Tensor>, Variable)>,
        global_step: Option<Variable>,
        name: S,
    ) -> Tensor;

    /// Compute gradients of `loss` for the variables in `var_list`.
    ///
    /// This is the first part of `minimize()`. It returns a list of (gradient, variable) pairs
    /// where "gradient" is the gradient for "variable". Note that "gradient" can be a Tensor,
    ///  or None if there is no gradient for the given variable.
    ///
    /// ### Args:
    /// * loss: A Tensor containing the value to minimize.
    /// * var_list: Optional list or tuple of tf.Variable to update to minimize loss. Defaults to the list of variables collected in the graph under the key GraphKey.TRAINABLE_VARIABLES.
    /// * gate_gradients: How to gate the computation of gradients. Can be GATE_NONE, GATE_OP, or GATE_GRAPH.
    /// * aggregation_method: Specifies the method used to combine gradient terms. Valid values are defined in the class AggregationMethod.
    /// * colocate_gradients_with_ops: If True, try colocating gradients with the corresponding op.
    /// * grad_loss: Optional. A Tensor holding the gradient computed for loss.
    ///
    /// ### Returns:
    /// * A list of (gradient, variable) pairs. Variable is always present, but gradient can be None.
    fn compute_gradients<Tx: TensorOps>(
        &mut self,
        scope: Scope,
        loss: Tx,
        var_list: Option<Vec<Variable>>,
        gate_gradients: GateGradients,
        aggregation_method: &str,
        colocate_gradients_with_ops: bool,
        grad_loss: Option<Tensor>,
    ) -> Vec<(Option<Tensor>, Variable)>;

    /// Add operations to minimize loss by updating var_list.
    ///
    /// This method simply combines calls compute_gradients() and apply_gradients().
    /// If you want to process the gradient before applying them call compute_gradients()
    /// and apply_gradients() explicitly instead of using this function.
    ///
    /// ### Args:
    /// * loss: A Tensor containing the value to minimize.
    /// * global_step: Optional Variable to increment by one after the variables have been updated.
    /// * var_list: Optional list or tuple of Variable objects to update to minimize loss.
    /// * gate_gradients: How to gate the computation of gradients.
    /// * aggregation_method: Specifies the method used to combine gradient terms. Valid values are defined in the class AggregationMethod.
    /// * colocate_gradients_with_ops: If True, try colocating gradients with the corresponding op.
    /// * name: Optional name for the returned operation.
    /// * grad_loss: Optional. A Tensor holding the gradient computed for loss.
    ///
    /// ### Returns:
    /// * An Operation that updates the variables in var_list. If global_step was not None, that operation also increments global_step.
    fn minimize<Tx: TensorOps, S: AsRef<Path>>(
        &mut self,
        scope: Scope,
        loss: Tx,
        global_step: Option<Variable>,
        var_list: Option<Vec<Variable>>,
        gate_gradients: GateGradients,
        aggregation_method: &str,
        colocate_gradients_with_ops: bool,
        name: S,
        grad_loss: Option<Tensor>,
    ) -> Tensor;

    fn get_name(&self) -> &str;

    /// Return a slot named name created for var by the Optimizer.
    ///
    /// Some Optimizer subclasses use additional variables.
    /// For example Momentum and Adagrad use variables to accumulate updates.
    /// This method gives access to these Variable objects if for some reason you need them.
    ///
    /// Use get_slot_names() to get the list of slot names created by the Optimizer.
    fn get_slot<S: AsRef<Path>>(&self, var: Variable, name: S) -> Option<Variable>;

    /// Return a list of the names of slots created by the Optimizer.
    fn get_slot_names(&self) -> Vec<&str>;
}

enum GateGradients {
    GateNone,
    GateOp,
    GateGraph,
}

///// GradientDescentOptimizer /////

#[derive(Clone)]
/// Optimizer that implements the gradient descent algorithm.
struct GradientDescentOptimizer;

impl GradientDescentOptimizer {
    fn new<V, S>(learning_rate: V, use_locking: bool, name: S) -> Self
    where
        V: TensorOps,
        S: AsRef<Path>,
    {
        unimplemented!()
    }
}

impl Optimizer for GradientDescentOptimizer {
    fn apply_gradients<S: AsRef<Path>>(
        &mut self,
        scope: Scope,
        grads_and_vars: Vec<(Option<Tensor>, Variable)>,
        global_step: Option<Variable>,
        name: S,
    ) -> Tensor {
        unimplemented!()
    
    }
    fn compute_gradients<Tx: TensorOps>(
        &mut self,
        scope: Scope,
        loss: Tx,
        var_list: Option<Vec<Variable>>,
        gate_gradients: GateGradients,
        aggregation_method: &str,
        colocate_gradients_with_ops: bool,
        grad_loss: Option<Tensor>,
    ) -> Vec<(Option<Tensor>, Variable)> {
        unimplemented!()
    } 

    fn minimize<Tx: TensorOps, S: AsRef<Path>>(
        &mut self,
        scope: Scope,
        loss: Tx,
        global_step: Option<Variable>,
        var_list: Option<Vec<Variable>>,
        gate_gradients: GateGradients,
        aggregation_method: &str,
        colocate_gradients_with_ops: bool,
        name: S,
        grad_loss: Option<Tensor>,
    ) -> Tensor {
        unimplemented!()
    }

    fn get_name(&self) -> &str {
        unimplemented!()
    }

    fn get_slot<S: AsRef<Path>>(&self, var: Variable, name: S) -> Option<Variable> {
        unimplemented!()
    }

    fn get_slot_names(&self) -> Vec<&str> {
        unimplemented!()
    }
}

/*
fn configure(&mut self) -> &mut OptimizerInterface;

fn composite_optimizer(
    optimizer1: Optimizer,
    optimizer2: Optimizer,
    switch: bool,
    use_locking: bool,
) -> Self;
*/

/*
// AdamOptimizer
#[derive(Clone)]
struct AdamOptimizer;

impl AdamOptimizer {
    fn apply_gradients<S: AsRef<Path>>(
        &mut self,
        scope: Scope,
        grads_and_vars: Vec<(Option<Tensor>, Variable)>,
        global_step: Option<Variable>,
        name: S,
    ) -> Tensor {
        unimplemented!()
    
    }
    fn compute_gradients<Tx: TensorOps>(
        &mut self,
        scope: Scope,
        loss: Tx,
        var_list: Option<Vec<Variable>>,
        gate_gradients: GateGradients,
        aggregation_method: &str,
        colocate_gradients_with_ops: bool,
        grad_loss: Option<Tensor>,
    ) -> Vec<(Option<Tensor>, Variable)> {
        unimplemented!()
    } 

    fn minimize<Tx: TensorOps, S: AsRef<Path>>(
        &mut self,
        scope: Scope,
        loss: Tx,
        global_step: Option<Variable>,
        var_list: Option<Vec<Variable>>,
        gate_gradients: GateGradients,
        aggregation_method: &str,
        colocate_gradients_with_ops: bool,
        name: S,
        grad_loss: Option<Tensor>,
    ) -> Tensor {
        unimplemented!()
    }

    fn get_name(&self) -> &str {
        unimplemented!()
    }

    fn get_slot<S: AsRef<Path>>(&self, var: Variable, name: S) -> Option<Variable> {
        unimplemented!()
    }

    fn get_slot_names(&self) -> Vec<&str> {
        unimplemented!()
    }
}


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

    pub fn run(self) -> Result<Tensor> {
    }
}
*/

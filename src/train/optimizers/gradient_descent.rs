//! Gradient descent optimizer

use super::*;
use std::path::PathBuf;

#[derive(Clone)]
/// Optimizer that implements the gradient descent algorithm.
pub struct GradientDescentOptimizer {
    learning_rate: Tensor,
    use_locking: bool,
    name: PathBuf,
    slots: SlotDict,
}

impl GradientDescentOptimizer {
    pub fn new<V, S>(learning_rate: V, use_locking: bool, name: S) -> Self
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
        scope: &mut Scope,
        grads_and_vars: Vec<(Option<Tensor>, Tensor)>,
        global_step: Option<Variable>,
        name: S,
    ) -> Result<NodeIdent> {
        unimplemented!()
    }

    fn compute_gradients<Tx: TensorOps>(
        &mut self,
        scope: &mut Scope,
        loss: Tx,
        var_list: Vec<Tensor>,
        gate_gradients: GateGradients,
        aggregation_method: &str,
        colocate_gradients_with_ops: bool,
        grad_loss: Option<Tensor>,
    ) -> Result<Vec<(Option<Tensor>, Tensor)>> {
        unimplemented!()
    }

    fn get_name(&self) -> Option<&str> {
        unimplemented!()
    }

    fn get_slot<S: AsRef<Path>>(&self, var: Variable, name: S) -> Option<Variable> {
        unimplemented!()
    }

    fn get_slot_names(&self) -> Vec<&str> {
        unimplemented!()
    }

    fn valid_dtypes(&self) -> &[DataType] {
        unimplemented!()
    }

    fn create_slots<I>(&self, scope: &mut Scope, var_list: I) -> Result<()>
    where
        I: IntoIterator<Item = Variable>,
    {
        unimplemented!()
    }

    fn prepare(&self, scope: &mut Scope) -> Result<()> {
        unimplemented!()
    }

    fn apply_dense(&self, scope: &mut Scope, grad: &Tensor, var: &Tensor) -> Result<Tensor> {
        unimplemented!()
    }

    fn resource_apply_dense(
        &self,
        scope: &mut Scope,
        grad: &Tensor,
        handle: &Tensor,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    fn resource_apply_sparse_duplicate_indices(
        &self,
        scope: &mut Scope,
        grad: &Tensor,
        handle: &Tensor,
        indices: &Tensor,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    fn resource_apply_sparse(
        &self,
        scope: &mut Scope,
        grad: &Tensor,
        handle: &Tensor,
        indices: &Tensor,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    fn finish<S: AsRef<Path>>(
        &self,
        scope: &mut Scope,
        update_ops: Vec<Tensor>,
        name_scope: S,
    ) -> Result<NodeIdent> {
        unimplemented!()
    }

    fn slot_dict(&self, name: &str) -> Result<HashMap<Variable, Tensor>> {
        unimplemented!()
    }

    fn get_or_make_slot(
        &mut self,
        scope: &mut Scope,
        var: Variable,
        val: Tensor,
        slot_name: &str,
        op_name: &str,
    ) -> Result<Variable> {
        unimplemented!()
    }

    fn get_or_make_slot_with_initializer<Ti, Ts>(
        &mut self,
        scope: &mut Scope,
        var: Variable,
        initializer: Ti,
        shape: Ts,
        dtype: DataType,
        slot_name: &str,
        op_name: &str,
    ) -> Result<Variable>
    where
        Ti: TensorOps,
        Ts: TensorOps,
    {
        unimplemented!()
    }

    fn zeros_slot(&self, var: Variable, slot_name: &str, op_name: &str) -> Result<Variable> {
        unimplemented!()
    }
}

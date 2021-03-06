//! Gradient descent optimizer

use super::{math_ops, DataType, HashMap, NodeIdent, Optimizer, Result, Scope, ShapeOps, SlotDict,
            Tensor, TensorOps, Variable};
use framework::GetOp;
use ops::training_ops;
use ops::resource_variable_ops;

#[derive(Clone)]
/// Optimizer that implements the gradient descent algorithm.
pub struct GradientDescentOptimizer {
    learning_rate: Tensor,
    use_locking: bool,
    name: String,
    slots: SlotDict,
}

impl GradientDescentOptimizer {
    pub fn new<V>(
        context: &mut Scope,
        learning_rate: V,
        use_locking: bool,
        name: Option<String>,
    ) -> Self
    where
        V: TensorOps,
    {
        let name = if let Some(name) = name {
            name
        } else {
            "GradientDescent".to_owned()
        };
        let learning_rate = learning_rate.into_tensor(context);
        GradientDescentOptimizer {
            learning_rate,
            use_locking,
            name,
            slots: HashMap::new(),
        }
    }
}

impl Optimizer for GradientDescentOptimizer {
    impl_util_methods!();

    fn apply_dense(&self, context: &mut Scope, grad: &Tensor, var: &Variable) -> Result<Tensor> {
        let lr = math_ops::cast(context, self.learning_rate, var.dtype, "")?;
        training_ops::apply_gradient_descent(context, *var, lr, grad, self.use_locking, "")
    }

    fn resource_apply_dense(
        &self,
        context: &mut Scope,
        grad: &Tensor,
        handle: &Variable,
    ) -> Result<Tensor> {
        let lr = math_ops::cast(context, self.learning_rate, grad.dtype, "")?;
        training_ops::resource_apply_gradient_descent(
            context,
            *handle,
            lr,
            grad,
            self.use_locking,
            "",
        )
    }

    fn resource_apply_sparse_duplicate_indices(
        &self,
        context: &mut Scope,
        grad: &Tensor,
        handle: &Variable,
        indices: &Tensor,
    ) -> Result<NodeIdent> {
        Ok(*resource_variable_ops::resource_scatter_add(
            context,
            (*handle).into(),
            *indices,
            *grad,
            "",
        )?.get_op())
    }
}

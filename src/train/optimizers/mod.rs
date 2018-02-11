use std::collections::HashMap;
use std::path::Path;

use super::{DataType, Error, ErrorKind, NodeIdent, Result, Scope, Tensor, TensorOps, Variable};
use ops::{array_ops, control_flow_ops, math_ops, state_ops};
use ops::gradients_impl as gradients;

//mod gradient_descent;
//pub use self::gradient_descent::GradientDescentOptimizer;

pub trait Optimizer: Sized {
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
        scope: &mut Scope,
        loss: Tx,
        global_step: Option<Variable>,
        var_list: Vec<Tensor>,
        gate_gradients: GateGradients,
        aggregation_method: &str,
        colocate_gradients_with_ops: bool,
        name: S,
        grad_loss: Option<Tensor>,
    ) -> Result<NodeIdent> {
        let grads_and_vars = self.compute_gradients(
            scope,
            loss,
            var_list,
            gate_gradients,
            aggregation_method,
            colocate_gradients_with_ops,
            grad_loss,
        )?;

        if grads_and_vars.iter().all(|&(g, v)| g.is_none()) {
            return Err(Error::from(
                "No gradients provided for any variable, check your graph for ops
                that do not support gradients",
            ));
        }

        self.apply_gradients(scope, grads_and_vars, global_step, name)
    }

    /// Compute gradients of `loss` for the variables in `var_list`.
    ///
    /// This is the first part of `minimize()`. It returns a list of (gradient, variable) pairs
    /// where "gradient" is the gradient for "variable". Note that "gradient" can be a Tensor,
    ///  or None if there is no gradient for the given variable.
    ///
    /// ### Args:
    /// * loss: A Tensor containing the value to minimize.
    /// * var_list: Optional list or tuple of tf.Variable to update to minimize loss.
    /// * gate_gradients: How to gate the computation of gradients.
    /// * aggregation_method: Specifies the method used to combine gradient terms.
    ///                       Valid values are defined in the class AggregationMethod.
    /// * colocate_gradients_with_ops: If True, try colocating gradients with the corresponding op.
    /// * grad_loss: Optional. A Tensor holding the gradient computed for loss.
    ///
    /// ### Returns:
    /// * A list of (gradient, variable) pairs. Variable is always present, but gradient can be None.
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
        let loss = loss.into_tensor(scope);
        self.assert_valid_dtypes(&[loss])?;
        if let Some(grad_loss) = grad_loss {
            self.assert_valid_dtypes(&[grad_loss])?;
        }
        if var_list.is_empty() {
            return Err(Error::from("No variables to optimize."));
        }
        let processors: Result<Vec<_>> = var_list.iter().map(|v| get_processor(scope, v)).collect();
        let processors = processors?;
        let var_refs: Vec<_> = processors.iter().map(|p| p.target()).collect();
        let grad_loss = if let Some(grad_loss) = grad_loss {
            vec![grad_loss]
        } else {
            Vec::with_capacity(0)
        };
        let mut grads = gradients::gradients(
            scope,
            vec![loss],
            var_refs,
            Some(grad_loss),
            colocate_gradients_with_ops,
            gate_gradients.is_gate_op(),
            aggregation_method,
            None,
            "",
        )?;
        if let GateGradients::GateGraph = gate_gradients {
            let exist_grads: Vec<Tensor> = grads.iter().filter_map(|g| *g).collect();
            let mut grouped =
                control_flow_ops::tuple(scope, exist_grads, None as Option<&[Tensor]>, "")?;
            let mut new_grads = Vec::with_capacity(grads.len());
            for g in &grads {
                if g.is_none() {
                    new_grads.push(None);
                } else {
                    new_grads.push(Some(grouped.remove(0)));
                }
            }
            grads = new_grads;
        }
        let grads_and_vars: Vec<(_, Tensor)> = grads
            .into_iter()
            .zip(var_list.into_iter())
            .map(|(g, v)| (g, v.into()))
            .collect();
        self.assert_valid_dtypes(
            grads_and_vars
                .iter()
                .filter(|&&(g, v)| g.is_some() && !(DataType::Resource == v.dtype))
                .map(|&(_, ref v)| v),
        )?;
        Ok(grads_and_vars)
    }

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
        scope: &mut Scope,
        grads_and_vars: Vec<(Option<Tensor>, Tensor)>,
        global_step: Option<Variable>,
        name: S,
    ) -> Result<NodeIdent> {
        // This is a default implementation of apply_gradients() that can be shared
        // by most optimizers.  It relies on the subclass implementing the following
        // methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().

        if grads_and_vars.is_empty() {
            return Err(Error::from("No variables provided."));
        }
        let mut converted_grads_and_vars = vec![];
        for (g, v) in grads_and_vars {
            let p = get_processor(scope, &v)?;
            converted_grads_and_vars.push((g, v, p));
        }
        {
            let var_list: Vec<_> = converted_grads_and_vars
                .iter()
                .filter(|&&(g, _, _)| g.is_some())
                .map(|&(_, v, _)| v)
                .collect();
            if !var_list.is_empty() {
                return Err(Error::from(
                    "No gradients provided for any variable, check your graph for ops
                that do not support gradients",
                ));
            }
            let scope = &mut scope.clear_control_dependencies();
            let var_list: Result<Vec<_>> = var_list
                .iter()
                .map(|v| get_variable_for(scope, v))
                .collect();
            self.create_slots(scope, var_list?)?;
        }
        let mut update_ops = vec![];
        let scope = &mut scope.name_scope(name.as_ref().to_str().unwrap(), self.get_name());
        self.prepare(scope)?;
        for (grad, var, processor) in converted_grads_and_vars {
            if grad.is_none() {
                continue;
            }
            // We colocate all ops created in _apply_dense or _apply_sparse
            // on the same device as the variable.
            /* TODO:
            scope_name = var.op.name if context.in_graph_mode() else ""
            with ops.name_scope("update_" + scope_name), ops.colocate_with(var):
            */
            update_ops.push(processor.update_op(scope, self, grad.as_ref().unwrap())?)
        }

        let apply_updates;
        if let Some(global_step) = global_step {
            let f = self.finish(scope, update_ops, "update")?;
            let scope = &mut scope.control_dependencies(&[f]);
            // with ops.colocate_with(global_step):
            apply_updates = state_ops::assign_add(scope, global_step, 1_i32, false, name)?.into();
        } else {
            apply_updates = self.finish(scope, update_ops, name)?;
        }
        /* TODO:
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        if apply_updates not in train_op:
            train_op.append(apply_updates)
        */
        Ok(apply_updates)
    }

    fn get_name(&self) -> Option<&str>;

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

    fn assert_valid_dtypes<'a, I>(&self, tensors: I) -> Result<()>
    where
        I: IntoIterator<Item = &'a Tensor>,
    {
        let valid_dtypes = self.valid_dtypes();
        for tensor in tensors.into_iter() {
            if valid_dtypes
                .iter()
                .find(|dt| dt == &&tensor.dtype)
                .is_none()
            {
                return Err(Error::from(format!(
                    "Invalid type `{:?}`, expected one of: {:?}",
                    tensor.dtype, valid_dtypes,
                )));
            }
        }
        Ok(())
    }

    // --------------
    // Methods to be implemented by subclasses if they want to use the
    // inherited implementation of apply_gradients() or compute_gradients().
    // --------------

    /// Valid types for loss, variables and gradients.
    ///
    /// Subclasses should override to allow other float types.
    fn valid_dtypes(&self) -> &[DataType] {
        &[DataType::BFloat16, DataType::Float, DataType::Double]
    }

    /// Create all slots needed by the variables.
    fn create_slots<I>(&self, scope: &mut Scope, var_list: I) -> Result<()>
    where
        I: IntoIterator<Item = Variable>,
    {
        // No slots needed by default
        Ok(())
    }

    /// Create all needed tensors before applying gradients.
    ///
    /// This is called with the name_scope using the "name" that
    /// users have chosen for the application of gradients.
    fn prepare(&self, scope: &mut Scope) -> Result<()> {
        Ok(())
    }

    /// Add ops to apply dense gradients to `var`.
    fn apply_dense(&self, scope: &mut Scope, grad: &Tensor, var: &Tensor) -> Result<Tensor> {
        Err(ErrorKind::UnimplementedTraitMethod.into())
    }

    /// Add ops to apply dense gradients to the variable `handle`.
    fn resource_apply_dense(
        &self,
        scope: &mut Scope,
        grad: &Tensor,
        handle: &Tensor,
    ) -> Result<Tensor> {
        Err(ErrorKind::UnimplementedTraitMethod.into())
    }

    /// Add ops to apply sparse gradients to `handle`, with repeated indices.
    ///
    /// Optimizers which override this method must deal with repeated indices. See
    /// the docstring of `_apply_sparse_duplicate_indices` for details. By default
    /// the correct behavior, to sum non-unique indices and their associated
    /// gradients, is enforced by first pre-processing `grad` and `indices` and
    /// passing them on to `_resource_apply_sparse`. Optimizers which deal correctly
    /// with duplicate indices may instead override this method to avoid the
    /// overhead of summing.
    fn resource_apply_sparse_duplicate_indices(
        &self,
        scope: &mut Scope,
        grad: &Tensor,
        handle: &Tensor,
        indices: &Tensor,
    ) -> Result<Tensor> {
        let (summed_grad, unique_indices) = deduplicate_indexed_slices(scope, grad, indices)?;
        self.resource_apply_sparse(scope, &summed_grad, handle, indices)
    }

    /// Add ops to apply sparse gradients to the variable `handle`.
    ///
    /// Similar to `_apply_sparse`, the `indices` argument to this method has been
    /// de-duplicated. Optimizers which deal correctly with non-unique indices may
    /// instead override `_resource_apply_sparse_duplicate_indices` to avoid this
    /// overhead.
    fn resource_apply_sparse(
        &self,
        scope: &mut Scope,
        grad: &Tensor,
        handle: &Tensor,
        indices: &Tensor,
    ) -> Result<Tensor> {
        Err(ErrorKind::UnimplementedTraitMethod.into())
    }

    /*
    /// Add ops to apply sparse gradients to `var`, with repeated sparse indices.
    ///
    /// Optimizers which override this method must deal with IndexedSlices objects
    /// such as the following:
    ///
    ///   `IndexedSlicesValue(values=[1, 1], indices=[0, 0], dense_shape=[1])`
    ///
    /// The correct interpretation is:
    ///
    ///   `IndexedSlicesValue(values=[2], indices=[0], dense_shape=[1])`
    ///
    /// Many optimizers deal incorrectly with repeated indices when updating based
    /// on sparse gradients (e.g. summing squares rather than squaring the sum, or
    /// applying momentum terms multiple times). Adding first is always the correct
    /// behavior, so this is enforced here by reconstructing the IndexedSlices to
    /// have only unique indices, then calling _apply_sparse.
    ///
    /// Optimizers which deal correctly with repeated indices may instead override
    /// this method to avoid the overhead of summing indices.
    fn apply_sparse_duplicate_indices(
        &self,
        scope: &mut Scope,
        grad: Tensor,
        var: Variable,
    ) -> Result<()> {}

    /// Add ops to apply sparse gradients to `var`.
    ///
    /// The IndexedSlices object passed to `grad` in this function is by default
    /// pre-processed in `_apply_sparse_duplicate_indices` to remove duplicate
    /// indices (see its docstring for details). Optimizers which can tolerate or
    /// have correct special cases for duplicate sparse indices may override
    /// `_apply_sparse_duplicate_indices` instead of this function, avoiding that
    /// overhead.
    fn apply_sparse(&self, scope: &mut Scope, grad: Tensor, var: Variable) -> Result<Tensor> {}
    */

    /// Do what is needed to finish the update.
    ///
    /// This is called with the `name_scope` using the "name" that
    /// users have chosen for the application of gradients.
    fn finish<S: AsRef<Path>>(
        &self,
        scope: &mut Scope,
        update_ops: Vec<Tensor>,
        name_scope: S,
    ) -> Result<NodeIdent> {
        Ok(control_flow_ops::Group::new(scope, &update_ops, name_scope)?.into())
    }

    // --------------
    // Utility methods.
    // --------------

    /// Returns a dict for caching slots created under the given name.
    fn slot_dict(&self, name: &str) -> Result<HashMap<Variable, Tensor>>;

    /// Find or create a slot for a variable.
    fn get_or_make_slot(
        &mut self,
        scope: &mut Scope,
        var: Variable,
        val: Tensor,
        slot_name: &str,
        op_name: &str,
    ) -> Result<Variable>;

    /// Find or create a slot for a variable, using an Initializer.
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
        Ts: TensorOps;

    /// Find or create a slot initialized with 0.0.
    fn zeros_slot(&self, var: Variable, slot_name: &str, op_name: &str) -> Result<Variable>;
}

pub enum GateGradients {
    GateNone,
    GateOp,
    GateGraph,
}

impl GateGradients {
    fn is_gate_op(&self) -> bool {
        match *self {
            GateGradients::GateOp => true,
            _ => false,
        }
    }
}

enum Processor {
    RefVariable(Tensor),
    DenseResourceVariable(Tensor),
    //DenseReadResourceVariableProcessor,
    StreamingModelPort(Tensor),
}

impl Processor {
    /// Returns the update ops for updating the variable.
    fn update_op<O: Optimizer>(
        &self,
        scope: &mut Scope,
        optimizer: &O,
        grad: &Tensor,
    ) -> Result<Tensor> {
        match *self {
            Processor::RefVariable(ref v) => {
                let update_op = optimizer.apply_dense(scope, grad, v)?;
                /*
                if let Some(func) = v.constraint() {
                    let scope = &mut scope.control_dependencies(&[update_op]);
                    assign(scope, v, func(v))
                } else {
                    Ok(update_op)
                }
                */
                Ok(update_op)
            }
            Processor::DenseResourceVariable(ref v) => {
                optimizer.resource_apply_dense(scope, grad, v)
            }
            Processor::StreamingModelPort(ref v) => Ok(*grad),
        }
    }

    fn target(&self) -> Tensor {
        match *self {
            Processor::RefVariable(v) => v,
            Processor::DenseResourceVariable(v) => v,
            Processor::StreamingModelPort(v) => v,
        }
    }
}

fn get_processor(scope: &mut Scope, v: &Tensor) -> Result<Processor> {
    if v.op_type(scope) == "VarHandleOp" {
        Ok(Processor::DenseResourceVariable(*v))
    } else if v.is_ref() {
        Ok(Processor::RefVariable(*v))
    } else if v.op_type(scope) == "SubmodelPort" {
        Ok(Processor::StreamingModelPort(*v))
    } else {
        Err(Error::from("Trying to optimize unsupported type"))
    }
}

/// Returns the ResourceVariable responsible for v, or v if not necessary.
fn get_variable_for(scope: &mut Scope, v: &Tensor) -> Result<Variable> {
    if v.op_type(scope) == "VarHandleOp" {
        panic!("VarHandleOp not supported yet for optimizers")
    } else {
        Variable::from_tensor(scope, v)
    }
}

/// Sums `values` associated with any non-unique `indices`.
fn deduplicate_indexed_slices(
    scope: &mut Scope,
    values: &Tensor,
    indices: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (unique_indices, new_index_positions) = array_ops::unique(scope, indices, None, "")?;
    let s = array_ops::shape(scope, unique_indices, None, "")?;
    let s = array_ops::slice(scope, s, 0, 1, "")?;
    let summed_values = math_ops::unsorted_segment_sum(scope, values, new_index_positions, s, "")?;
    Ok((summed_values, unique_indices))
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

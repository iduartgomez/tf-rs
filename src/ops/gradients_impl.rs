use super::*;

use std::collections::{HashMap, HashSet, VecDeque};

///  Constructs symbolic derivatives of sum of `ys` w.r.t. x in `xs`.
///
///  `ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys`
///  is a list of `Tensor`, holding the gradients received by the
///  `ys`. The list must be the same length as `ys`.
///
///  `gradients()` adds ops to the graph to output the derivatives of `ys` with
///  respect to `xs`.  It returns a list of `Tensor` of length `len(xs)` where
///  each tensor is the `sum(dy/dx)` for y in `ys`.
///
///  `grad_ys` is a list of tensors of the same length as `ys` that holds
///  the initial gradients for each y in `ys`.  When `grad_ys` is None,
///  we fill in a tensor of '1's of the shape of y for each y in `ys`.  A
///  user can provide their own initial `grad_ys` to compute the
///  derivatives using a different initial gradient for each y (e.g., if
///  one wanted to weight the gradient differently for each value in
///  each y).
///
///  `stop_gradients` is a `Tensor` or a list of tensors to be considered constant
///  with respect to all `xs`. These tensors will not be backpropagated through,
///  as though they had been explicitly disconnected using `stop_gradient`.  Among
///  other things, this allows computation of partial derivatives as opposed to
///  total derivatives. For example:
///
/// ```python
///    a = tf.constant(0.)
///    b = 2 * a
///    g = tf.gradients(a + b, [a, b], stop_gradients=[a, b])
/// ```
///
///  Here the partial derivatives `g` evaluate to `[1.0, 1.0]`, compared to the
///  total derivatives `tf.gradients(a + b, [a, b])`, which take into account the
///  influence of `a` on `b` and evaluate to `[3.0, 1.0]`.  Note that the above is
///  equivalent to:
///
/// ```python
///    a = tf.stop_gradient(tf.constant(0.))
///    b = tf.stop_gradient(2 * a)
///    g = tf.gradients(a + b, [a, b])
/// ```
///
///  `stop_gradients` provides a way of stopping gradient after the graph has
///  already been constructed, as compared to `tf.stop_gradient` which is used
///  during graph construction.  When the two approaches are combined,
///  backpropagation stops at both `tf.stop_gradient` nodes and nodes in
///  `stop_gradients`, whichever is encountered first.
///
///  ### Args:
///    * ys: A list of tensors to be differentiated.
///    * xs: A list of tensors to be used for differentiation.
///    * grad_ys: Optional. A list of tensors the same size as
///      `ys` and holding the gradients computed for each y in `ys`.
///    * colocate_gradients_with_ops: If true, try colocating gradients with
///      the corresponding op.
///    * gate_gradients: If true, add a tuple around the gradients returned
///      for an operations.  This avoids some race conditions.
///    * aggregation_method: Specifies the method used to combine gradient terms.
///      Accepted values are constants defined in the class `AggregationMethod`.
///    * stop_gradients: Optional. A list of tensors not to differentiate
///      through.
///    * name: Optional name to use for grouping all the gradient ops together.
///      defaults to 'gradients'.
///
///  ### Returns:
///    A list of `sum(dy/dx)` for each x in `xs`.
#[doc(hidden)]
pub fn gradients<Tys, Txs, S>(
    scope: &mut Scope,
    ys: Vec<Tys>,
    xs: Vec<Txs>,
    grad_ys: Option<Vec<Tensor>>,
    colocate_gradients_with_ops: bool,
    gate_gradients: bool,
    aggregation_method: Option<AggregationMethod>,
    stop_gradients: Option<Vec<Tensor>>,
    name: S,
) -> Result<Vec<Option<Vec<Tensor>>>>
where
    Txs: TensorOps,
    Tys: TensorOps,
    S: AsRef<Path>,
{
    let stop_gradients = if let Some(sg) = stop_gradients {
        sg
    } else {
        Vec::new()
    };
    let grad_ys = if let Some(grads) = grad_ys {
        if grads.len() != ys.len() {
            return Err(Error::from(
                "tf-rs: `grad_ys` must be of same length as `ys` in `gradients` function call",
            ));
        }
        grads.into_iter().map(|x| Some(x)).collect()
    } else {
        vec![None; ys.len()]
    };

    let scope = &mut scope.name_scope(name.as_ref().to_str().unwrap(), Some("gradients"));
    let grad_scope = scope.name().to_owned();
    let mut ys = ys.into_iter()
        .map(|x| x.into_tensor(scope))
        .collect::<Vec<_>>();
    let xs = xs.into_iter()
        .map(|x| x.into_tensor(scope))
        .collect::<Vec<_>>();
    let grad_ys = default_grad_ys(grad_ys, &ys, colocate_gradients_with_ops)?;

    // The approach we take here is as follows: Create a list of all ops in the
    // subgraph between the ys and xs.  Visit these ops in reverse order of ids
    // to ensure that when we visit an op the gradients w.r.t its outputs have
    // been collected.  Then aggregate these gradients if needed, call the op's
    // gradient function, and add the generated gradients to the gradients for
    // its input.

    // Initialize the pending count for ops in the connected subgraph from ys
    // to the xs.
    if ys.len() > 1 {
        ys = ys.into_iter()
            .map(|y| {
                if !y.consumers(scope)?.is_empty() {
                    scope.identity(y, "")
                } else {
                    Ok(y)
                }
            })
            .collect::<Result<Vec<_>>>()?;
    }
    //let from_ops = &xs;
    let to_ops = &ys;
    let (pending_count, mut loop_state) =
        pending_count(scope, to_ops, &xs, colocate_gradients_with_ops)?;

    // Iterate over the collected ops.
    //
    // grads: op => list of gradients received on each output endpoint of the
    // op.  The gradients for each endpoint are initially collected as a list.
    // When it is time to call the op's gradient function, for each endpoint we
    // aggregate the list of received gradients into a Add() Operation if there
    // is more than one.
    let mut grads: HashMap<NodeIdent, Vec<Vec<Tensor>>> = HashMap::new();

    // Add the initial gradients for the ys.
    for (y, grad_y) in ys.iter().zip(grad_ys.iter()) {
        set_grad(scope, &mut grads, y, grad_y)?;
    }

    // Initialize queue with to_ops.
    let mut queue = VecDeque::new();
    // Add the ops in 'to_ops' into the queue.
    let mut to_ops_set = HashSet::new();
    for op in to_ops {
        // 'ready' handles the case where one output gradient relies on
        // another output's gradient.
        let id = op.get_ident();
        let ready = pending_count[&id] == 0;
        if ready && to_ops_set.contains(&id) {
            to_ops_set.insert(id);
            queue.push_back(id);
        }
    }

    if let Some(ref loop_state) = loop_state {
        let loop_exits = loop_state.process_unused_loop_exits(&pending_count, &to_ops_set)?;
        for y in &loop_exits {
            if is_trainable(scope, y) {
                set_grad(scope, &mut grads, y, &loop_state.zeros_like_for_exit(y)?)?;
                queue.push_back(y.get_ident());
            }
        }
    }

    let stop_ops = stop_ops(scope, &xs, &stop_gradients, &pending_count)?;
    while let Some(ref op) = queue.pop_front() {
        // TODO: with _maybe_colocate_with(op, colocate_gradients_with_ops):
        if let Some(ref loop_state) = loop_state {
            loop_state.enter_grad_while_context(op, true);
        }
        let mut out_grads = aggregated_grads(&grads, op, &loop_state, &aggregation_method)?;
        if let Some(ref loop_state) = loop_state {
            loop_state.exit_grad_while_context(op, true);
        }

        let mut grad_fn = None;
        let mut func_call = None;
        let is_func_call = scope.is_function(op);
        let has_out_grads = !out_grads.is_empty();
        if has_out_grads && !stop_ops.contains(op) {
            if is_func_call {
                let func = scope.get_function(op)?;
                grad_fn = Some(func.get_gradient_func()
                    .ok_or(Error::from(ErrorKind::UndefinedGrad))?);
                func_call = Some(func);
            } else {
                // A grad_fn must be defined, either as a function or as None
                // for ops that do not have gradients.
                grad_fn = Some(scope
                    .get_gradient_function(op)
                    .ok_or(Error::from(ErrorKind::UndefinedGrad))?);
            }
        }
        if let Some(ref loop_state) = loop_state {
            loop_state.enter_grad_while_context(op, false);
        }
        let mut in_grads;
        if (grad_fn.is_some() || is_func_call) && has_out_grads {
            // NOTE: If _AggregatedGrads didn't compute a value for the i'th
            // output, it means that the cost does not depend on output[i],
            // therefore dC/doutput[i] is 0.
            for (i, out_grad) in out_grads.iter_mut().enumerate() {
                if out_grad.is_none()
                    && ((grad_fn.is_none() && is_func_call)
                        || is_trainable(scope, &(op.get_outputs(scope)?[i])))
                {
                    // Only floating-point outputs get a zero gradient. Gradient
                    // functions should ignore the gradient for other outputs.
                    if let Some(ref loop_state) = loop_state {
                        *out_grad = Some(loop_state.zeros_like(op, i)?);
                    } else {
                        *out_grad = Some(control_flow_ops::zeros_like_outside_loop(scope, op, i)?);
                    }
                }
            }
            let name = op.get_name(scope);
            let scope = &mut scope.name_scope(name + "_grad", None);
            if let Some(grad_fn) = grad_fn {
                // If grad_fn was found, do not use SymbolicGradient even for
                // functions.
                in_grads = maybe_compile(scope, &grad_scope, op, func_call, grad_fn, &out_grads)?;
            } else {
                // For function call ops, we add a 'SymbolicGradient'
                // node to the graph to compute gradients.
                in_grads = maybe_compile(
                    scope,
                    &grad_scope,
                    op,
                    func_call,
                    Box::new(sym_grad),
                    &out_grads,
                )?;
            }
            // _VerifyGeneratedGradients(in_grads, op)
            if gate_gradients && in_grads.iter().any(|x| x.is_some()) {
                let mut some_in_grads = in_grads.into_iter().filter_map(|x| x).collect();
                some_in_grads =
                    control_flow_ops::tuple(scope, some_in_grads, None as Option<&[Tensor]>, "")?;
                in_grads = some_in_grads.into_iter().map(|x| Some(x)).collect();
            }
        // TODO: _LogOpGradients(op, out_grads, in_grads)
        } else {
            // If no grad_fn is defined or none of out_grads is available,
            // just propagate a list of None backwards.
            in_grads = vec![None; op.get_inputs(scope)?.len()];
        }
        for (i, (t_in, mut in_grad)) in op.get_inputs(scope)?
            .into_iter()
            .zip(in_grads.into_iter())
            .enumerate()
        {
            if let Some(mut in_grad_t) = in_grad {
                if t_in.dtype != DataType::Resource {
                    let shape = t_in.get_shape(scope)
                        .definition_i64()
                        .ok_or(Error::from(ErrorKind::UndefinedTensorShape))?;
                    in_grad_t = in_grad_t.set_shape(scope, shape.as_slice())?;
                }
                set_grad(scope, &mut grads, &t_in, &in_grad_t)?;
            }
        }
        if let Some(ref loop_state) = loop_state {
            loop_state.exit_grad_while_context(op, false);
        }

        // Update pending count for the inputs of op and enqueue ready ops.
        update_pending_and_enqueue_ready(&grads, op, &mut queue, &pending_count, &loop_state)?;
    }

    if let Some(ref mut loop_state) = loop_state {
        loop_state.post_processing();
    }
    xs.into_iter().map(|x| get_grad(scope, &grads, x)).collect()
}

fn default_grad_ys(
    grads: Vec<Option<Tensor>>,
    ys: &[Tensor],
    colocate_gradients_with_ops: bool,
) -> Result<Vec<Tensor>> {
    unimplemented!()
}

/// Sets gradient `grad` in `grads` for tensor `t`.
fn set_grad(
    scope: &Scope,
    grads: &mut HashMap<NodeIdent, Vec<Vec<Tensor>>>,
    t: &Tensor,
    grad: &Tensor,
) -> Result<()> {
    let op = t.get_op(scope);
    let op_grads;
    if !grads.contains_key(&op) {
        grads.insert(
            op,
            (0..op.get_outputs(scope)?.len())
                .into_iter()
                .map(|_| Vec::<Tensor>::new())
                .collect::<Vec<_>>(),
        );
        op_grads = grads.get_mut(&op).unwrap();
    } else {
        op_grads = grads.get_mut(&op).unwrap();
    }
    let t_grads = &mut op_grads[t.idx as usize];
    t_grads.push(*grad);
    /* TODO:
    if isinstance(t_grads, list):
        t_grads.append(grad)
    else:
        assert control_flow_util.IsLoopSwitch(op)
        op_grads[t.value_index] = grad
    */
    Ok(())
}

/// Gets gradient for tensor `t`.
fn get_grad(
    scope: &Scope,
    grads: &HashMap<NodeIdent, Vec<Vec<Tensor>>>,
    t: Tensor,
) -> Result<Option<Vec<Tensor>>> {
    let op = t.get_op(scope);
    if let Some(op_grads) = grads.get(&op) {
        Ok(Some(op_grads[t.idx as usize].clone()))
    } else {
        Ok(None)
    }
}

fn pending_count(
    scope: &mut Scope,
    to_ops: &[Tensor],
    from_ops: &[Tensor],
    colocate_gradients_with_ops: bool,
) -> Result<(HashMap<NodeIdent, usize>, Option<ControlFlowState>)> {
    unimplemented!()
}

fn stop_ops(
    scope: &mut Scope,
    from_ops: &[Tensor],
    stop_gradients: &[Tensor],
    pending_count: &HashMap<NodeIdent, usize>,
) -> Result<HashSet<NodeIdent>> {
    unimplemented!()
}

fn is_trainable(scope: &Scope, tensor: &Tensor) -> bool {
    unimplemented!()
}

fn aggregated_grads(
    grads: &HashMap<NodeIdent, Vec<Vec<Tensor>>>,
    op: &NodeIdent,
    loop_state: &Option<ControlFlowState>,
    aggregation_method: &Option<AggregationMethod>,
) -> Result<Vec<Option<Tensor>>> {
    unimplemented!()
}

fn sym_grad(scope: &mut Scope, out_grads: &[Option<Tensor>]) -> Result<Vec<Option<Tensor>>> {
    unimplemented!()
}

fn maybe_compile(
    context: &mut Scope,
    scope: &str,
    op: &NodeIdent,
    func_call: Option<Function>,
    mut grad_fn: GradFunc,
    inputs: &[Option<Tensor>],
) -> Result<Vec<Option<Tensor>>> {
    let scope = scope.trim_right_matches("/").replace("/", "_");
    let xla_compile;
    let xla_separate_compiled_gradients;
    let xla_scope;
    if let Some(func) = func_call {
        xla_compile = func.get_attr("_XlaCompile")
            .ok_or(Error::from(ErrorKind::UndefinedFuncAttr))?
            .unwrap_b();
        xla_separate_compiled_gradients = func.get_attr("_XlaSeparateCompiledGradients")
            .ok_or(Error::from(ErrorKind::UndefinedFuncAttr))?
            .unwrap_b();;
        xla_scope = func.get_attr("_XlaScope")
            .ok_or(Error::from(ErrorKind::UndefinedFuncAttr))?
            .unwrap_s();
    } else {
        let xla_compile_2 = op.get_attr(context, "_XlaCompile");
        let xla_separate_compiled_gradients_2 =
            op.get_attr(context, "_XlaSeparateCompiledGradients");
        let xla_scope_2 = op.get_attr(context, "__XlaScope");
        if xla_compile_2.is_none() | xla_separate_compiled_gradients_2.is_none()
            | xla_scope_2.is_none()
        {
            return grad_fn(context, inputs);
        } else {
            xla_compile = xla_compile_2.unwrap().unwrap_b();
            xla_separate_compiled_gradients = xla_separate_compiled_gradients_2.unwrap().unwrap_b();
            xla_scope = xla_scope_2.unwrap().unwrap_s();
        }
    }

    if !xla_compile {
        return grad_fn(context, inputs);
    }

    // If the gradients are supposed to be compiled separately, we give them a
    // _XlaScope name that is based on the name_scope of the gradients.  Otherwise
    // they just inherit the existing _XlaScope name, which lets them be merged
    // together with the non-gradient computation.
    let xla_grad_scope;
    if xla_separate_compiled_gradients {
        xla_grad_scope = format!("{}_grad_{}", xla_scope, scope);
    } else {
        xla_grad_scope = xla_scope;
    }

    /* FIXME:
        attrs = {
            "_XlaCompile": attr_value_pb2.AttrValue(b=xla_compile),
            "_XlaScope": attr_value_pb2.AttrValue(s=xla_grad_scope.encode())
        }
        with ops.get_default_graph()._attr_scope(attrs):
    */
    grad_fn(context, inputs)
}

fn update_pending_and_enqueue_ready(
    grads: &HashMap<NodeIdent, Vec<Vec<Tensor>>>,
    op: &NodeIdent,
    queue: &mut VecDeque<NodeIdent>,
    pending_count: &HashMap<NodeIdent, usize>,
    loop_state: &Option<ControlFlowState>,
) -> Result<()> {
    unimplemented!()
}

struct ControlFlowState;

impl ControlFlowState {
    fn process_unused_loop_exits(
        &self,
        pending_count: &HashMap<NodeIdent, usize>,
        to_ops_set: &HashSet<NodeIdent>,
    ) -> Result<HashSet<Tensor>> {
        unimplemented!()
    }

    fn zeros_like_for_exit(&self, val: &Tensor) -> Result<Tensor> {
        unimplemented!()
    }

    fn zeros_like(&self, op: &NodeIdent, index: usize) -> Result<Tensor> {
        unimplemented!()
    }

    fn enter_grad_while_context(&self, op: &NodeIdent, before: bool) {
        unimplemented!()
    }

    fn exit_grad_while_context(&self, op: &NodeIdent, before: bool) {
        unimplemented!()
    }

    fn post_processing(&mut self) {
        unimplemented!()
    }
}

pub enum AggregationMethod {
    /// All of the gradient terms are summed as part of one
    /// operation using the "AddN" op. It has the property that all
    /// gradients must be ready before any aggregation is performed.
    AddN,
    /// Experimental and may not be supported in future releases.
    ExperimentalTree,
    /// Experimental and may not be supported in future releases.
    ExperimentalAccumulateN,
}

impl Default for AggregationMethod {
    fn default() -> AggregationMethod {
        AggregationMethod::AddN
    }
}

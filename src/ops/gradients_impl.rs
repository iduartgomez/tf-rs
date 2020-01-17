use std::collections::{HashMap, HashSet, VecDeque};

use TensorShape;
use framework::attr_value_pb::NameAttrList;
use super::{array_ops, control_flow_ops, math_ops, DTypeOps, DataType, Error, ErrorKind, Function,
            GetOp, GradFunc, NodeIdent, Path, Result, Scope, ShapeOps, Tensor, TensorOps};

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
#[allow(clippy::cognitive_complexity)]
pub fn gradients<Tys, Txs, S>(
    context: &mut Scope,
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
        grads.into_iter().map(Some).collect()
    } else {
        vec![None; ys.len()]
    };

    let scope = &mut context.name_scope(name.as_ref().to_str().unwrap(), Some("gradients"));
    let grad_scope = scope.name().to_owned();
    let mut ys = ys.into_iter()
        .map(|x| x.into_tensor(scope))
        .collect::<Vec<_>>();
    let xs = xs.into_iter()
        .map(|x| x.into_tensor(scope))
        .collect::<Vec<_>>();
    let grad_ys = default_grad_ys(scope, grad_ys, &ys, colocate_gradients_with_ops)?;

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
    let to_ops: Vec<_> = ys.iter().map(|x| x.get_op()).collect();
    let (pending_count, mut loop_state) = {
        let from_ops = xs.iter().map(|x| x.get_op());
        pending_count(scope, &to_ops, from_ops, colocate_gradients_with_ops)?
    };

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
    for id in to_ops {
        // 'ready' handles the case where one output gradient relies on
        // another output's gradient.
        let ready = pending_count[id] == 0;
        if ready && to_ops_set.contains(id) {
            to_ops_set.insert(*id);
            queue.push_back(*id);
        }
    }

    if let Some(ref loop_state) = loop_state {
        let loop_exits = loop_state.process_unused_loop_exits(&pending_count, &to_ops_set)?;
        for y in &loop_exits {
            if is_trainable(y) {
                set_grad(scope, &mut grads, y, &loop_state.zeros_like_for_exit(y)?)?;
                queue.push_back(*y.get_op());
            }
        }
    }

    let stop_ops = {
        let from_ops = xs.iter().map(|x| x.get_op());
        stop_ops(scope, from_ops, &stop_gradients, &pending_count)?
    };
    while let Some(ref op) = queue.pop_front() {
        // TODO: with _maybe_colocate_with(op, colocate_gradients_with_ops):
        if let Some(ref loop_state) = loop_state {
            loop_state.enter_grad_while_context(op, true);
        }
        let mut out_grads = aggregated_grads(scope, &grads, op, &loop_state, aggregation_method)?;
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
                    .ok_or_else(|| Error::from(ErrorKind::UndefinedGrad))?);
                func_call = Some(func);
            } else {
                // A grad_fn must be defined, either as a function or as None
                // for ops that do not have gradients.
                grad_fn = Some(scope
                    .get_gradient_function(op)
                    .ok_or_else(|| Error::from(ErrorKind::UndefinedGrad))?);
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
                        || is_trainable(&(op.get_outputs(scope)?[i])))
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
            let name = op.get_name(scope)?;
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
                in_grads = some_in_grads.into_iter().map(Some).collect();
            }
        // TODO: _LogOpGradients(op, out_grads, in_grads)
        } else {
            // If no grad_fn is defined or none of out_grads is available,
            // just propagate a list of None backwards.
            in_grads = vec![None; op.get_inputs(scope)?.len()];
        }
        for (i, (t_in, in_grad)) in op.get_inputs(scope)?
            .into_iter()
            .zip(in_grads.into_iter())
            .enumerate()
        {
            if let Some(mut in_grad_t) = in_grad {
                if t_in.dtype != DataType::Resource {
                    let shape = t_in.get_shape(scope)
                        .definition_i64()
                        .ok_or_else(|| Error::from(ErrorKind::UndefinedTensorShape))?;
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
    xs.into_iter()
        .map(|x| get_grad(scope, &mut grads, x))
        .collect()
}

///  Fill in default values for grad_ys.
///
///  ### Args:
///    * grad_ys: List of gradients, can contain None.
///    * ys: List of tensors.
///    * colocate_gradients_with_ops: If True, try colocating gradients with
///      the corresponding op.
///
///  ### Returns:
///    A list of gradients to use, without None.
fn default_grad_ys(
    scope: &mut Scope,
    grad_ys: Vec<Option<Tensor>>,
    ys: &[Tensor],
    colocate_gradients_with_ops: bool,
) -> Result<Vec<Tensor>> {
    if grad_ys.len() != ys.len() {
        return Err(Error::from(format!(
            "Passed {} grad_ys for {} ys",
            grad_ys.len(),
            ys.len()
        )));
    }
    let mut new_grad_ys = vec![];
    for i in 0..grad_ys.len() {
        let grad_y = &grad_ys[i];
        let y = &ys[i];
        // TODO: with _maybe_colocate_with(y.op, colocate_gradients_with_ops):
        if grad_y.is_none() {
            if y.dtype.is_complex() {
                return Err(Error::from(format!(
                    "Gradients of complex tensors must set grad_ys (y.dtype = {:?})",
                    y.dtype
                )));
            }
            let g = {
                let s = array_ops::shape(scope, y, None, "")?;
                // ($context:ident; $dtype:expr; $val:expr; $shape:expr; $name:expr)
                let c =
                    dtype_to_const!(scope; y.dtype; &[1]; &[] as &[i32]; format!("grad_ys_{}", i))?;
                array_ops::fill(scope, s, c, "")?
            };
            new_grad_ys.push(g);
            continue;
        }
        let grad_y = grad_y.unwrap();
        if y.dtype.is_floating() || y.dtype.is_integer() {
            if !grad_y.dtype.is_floating() || !grad_y.dtype.is_integer() {
                return Err(Error::from(format!(
                    "Gradient type {:?} generated for real or 
                    integer-valued tensor {:?} with type {:?} must be 
                    real or integer",
                    grad_y.dtype, y, y.dtype
                )));
            }
        } else if y.dtype.is_complex() {
            if !grad_y.dtype.is_complex() {
                return Err(Error::from(format!(
                    "Gradient type {:?} generated for complex-valued
                    tensor {:?} with type {:?} must be real",
                    grad_y.dtype, y, y.dtype
                )));
            }
        } else {
            return Err(Error::from(format!(
                "Tensor {:?} with type {:?} must be numeric
                to obtain a default gradient",
                y, y.dtype
            )));
        }
        // Create a grad_y tensor in the name scope of the gradient.
        // Required for TensorArrays to identify which gradient call a
        // grad_y value is coming from.
        new_grad_ys.push(scope.identity(grad_y, format!("grad_ys_{}", i))?)
    }
    Ok(new_grad_ys)
}

/// Sets gradient `grad` in `grads` for tensor `t`.
fn set_grad(
    scope: &Scope,
    grads: &mut HashMap<NodeIdent, Vec<Vec<Tensor>>>,
    t: &Tensor,
    grad: &Tensor,
) -> Result<()> {
    let op = *t.get_op();
    let op_grads = grads.entry(op).or_insert(
        (0..op.get_outputs(scope)?.len())
                .map(|_| Vec::<Tensor>::new())
                .collect::<Vec<_>>()
    );
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
    grads: &mut HashMap<NodeIdent, Vec<Vec<Tensor>>>,
    t: Tensor,
) -> Result<Option<Vec<Tensor>>> {
    let op = t.get_op();
    if let Some(mut op_grads) = grads.remove(&op) {
        Ok(Some(op_grads.remove(t.idx as usize)))
    } else {
        Ok(None)
    }
}

/// Gets all gradients for op.
fn get_grads(
    scope: &Scope,
    grads: &HashMap<NodeIdent, Vec<Vec<Tensor>>>,
    op: &NodeIdent,
) -> Result<Vec<Vec<Tensor>>> {
    use std::iter::FromIterator;
    if let Some(grads) = grads.get(op) {
        Ok(grads.clone())
    } else {
        Ok(Vec::from_iter(
            (0..op.get_outputs(scope)?.len()).map(|_| vec![]),
        ))
    }
}

///  Initialize the pending count for ops between two lists of Operations.
///
///  'pending_count[op._id]' indicates the number of backprop inputs
///  to this operation.
///
///  ### Args:
///    * to_ops: list of Operations.
///    * from_ops: list of Operations.
///    * colocate_gradients_with_ops: Python bool.  See docstring of gradients().
///
///  ### Returns:
///    A tuple containing: (1) a list of integers indexed by operation id,
///    indicating the number of backprop inputs to this operation, and (2)
///    a ControlFlowState object which is not None if the ops between from_ops
///    and to_ops contain control flow loops.
fn pending_count<'a, I: 'a>(
    scope: &mut Scope,
    to_ops: &[&NodeIdent],
    from_ops: I,
    colocate_gradients_with_ops: bool,
) -> Result<(HashMap<NodeIdent, usize>, Option<ControlFlowState>)>
where
    I: Iterator<Item = &'a NodeIdent>,
{
    let mut queue = VecDeque::new();
    // Mark reachable ops from from_ops.
    let mut reached_ops = HashMap::new();
    for op in to_ops {
        reached_ops.insert(**op, true);
        queue.push_back(**op);
    }
    mark_reached_ops(scope, from_ops, &mut reached_ops)?;

    // Mark between ops.
    let mut between_ops = HashMap::new();
    let mut between_op_list = Vec::new();
    while let Some(op) = queue.pop_front() {
        // We are interested in this op.
        if let Some(val) = reached_ops.get_mut(&op) {
            between_ops.insert(op, true);
            between_op_list.push(op);
            // Clear the boolean so we won't add the inputs again.
            *val = false;
            for inp in op.get_inputs(scope)? {
                queue.push_back(*inp.get_op());
            }
        }
    }

    // 'loop_state' is None if there are no while loops.
    let loop_state = ControlFlowState::maybe_create_control_flow_state(
        scope,
        &between_op_list,
        &between_ops,
        colocate_gradients_with_ops,
    )?;

    // Initialize pending count for between ops.
    let mut pending_count = HashMap::new();
    for op in between_op_list {
        for x in op.get_inputs(scope)? {
            if between_ops[x.get_op()] {
                let val = pending_count.entry(*x.get_op()).or_insert(0);
                *val += 1
            }
        }
    }
    Ok((pending_count, loop_state))
}

/// Mark all ops reached from "from_ops".
///
/// ### Args:
///   * from_ops: list of Operations.
///   * reached_ops: list of booleans, indexed by operation id.
fn mark_reached_ops<'a, I: 'a>(
    scope: &Scope,
    from_ops: I,
    reached_ops: &mut HashMap<NodeIdent, bool>,
) -> Result<()>
where
    I: Iterator<Item = &'a NodeIdent>,
{
    let mut queue = VecDeque::<NodeIdent>::new();
    queue.extend(from_ops);
    while let Some(ref op) = queue.pop_front() {
        let val = reached_ops.entry(*op).or_insert(true);
        *val = true;
        for output in op.get_outputs(scope)? {
            queue.extend(output.consumers(scope)?);
        }
    }
    Ok(())
}

///  The set of ops that terminate the gradient computation.
///
///  This computes the frontier of the forward graph *before* which backprop
///  should stop. Operations in the returned set will not be differentiated.
///  This set is defined as the subset of `from_ops` containing ops that have
///  no predecessor in `from_ops`. `pending_count` is the result of
///  `_PendingCount(g, xs, from_ops)`. An 'op' has predecessors in `from_ops`
///  iff pending_count[op._id] > 0.
///
///  In addition, none of `stop_gradient_ops` will be differentiated.
///
///  ### Args:
///    * from_ops: list of Operations.
///    * stop_gradient_ops: list of Operations never to backprop through.
///    * pending_count: List of integers, indexed by operation id.
///
///  ### Returns:
///    The set of operations.
fn stop_ops<'a, I: 'a>(
    scope: &mut Scope,
    from_ops: I,
    stop_gradients: &[Tensor],
    pending_count: &HashMap<NodeIdent, usize>,
) -> Result<HashSet<NodeIdent>>
where
    I: Iterator<Item = &'a NodeIdent>,
{
    let mut stop_ops = HashSet::new();
    for op in from_ops {
        let mut is_stop_op = true;
        for inp in op.get_inputs(scope)? {
            let inp_op = inp.get_op();
            if pending_count[inp_op] > 0 {
                is_stop_op = false;
                break;
            }
        }
        if is_stop_op {
            stop_ops.insert(*op);
        }
    }
    Ok(stop_ops)
}

fn is_trainable(tensor: &Tensor) -> bool {
    match tensor.dtype {
        DataType::BFloat16
        | DataType::Float
        | DataType::Double
        | DataType::Complex64
        | DataType::Complex128 => true,
        _ => false,
    }
}

/// Backprop through a function call node op given its outputs' gradients.
fn sym_grad(
    scope: &mut Scope,
    op: &NodeIdent,
    out_grads: &[Option<Tensor>],
) -> Result<Vec<Option<Tensor>>> {
    let mut f_types = vec![];
    let mut f_in = vec![];
    for x in op.get_inputs(scope)? {
        f_types.push(x.dtype);
        f_in.push(x);
    }
    f_in.extend(out_grads.iter().filter_map(|x| *x));

    let f = NameAttrList::new(op.get_type(scope)?);
    functional_ops::symbolic_gradient(scope, f_in, f_types, f.serialize())
}

///  Get the aggregated gradients for op.
///
///  ### Args:
///    * grads: The map of memoized gradients.
///    * op: The op to get gradients for.
///    * loop_state: An object for maintaining the state of the while loops in the
///                graph. It is of type ControlFlowState. None if the graph
///                contains no while loops.
///    * aggregation_method: Specifies the method used to combine gradient terms.
///      Accepted values are constants defined in the class `AggregationMethod`.
///
///  ### Returns:
///    A list of gradients, one per each output of `op`. If the gradients
///      for a particular output is a list, this function aggregates it
///      before returning.
#[allow(unreachable_patterns)]
fn aggregated_grads(
    scope: &mut Scope,
    grads: &HashMap<NodeIdent, Vec<Vec<Tensor>>>,
    op: &NodeIdent,
    loop_state: &Option<ControlFlowState>,
    aggregation_method: Option<AggregationMethod>,
) -> Result<Vec<Option<Tensor>>> {
    let aggregation_method = if let Some(method) = aggregation_method {
        method
    } else {
        AggregationMethod::default()
    };
    match aggregation_method {
        AggregationMethod::AddN
        | AggregationMethod::ExperimentalTree
        | AggregationMethod::ExperimentalAccumulateN => {}
        _ => {
            return Err(Error::from(format!(
                "Invalid aggregation_method specified {:?}.",
                aggregation_method
            )))
        }
    }
    let mut out_grads = vec![];
    for out_grad in get_grads(scope, grads, op)? {
        if loop_state.is_some() {
            if !control_flow_util::is_loop_switch(op) {
                return Err(Error::from("operation is not a loop switch"));
            }
            continue;
        }
        // Aggregate multiple gradients, and convert [] to None.
        if !out_grad.is_empty() {
            if out_grad.len() < 2 {
                out_grads.push(Some(out_grad[0]));
            } else {
                let tensor_shape = accumulator_shape(scope, &out_grad);
                if aggregation_method == AggregationMethod::ExperimentalAccumulateN
                    && out_grad.len() > 2 && tensor_shape.is_fully_defined()
                {
                    // The benefit of using AccumulateN is that its inputs can be combined
                    // in any order and this can allow the expression to be evaluated with
                    // a smaller memory footprint.  When used with gpu_allocator_retry,
                    // it is possible to compute a sum of terms which are much larger than
                    // total GPU memory.
                    // AccumulateN can currently only be used if we know the shape for
                    // an accumulator variable.  If this is not known, or if we only have
                    // 2 grads then we fall through to the "tree" case below.
                    out_grads.push(Some(math_ops::accumulate_n(
                        scope,
                        out_grad,
                        None,
                        None,
                        "",
                    )?));
                } else if aggregation_method == AggregationMethod::ExperimentalAccumulateN
                    || aggregation_method == AggregationMethod::ExperimentalTree
                {
                    //  Aggregate all gradients by doing pairwise sums: this may
                    //  reduce performance, but it can improve memory because the
                    //  gradients can be released earlier.
                    let name = op.get_name(scope)? + "_gradient_sum";
                    let scope = &mut scope.name_scope(name, None);
                    let mut running_sum = out_grad[0];
                    for grad in out_grad.into_iter().skip(1) {
                        running_sum = math_ops::add_n(scope, vec![running_sum, grad], "")?;
                    }
                    out_grads.push(Some(running_sum));
                } else {
                    out_grads.push(Some(multi_device_add_n(scope, out_grad)?));
                }
                /* TODO:
                logging.vlog(2, "  _AggregatedGrads %d x %s using %s",
                    len(out_grad), tensor_shape, used)
                */
            }
        } else {
            // not out_grad
            // out_grads[i] is [], thus its aggregation is simply None.
            out_grads.push(None);
        }
    }
    Ok(out_grads)
}

fn multi_device_add_n(scope: &mut Scope, t: Vec<Tensor>) -> Result<Tensor> {
    unimplemented!()
}

fn accumulator_shape(scope: &Scope, inputs: &[Tensor]) -> TensorShape {
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
    let scope = scope.trim_end_matches('/').replace("/", "_");
    let xla_compile;
    let xla_separate_compiled_gradients;
    let xla_scope;
    if let Some(func) = func_call {
        xla_compile = func.get_attr("_XlaCompile")
            .ok_or_else(|| Error::from(ErrorKind::UndefinedFuncAttr))?
            .unwrap_b();
        xla_separate_compiled_gradients = func.get_attr("_XlaSeparateCompiledGradients")
            .ok_or_else(|| Error::from(ErrorKind::UndefinedFuncAttr))?
            .unwrap_b();
        xla_scope = func.get_attr("_XlaScope")
            .ok_or_else(|| Error::from(ErrorKind::UndefinedFuncAttr))?
            .unwrap_s();
    } else {
        let xla_compile_2 = op.get_attr(context, "_XlaCompile");
        let xla_separate_compiled_gradients_2 =
            op.get_attr(context, "_XlaSeparateCompiledGradients");
        let xla_scope_2 = op.get_attr(context, "__XlaScope");
        if xla_compile_2.is_none() | xla_separate_compiled_gradients_2.is_none()
            | xla_scope_2.is_none()
        {
            return grad_fn(context, op, inputs);
        } else {
            xla_compile = xla_compile_2.unwrap().unwrap_b();
            xla_separate_compiled_gradients = xla_separate_compiled_gradients_2.unwrap().unwrap_b();
            xla_scope = xla_scope_2.unwrap().unwrap_s();
        }
    }

    if !xla_compile {
        return grad_fn(context, op, inputs);
    }

    // If the gradients are supposed to be compiled separately, we give them a
    // _XlaScope name that is based on the name_scope of the gradients.  Otherwise
    // they just inherit the existing _XlaScope name, which lets them be merged
    // together with the non-gradient computation.
    let _xla_grad_scope;
    if xla_separate_compiled_gradients {
        _xla_grad_scope = format!("{}_grad_{}", xla_scope, scope);
    } else {
        _xla_grad_scope = xla_scope;
    }

    /* FIXME:
        attrs = {
            "_XlaCompile": attr_value_pb2.AttrValue(b=xla_compile),
            "_XlaScope": attr_value_pb2.AttrValue(s=xla_grad_scope.encode())
        }
        with ops.get_default_graph()._attr_scope(attrs):
    */
    grad_fn(context, op, inputs)
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
    fn maybe_create_control_flow_state(
        state: &mut Scope,
        between_op_list: &[NodeIdent],
        between_ops: &HashMap<NodeIdent, bool>,
        colocate_gradients_with_ops: bool,
    ) -> Result<Option<ControlFlowState>> {
        unimplemented!()
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

mod control_flow_util {
    use framework::NodeIdent;

    pub fn is_loop_switch(op: &NodeIdent) -> bool {
        unimplemented!()
    }
}

mod functional_ops {
    use super::*;

    pub fn symbolic_gradient(
        scope: &mut Scope,
        input: Vec<Tensor>,
        t_out: Vec<DataType>,
        f: Vec<u8>,
    ) -> Result<Vec<Option<Tensor>>> {
        unimplemented!()
    }
}

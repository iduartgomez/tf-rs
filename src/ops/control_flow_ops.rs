//! Control Flow Operations.

use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

use TensorShape;

use super::*;

type CondSubGraph<'a> = Box<dyn FnMut(&mut Scope) -> Result<Vec<Tensor>> + 'a>;
type WhileCondGraph<'a> = Box<dyn Fn(&mut Scope, &[Tensor]) -> Result<Tensor> + 'a>;
type WhileBodyGraph<'a> = Box<dyn Fn(&mut Scope, &[Tensor]) -> Result<Vec<Tensor>> + 'a>;

#[derive(Debug, Clone)]
pub(crate) enum ControlFlow {
    CondContext(CondContext),
    None,
}

impl ControlFlow {
    pub(crate) fn get_cond(&self) -> Option<&CondContext> {
        match *self {
            ControlFlow::CondContext(ref cond) => Some(cond),
            _ => None,
        }
    }

    fn get_mut_cond(&mut self) -> Option<&mut CondContext> {
        match *self {
            ControlFlow::CondContext(ref mut cond) => Some(cond),
            _ => None,
        }
    }
}

impl PartialEq for ControlFlow {
    fn eq(&self, rhs: &ControlFlow) -> bool {
        match *self {
            ControlFlow::CondContext(_) => {
                if let ControlFlow::CondContext(_) = *rhs {
                    true
                } else {
                    false
                }
            }
            ControlFlow::None => false,
        }
    }
}

///// Assert /////

/// Asserts that the given condition is true.
///
/// If condition evaluates to false, print the list of tensors in data. summarize determines how many entries of the tensors to print.
///
/// _NOTE:_ To ensure that Assert executes, one usually attaches a dependency:
#[derive(Debug, Clone)]
pub struct Assert<'a> {
    ident: NodeIdent,
    elements: [Tensor; 1],
    name: Option<PathBuf>,
    attributes: Vec<(&'a str, bool, Attribute<'a>)>,
    input_lists: Vec<(usize, Vec<Tensor>)>,
}

impl<'a> Assert<'a> {
    fn new<S: AsRef<Path>>(condition: Tensor, data: Vec<Tensor>, name: S) -> Result<Assert<'a>> {
        if condition.dtype != DataType::Bool {
            return Err(Error::from(ErrorKind::Stub));
        }
        Ok(Assert {
            ident: NodeIdent::new(),
            elements: [condition],
            name: generate_name!(is_none: name),
            attributes: vec![],
            input_lists: vec![(1, data)],
        })
    }

    /// Default is [3], must be an slice of len == 1.
    fn summarize(mut self, val: &'a [i64]) -> Self {
        self.attributes
            .push(("summarize", false, Attribute::Int(val)));
        self
    }
}

impl<'a> Operation<'a> for Assert<'a> {
    type Outputs = ();

    add_new_op!(CORE_FN: Assert);

    fn digest(self, context: &mut Scope, op: OperationData) -> Result<Self::Outputs> {
        add_new_op!(REGISTER_AS_OP: (self, context, op); Assert);
        Ok(())
    }
}

impl_into_ident!(Assert);

/// Assert the condition `x == y` holds element-wise.
///
/// Example of adding a dependency to an operation:
///
/// ```code
/// let assert = assert_equal(root, x, y, "")?;
/// let scope = &mut root.control_dependencies(&[assert]);
/// let output = reduce_sum(scope, x, &[1_i32], false, "")?;
/// ```
///
/// This condition holds if for every pair of (possibly broadcast) elements `x[i]`, `y[i]`,
/// we have `x[i] == y[i]`. If both `x` and `y` are empty, this is trivially satisfied.
pub fn assert_eq<'a, Tx, Ty, S>(
    context: &mut Scope,
    x: Tx,
    y: Ty,
    data: Option<Vec<Tensor>>,
    summarize: Option<&'a [i64]>,
    name: S,
) -> Result<Assert<'a>>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);

    let data = if let Some(data) = data {
        data
    } else {
        vec![x, y]
    };

    let scope = &mut context.name_scope(name.as_ref(), Some("assert_equal".as_ref()));
    let eq = equal(scope, x, y, "")?;
    let cond = reduce_all(scope, eq, &[] as &[i32], false, "")?;
    let mut assert = Assert::new(cond, data, "")?;
    if let Some(summarize) = summarize {
        assert = assert.summarize(summarize);
    }
    scope.install(assert.clone())?;
    Ok(assert)
}

#[cfg(test)]
mod test_assert_eq {
    #![allow(unused_imports)]
    use super::*;

    #[test]
    fn test_assert_eq_correct() {
        let mut context = Scope::new();
        let x = context.constant(&[2_i32], &[] as &[i32], "x").unwrap();
        let y = context.constant(&[2_i32], &[] as &[i32], "y").unwrap();
        let assert = assert_eq(&mut context, x, y, None, None, "").unwrap();
        let op = {
            let mut context = context.control_dependencies(&[assert]);
            math_ops::add(&mut context, x, y, "").unwrap()
        };
        let results = test_suite!(run_op: [op]; context, input: {});
        test_suite!(results; assert: {[0;Int32] == [4_i32]});
    }

    #[test]
    fn test_assert_eq_incorrect() {
        let mut context = Scope::new();
        let x = context.constant(&[2_i32], &[] as &[i32], "x").unwrap();
        let y = context.constant(&[4_i32], &[] as &[i32], "y").unwrap();
        let assert = assert_eq(&mut context, x, y, None, None, "").unwrap();
        let op = {
            let mut context = context.control_dependencies(&[assert]);
            math_ops::add(&mut context, x, y, "").unwrap()
        };
        test_suite!(run_err: [op]; context, input: {});
    }
}

/// Assert the condition `x > y` holds element-wise.
///
/// Example of adding a dependency to an operation:
///
/// ```code
/// let assert = assert_equal(root, x, y, "")?;
/// let scope = &mut root.control_dependencies(&[assert]);
/// let output = reduce_sum(scope, x, &[1_i32], false, "")?;
/// ```
///
/// This condition holds if for every pair of (possibly broadcast) elements `x[i]`, `y[i]`,
/// we have `x[i] > y[i]`. If both `x` and `y` are empty, this is trivially satisfied.
pub fn assert_greater<'a, Tx, Ty, S>(
    context: &mut Scope,
    x: Tx,
    y: Ty,
    data: Option<Vec<Tensor>>,
    summarize: Option<&'a [i64]>,
    name: S,
) -> Result<Assert<'a>>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let x = x.into_tensor(context);
    let y = y.into_tensor(context);

    let data = if let Some(data) = data {
        data
    } else {
        vec![x, y]
    };

    let scope = &mut context.name_scope(name.as_ref(), Some("assert_greater".as_ref()));
    let eq = greater(scope, x, y, "")?;
    let cond = reduce_all(scope, eq, &[] as &[i32], false, "")?;
    let mut assert = Assert::new(cond, data, "")?;
    if let Some(summarize) = summarize {
        assert = assert.summarize(summarize);
    }
    scope.install(assert.clone())?;
    Ok(assert)
}

#[cfg(test)]
mod test_assert_greater {
    #![allow(unused_imports)]
    use super::*;

    #[test]
    fn test_assert_greater_correct() {
        let mut context = Scope::new();
        let x = context.constant(&[3_i32], &[] as &[i32], "x").unwrap();
        let y = context.constant(&[2_i32], &[] as &[i32], "y").unwrap();
        let assert = assert_greater(&mut context, x, y, None, None, "").unwrap();
        let op = {
            let mut context = context.control_dependencies(&[assert]);
            math_ops::add(&mut context, x, y, "").unwrap()
        };
        let results = test_suite!(run_op: [op]; context, input: {});
        test_suite!(results; assert: {[0;Int32] == [5_i32]});
    }

    #[test]
    fn test_assert_greater_incorrect() {
        let mut context = Scope::new();
        let x = context.constant(&[2_i32], &[] as &[i32], "x").unwrap();
        let y = context.constant(&[3_i32], &[] as &[i32], "y").unwrap();
        let assert = assert_greater(&mut context, x, y, None, None, "").unwrap();
        let op = {
            let mut context = context.control_dependencies(&[assert]);
            math_ops::add(&mut context, x, y, "").unwrap()
        };
        test_suite!(run_err: [op]; context, input: {});
    }
}

///// Cond /////

/// Return `true_fn()` if the predicate `pred` is true else `false_fn()`.
///
/// `true_fn` and `false_fn` both return lists of output tensors. `true_fn` and
/// `false_fn` must have the same non-zero number and type of outputs.
///
/// Note that the conditional execution applies only to the operations defined in
/// `true_fn` and `false_fn`.
pub fn cond<S>(
    context: &mut Scope,
    pred: Tensor,
    true_fn: CondSubGraph,
    false_fn: CondSubGraph,
    name: S,
) -> Result<Vec<Tensor>>
where
    S: AsRef<Path>,
{
    if pred.dtype != DataType::Bool {
        return Err(Error::from(ErrorKind::Stub));
    }
    if pred.get_shape(context) != TensorShape::from(Some(vec![])) {
        let msg = format!(
            "tf: expected shape `[]` for pred Tensor on `cond` op call, found shape: `{:?}`",
            pred.get_shape(context)
        );
        return Err(Error::from(msg));
    }

    let scope = &mut context.name_scope(name.as_ref(), Some("Cond".as_ref()));
    // Add the switch to the graph.
    let (p_0, p_1) = switch(scope, pred, pred, "")?;

    let pivot_0 = scope.identity(p_0, "switch_f")?;
    let pivot_1 = scope.identity(p_1, "switch_t")?;
    let pred = scope.identity(pred, "pred_id")?;
    // Disable the fetching of tensors that are only on one branch of cond.
    for tensor in &[p_0, p_1, pivot_0, pivot_1, pred] {
        scope.prevent_fetching(tensor);
    }

    // Build the graph for the true branch in a new context.
    let (_orig_res_t, res_t) = {
        let name = {
            let name: &Path = scope.name().as_ref();
            name.join("true_branch").to_str().unwrap().to_owned()
        };
        let mut context_t =
            scope.cond_scope(CondContext::new(pred, pivot_1, 1, name), "true_branch");
        context_t.build_cond_branch(true_fn)?
    };

    // Build the graph for the false branch in a new context.
    let (_orig_res_f, res_f) = {
        let name = {
            let name: &Path = scope.name().as_ref();
            name.join("false_branch").to_str().unwrap().to_owned()
        };
        let mut context_f =
            scope.cond_scope(CondContext::new(pred, pivot_0, 0, name), "false_branch");
        context_f.build_cond_branch(false_fn)?
    };

    if res_t.len() == 1 && res_f.len() == 1 {
        let res_t = res_t[0];
        let res_f = res_f[0];

        if (res_t.dtype != res_f.dtype) || (res_t.is_ref() != res_f.is_ref()) {
            return Err(Error::from(ErrorKind::Stub));
        }

        if res_t.is_ref() {
            Ok(vec![ref_merge(scope, vec![res_t, res_f], "")?.0])
        } else {
            Ok(vec![merge(scope, vec![res_t, res_f], "")?.0])
        }
    } else if res_t.len() == res_f.len() && !res_t.is_empty() {
        unimplemented!()
    } else {
        Err(Error::from(ErrorKind::Stub))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CondContext {
    pub name: String,
    /// The boolean tensor for the cond predicate
    pub pred: Tensor,
    /// The predicate tensor in this branch.
    pub pivot: Tensor,
    /// 0 or 1 representing this branch
    pub branch: u8,
    /// Values considered to have been already seen in this context.
    pub values: HashSet<NodeIdent>,
    /// Values referenced by but external to this context.
    pub external_values: HashMap<NodeIdent, Tensor>,
    pub new_switch: bool,
}

impl CondContext {
    fn new(pred: Tensor, pivot: Tensor, branch: u8, name: String) -> CondContext {
        CondContext {
            name,
            pred,
            pivot,
            branch,
            values: HashSet::new(),
            external_values: HashMap::new(),
            new_switch: false,
        }
    }
}

pub(crate) trait CondContextInterface {
    fn cond_scope<S: AsRef<Path>>(&mut self, cond_context: CondContext, name: S) -> Scope;

    fn process_output_tensor(&mut self, val: &Tensor) -> Tensor;

    fn build_cond_branch(&mut self, deps: CondSubGraph) -> Result<(Vec<Tensor>, Vec<Tensor>)>;
}

impl CondContextInterface for Scope {
    fn cond_scope<S: AsRef<Path>>(&mut self, mut cond_context: CondContext, name: S) -> Scope {
        self.allow_writes();
        let name = self.resolve_scope_name(name, "cond");
        let mut context = self.as_new_child(name);
        if let ControlFlow::CondContext(ref mut cond) = context.control_context {
            cond_context.values.extend(cond.values.iter())
        };
        context.control_context = ControlFlow::CondContext(cond_context);
        context
    }

    fn process_output_tensor(&mut self, val: &Tensor) -> Tensor {
        if self.control_context
            .get_cond()
            .unwrap()
            .values
            .get(&val.ident)
            .is_none()
        {
            let (pred, branch) = {
                let control_context = self.control_context.get_mut_cond().unwrap();
                control_context.new_switch = true;
                (control_context.pred, control_context.branch)
            };
            let real_val = switch_ref_or_tensor(self, *val, pred).unwrap();
            let real_val = if branch == 0 { real_val.0 } else { real_val.1 };
            let cond = self.control_context.get_mut_cond().unwrap();
            cond.values.insert(val.ident);
            cond.external_values.insert(val.ident, real_val);
            cond.new_switch = false;
            real_val
        } else {
            self.control_context.get_cond().unwrap().external_values[&val.ident]
        }
    }

    fn build_cond_branch(&mut self, mut deps: CondSubGraph) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let original_result = deps(self)?;
        let result: Vec<_> = original_result
            .iter()
            .map(|x| self.process_output_tensor(x))
            .collect();
        Ok((original_result, result))
    }
}

///// Switch /////

pub fn switch<Tx, Ty, S>(
    context: &mut Scope,
    data: Tx,
    pred: Ty,
    name: S,
) -> Result<(Tensor, Tensor)>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let data = data.into_tensor(context);
    let pred = pred.into_tensor(context);
    // returns (output_false, outpur_true)
    context.install(Switch::new(data, pred, name)?)
}

// Forwards `data` to the output port determined by `pred`.
//
// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
// the data goes to `output_false`.
add_new_op!(Switch,
    constructor: [add_new_op!(BIN CONSTRUCTOR: Switch, Init: []);],
    digest: [DIGEST_BIN_OUT: Switch, INPUT0, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [(Tensor, Tensor)],
);

///// RefSwitch /////

pub fn ref_switch<Tx, Ty, S>(
    context: &mut Scope,
    data: Tx,
    pred: Ty,
    name: S,
) -> Result<(Tensor, Tensor)>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let data = data.into_tensor(context);
    let pred = pred.into_tensor(context);
    context.install(RefSwitch::new(data, pred, name)?)
}

// Forwards `data` to the output port determined by `pred`.
//
// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
// the data goes to `output_false`.
add_new_op!(RefSwitch,
    constructor: [add_new_op!(BIN CONSTRUCTOR: RefSwitch, Init: []);],
    digest: [DIGEST_BIN_OUT: RefSwitch, INPUT0, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [(Tensor, Tensor)],
);

///// Merge /////

pub fn merge<S>(context: &mut Scope, values: Vec<Tensor>, name: S) -> Result<(Tensor, Tensor)>
where
    S: AsRef<Path>,
{
    context.install(Merge::new(values, name)?)
}

// Forwards the value of an available tensor from `inputs` to `output`.
//
// `Merge` waits for at least one of the tensors in `inputs` to become available.
// It is usually combined with `Switch` to implement branching.
//
// `Merge` forwards the first tensor to become available to `output`, and sets
// `value_index` to its index in `inputs`.
add_new_op!(Merge,
    constructor: [
        fn new<S: AsRef<Path>>(values: Vec<Tensor>, name: S) -> Result<Merge<'a>> {
            let output_type = values[0].dtype;
            Ok(
                Merge {
                    ident: NodeIdent::new(),
                    name: generate_name!(is_none: name),
                    attributes: Vec::with_capacity(0),
                    elements: vec![],
                    input_lists: vec![(0, values)],
                    output_type: output_type,
                    out2: DataType::Int32
                },
            )
        }
    ],
    digest: [DIGEST_BIN_OUT: Merge, DTYPE_ATTR, DTYPE_ATTR_2],
    extra_funcs: [], 
    extra_attr: [output_type: DataType, out2: DataType],
    output: [(Tensor, Tensor)],
);

///// RefMerge /////

pub fn ref_merge<S>(context: &mut Scope, values: Vec<Tensor>, name: S) -> Result<(Tensor, Tensor)>
where
    S: AsRef<Path>,
{
    context.install(RefMerge::new(values, name)?)
}

/// Forwards the value of an available tensor from `inputs` to `output`.
///
/// `Merge` waits for at least one of the tensors in `inputs` to become available.
/// It is usually combined with `Switch` to implement branching.
///
/// `Merge` forwards the first tensor for become available to `output`, and sets
/// `value_index` to its index in `inputs`.
add_new_op!(RefMerge,
    constructor: [
        fn new<S: AsRef<Path>>(values: Vec<Tensor>, name: S) -> Result<RefMerge<'a>> {
            let output_type = values[0].dtype;
            Ok(
                RefMerge {
                    ident: NodeIdent::new(),
                    name: generate_name!(is_none: name),
                    attributes: Vec::with_capacity(0),
                    elements: vec![],
                    input_lists: vec![(0, values)],
                    output_type: output_type,
                    out2: DataType::Int32
                },
            )
        }
    ],
    digest: [DIGEST_BIN_OUT: RefMerge, DTYPE_ATTR, DTYPE_ATTR_2], 
    extra_funcs: [], 
    extra_attr: [output_type: DataType, out2: DataType],
    output: [(Tensor, Tensor)],
);

///// WhileLoop /////

/// Repeat `body` while the condition `cond` is true.
///
/// `cond` is a callable returning a boolean scalar tensor. `body` is a callable
/// returning a list of tensors of the same arity (length and structure) and types
/// as `loop_vars`. `loop_vars` is a list of tensors that is passed to both
/// `cond` and `body`.
#[allow(dead_code)]
fn while_loop<S>(
    context: &mut Scope,
    cond: WhileCondGraph,
    body: WhileBodyGraph,
    loop_vars: &[Tensor],
    name: S,
) -> Result<Vec<Tensor>>
where
    S: AsRef<Path>,
{
    let name = if name_cmp!(name, "") {
        "while"
    } else {
        name.as_ref()
            .to_str()
            .ok_or_else(|| Error::from(ErrorKind::NoneError))?
    };

    // println!("MAIN GRAPH: {:#?}", context.graph.borrow());
    let inputs: Vec<_> = loop_vars
        .iter()
        .map(|x| Output {
            operation: context.get_src_op(x),
            index: x.idx,
        })
        .collect();

    let pred_fn = |g: &mut Graph, lv: &[Output]| -> tf::Result<Output> {
        println!("PRED_LV");

        let mut scope = context.clone();
        // let tensor_names = inputs
        //     .iter()
        //     .map(|x| x.operation.name().unwrap())
        //     .collect::<Vec<_>>();

        // let lv: Vec<_> = lv.iter()
        //     .map(|x| context.get_tensor_from_data(x.index, x.operation.clone()))
        //     .collect::<Result<Vec<_>>>()?;
        let result = cond(&mut scope, loop_vars)?;

        Ok(Output {
            operation: scope.get_src_op(result),
            index: result.idx,
        })
    };

    let body_fn = |g: &mut Graph, lv: &[Output]| -> tf::Result<Vec<Output>> {
        println!("BODY_LV");
        let mut scope = context.clone();
        // let lv: Vec<_> = lv.iter()
        //     .map(|x| context.get_tensor_from_data(x.index, x.operation.clone()))
        //     .collect::<Result<Vec<_>>>()?;
        let result = body(&mut scope, loop_vars)?;

        Ok(result
            .iter()
            .map(|x| Output {
                operation: scope.get_src_op(x),
                index: x.idx,
            })
            .collect::<Vec<_>>())
    };

    let graph = unsafe {
        // Make sure the graph is not borrowed:
        context.graph.try_borrow_mut()?;
        // Is safe to get a mutable reference to the graph:
        &mut *(&mut *context.graph.borrow_mut() as *mut _)
    };
    tf::WhileBuilder::new(graph, pred_fn, body_fn, inputs.as_ref())?
        .name(name)?
        .finish()?
        .iter()
        .map(|x| context.get_tensor_from_data(x.index, x.operation.clone()))
        .collect()
}

fn switch_ref_or_tensor(scope: &mut Scope, data: Tensor, pred: Tensor) -> Result<(Tensor, Tensor)> {
    // TODO: with ops.colocate_with(data)
    if data.is_ref() {
        ref_switch(scope, data, pred, "RefSwitch")
    } else {
        switch(scope, data, pred, "Switch")
    }
}

///// Tuple /////

///  Group tensors together.
///
///  This creates a tuple of tensors with the same values as the `tensors`
///  argument, except that the value of each tensor is only returned after the
///  values of all tensors have been computed.
///
///  `control_inputs` contains additional ops that have to finish before this op
///  finishes, but whose outputs are not returned.
///
///  This can be used as a "join" mechanism for parallel computations: all the
///  argument tensors can be computed in parallel, but the values of any tensor
///  returned by `tuple` are only available after all the parallel computations
///  are done.
///
///  ### Args:
///    * tensors: A list of `Tensor`s.
///    * control_inputs: List of additional ops to finish before returning.
///    * name: (optional) A name to use as a `name_scope` for the operation.
///
///  ### Returns:
///    Same as `tensors`.
pub fn tuple<'a, S, I, T: 'a>(
    context: &mut Scope,
    tensors: Vec<Tensor>,
    control_inputs: Option<I>,
    name: S,
) -> Result<Vec<Tensor>>
where
    S: AsRef<Path>,
    I: IntoIterator<Item = &'a T>,
    T: GetOp,
{
    let scope = &mut context.name_scope(name.as_ref().to_str().unwrap(), Some("tuple"));
    let mut gating_ops: Vec<_> = tensors.iter().map(|x| *x.get_op()).collect();
    if let Some(control_inputs) = control_inputs {
        for c in control_inputs.into_iter() {
            gating_ops.push(*c.get_op())
        }
    }
    // Note that in order to ensure ordering in the pbtxt, we must take care to
    // ensure the order here.
    let gating_ops: HashSet<NodeIdent> = HashSet::from_iter(gating_ops.into_iter());
    if gating_ops.is_empty() {
        return Err(Error::from(
            "`tuple` op must have at least one input Tensor or Operation.",
        ));
    }
    let gate = Group::new(scope, &gating_ops, "")?;
    let mut tpl = Vec::with_capacity(tensors.len());
    for t in tensors {
        tpl.push(with_dependencies(scope, &[gate], t, "")?);
    }
    Ok(tpl)
}

/// Create an op that groups multiple operations.
///
/// When this op finishes, all ops in input have finished. This op has no output.
/// This is useful for control flow in conjunction with `control_dependencies` for example.
#[derive(Debug, Clone, Copy)]
pub struct Group(NodeIdent);

impl Group {
    pub fn new<'a, I, T: 'a, S>(scope: &mut Scope, ops: I, name: S) -> Result<Group>
    where
        I: IntoIterator<Item = &'a T>,
        T: GetOp,
        S: AsRef<Path>,
    {
        let graph = &mut scope.graph.borrow_mut();
        let tensors = &*scope.tensors.borrow();
        let ops_reg = &mut *scope.ops.borrow_mut();

        let finished = {
            let mut ctrl_ops = Vec::new();
            for x in ops {
                let op = if let Some(tensor) = tensors.get(x.get_op()) {
                    &tensor.data_origin.0
                } else {
                    ops_reg
                        .get(x.get_op())
                        .ok_or_else(|| Error::from(ErrorKind::OpNotFound))?
                };
                ctrl_ops.push(op);
            }

            const OP: IdType = IdType::Operation("Group");
            let name = scope.resolve_name(Some(name.as_ref()), OP, false)?;
            no_op_(graph, name.to_str().unwrap(), ctrl_ops)?
        };
        let ident = NodeIdent::new();
        ops_reg.insert(ident, finished);

        Ok(Group(ident))
    }
}

impl Into<NodeIdent> for Group {
    fn into(self) -> NodeIdent {
        self.0
    }
}

impl GetOp for Group {
    fn get_op(&self) -> &NodeIdent {
        &self.0
    }

    fn source_index(&self) -> Option<i32> {
        None
    }
}

///  Produces the content of `output_tensor` only after `dependencies`.
///
///  In some cases, a user may want the output of an operation to be
///  consumed externally only after some other dependencies have run
///  first. This function ensures returns `output_tensor`, but only after all
///  operations in `dependencies` have run. Note that this means that there is
///  no guarantee that `output_tensor` will be evaluated after any `dependencies`
///  have run.
///
///  ### Args:
///    * dependencies: Iterable of operations to run before this op finishes.
///    * output_tensor: A `Tensor` or `IndexedSlices` that will be returned.
///    * name: (Optional) A name for this operation.
///
///  ### Returns:
///    Same as `output_tensor`.
pub fn with_dependencies<'a, I, T: 'a, S>(
    context: &mut Scope,
    dependencies: I,
    output_tensor: Tensor,
    name: S,
) -> Result<Tensor>
where
    I: IntoIterator<Item = &'a T>,
    T: GetOp,
    S: AsRef<Path>,
{
    let scope =
        &mut context.name_scope(name.as_ref().to_str().unwrap(), Some("control_dependency"));
    let name = scope.name().to_owned();
    // TODO: with ops.colocate_with(output_tensor)
    let scope = &mut scope.control_dependencies(dependencies);
    scope.identity(output_tensor, name)
}

pub(crate) fn zeros_like_outside_loop<Op>(
    context: &mut Scope,
    op: Op,
    index: usize,
) -> Result<Tensor>
where
    Op: GetOp,
{
    unimplemented!()
}

#[cfg(test)]
mod test {
    #![allow(unused_imports)]
    use super::*;

    #[test]
    fn test_cond() {
        use super::assign;
        let mut context = Scope::new();
        let var: Tensor = context
            .get_variable(Some(DataType::Int32), Some(&[] as &[i32]), "")
            .unwrap()
            .into();
        let x = context.constant(&[2_i32], &[] as &[i32], "").unwrap();
        let y = context.constant(&[5_i32], &[] as &[i32], "").unwrap();

        let f1 = Box::new(move |scope: &mut Scope| -> Result<Vec<Tensor>> {
            Ok(vec![assign(scope, var, x, true, "")?])
        });
        let f2 = Box::new(move |scope: &mut Scope| -> Result<Vec<Tensor>> {
            Ok(vec![assign(scope, var, y, true, "")?])
        });

        let pred = less(&mut context, y, x, "").unwrap();
        let op = cond(&mut context, pred, f1, f2, "").unwrap()[0];
        let r = test_suite!(run_op: [op, var]; context, input: {});
        test_suite!(r; assert_len: {[0;Int32] == 1});
        test_suite!(r; assert: {[0;Int32] == [5_i32], [1;Int32] == [5_i32]});

        let f1 = Box::new(move |mut scope: &mut Scope| -> Result<Vec<Tensor>> {
            let mult_x = scope.constant(&[10_i32], &[] as &[i32], "")?;
            let add_x = scope.constant(&[20_i32], &[] as &[i32], "")?;
            let v = multiply(&mut scope, x, mult_x, "")?;
            let v = add(&mut scope, v, add_x, "")?;
            Ok(vec![assign(scope, var, v, true, "")?])
        });
        let f2 = Box::new(move |scope: &mut Scope| -> Result<Vec<Tensor>> {
            Ok(vec![assign(scope, var, y, true, "")?])
        });

        let pred = less(&mut context, x, y, "").unwrap();
        let op = cond(&mut context, pred, f1, f2, "").unwrap()[0];
        let r = test_suite!(run_op: [op, var]; context, input: {});
        test_suite!(r; assert_len: {[0;Int32] == 1});
        test_suite!(r; assert: {[0;Int32] == [40_i32], [1;Int32] == [40_i32]});
    }

    #[test]
    fn test_while_loop() {
        println!("\nTEST WHILE LOOP\n");
        let mut context = Scope::new();
        let counter = context
            .constant(&[0_i32], &[] as &[i32], "Const=0")
            .unwrap();

        let pred = Box::new(|scope: &mut Scope, loop_vars: &[Tensor]| {
            println!("ENTERED PRED FUNC");
            let y = scope
                .constant(&[10_i32], &[] as &[i32], "Pred-Const=10")
                .unwrap();
            let x = loop_vars[0];
            less(scope, x, y, "")
        });

        let body = Box::new(
            |scope: &mut Scope, loop_vars: &[Tensor]| -> Result<Vec<Tensor>> {
                println!("ENTERED BODY FUNC");
                let y = scope
                    .constant(&[1_i32], &[] as &[i32], "Body-Const=1")
                    .unwrap();
                let x = loop_vars[0];
                Ok(vec![add(scope, x, y, "")?])
            },
        );

        let op = while_loop(&mut context, pred, body, &[counter.into()], "").unwrap()[0];
        let r = test_suite!(run_op: [op]; context, input: {});
        test_suite!(r; assert_len: {[0;Int32] == 1});
        test_suite!(r; assert: {[0;Int32] == [10_i32]});
    }
}

///// Lower level support ops /////
pub(crate) fn no_op_<I, T>(
    graph: &mut Graph,
    name: &str,
    control_inputs: I,
) -> Result<OperationData>
where
    I: IntoIterator<Item = T>,
    T: ::std::ops::Deref<Target = OperationData>,
{
    let mut noop = graph.new_operation("NoOp", name)?;
    super::add_control_input(&mut noop, control_inputs);
    Ok(noop.finish()?)
}

//! Control Flow Operations.

use std::collections::{HashMap, HashSet};
use tf::Shape;

use super::*;

type CondSubGraph<'a> = Box<FnMut(&mut Scope) -> Result<Vec<Tensor>> + 'a>;
type WhileCondGraph<'a> = Box<FnMut(&mut Scope, &mut [Tensor]) -> Result<Tensor> + 'a>;
type WhileBodyGraph<'a> = Box<
    FnMut(&mut Scope, &mut [Tensor]) -> Result<Vec<Tensor>>
        + 'a,
>;


#[derive(Debug, Clone)]
pub(crate) enum ControlFlow {
    CondContext(CondContext),
    WhileContext(WhileContext),
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

    fn get_while_loop(&self) -> Option<&WhileContext> {
        match *self {
            ControlFlow::WhileContext(ref while_loop) => Some(while_loop),
            _ => None,
        }
    }

    fn get_mut_while_loop(&mut self) -> Option<&mut WhileContext> {
        match *self {
            ControlFlow::WhileContext(ref mut while_loop) => Some(while_loop),
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
            ControlFlow::WhileContext(_) => {
                if let ControlFlow::WhileContext(_) = *rhs {
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
    fn new<S: AsRef<Path>>(
        condition: Tensor,
        data: Vec<Tensor>,
        name: S,
    ) -> Result<Assert<'a>> {
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
        self.attributes.push(("summarize", false, Attribute::Int(val)));
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
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    let x = x.into();
    let y = y.into();

    let data = if let Some(data) = data {
        data
    } else {
        vec![x, y]
    };

    let scope = &mut context.name_scope(name.as_ref(), Some("assert_equal".as_ref()));
    let eq = equal(scope, x, y, "")?;
    let cond = reduce_all(scope, eq, &[], false, "")?;
    let mut assert = Assert::new(cond, data, "")?;
    if let Some(summarize) = summarize {
        assert = assert.summarize(summarize);
    }
    Ok(assert)
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
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    let x = x.into();
    let y = y.into();

    let data = if let Some(data) = data {
        data
    } else {
        vec![x, y]
    };
    
    let scope = &mut context.name_scope(name.as_ref(), Some("assert_greater".as_ref()));
    let eq = greater(scope, x, y, "")?;
    let cond = reduce_all(scope, eq, &[], false, "")?;
    let mut assert = Assert::new(cond, data, "")?;
    if let Some(summarize) = summarize {
        assert = assert.summarize(summarize);
    }
    Ok(assert)
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
    if pred.get_shape(context) != Shape::from(Some(vec![])) {
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

    fn build_cond_branch(
        &mut self,
        deps: CondSubGraph,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)>;
}

impl CondContextInterface for Scope {
    fn cond_scope<S: AsRef<Path>>(&mut self, mut cond_context: CondContext, name: S) -> Scope {
        self.allow_writes();
        let name = self.resolve_new_scope_name(name, "cond");
        let mut context = self.as_new_child(name);
        match context.control_context {
            ControlFlow::CondContext(ref mut cond) => {
                cond_context.values.extend(cond.values.iter())
            }
            ControlFlow::WhileContext(ref whileloop) => {
                cond_context.values.extend(whileloop.values.iter())
            }
            _ => {}
        }
        context.control_context = ControlFlow::CondContext(cond_context);
        context
    }

    fn process_output_tensor(&mut self, val: &Tensor) -> Tensor {
        if self.control_context.get_cond().unwrap().values.get(&val.ident).is_none() {
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

    fn build_cond_branch(
        &mut self,
        mut deps: CondSubGraph,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let original_result = deps(self)?;
        let result: Vec<_> =
            original_result.iter().map(|x| self.process_output_tensor(x)).collect();
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
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    // returns (output_false, outpur_true)
    context.install(Switch::new(data.into(), pred.into(), name)?)
}

/// Forwards `data` to the output port determined by `pred`.
///
/// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
/// the data goes to `output_false`.
add_new_op!(Switch,
    constructor: [add_new_op!(BIN CONSTRUCTOR: Switch, Init: []);],
    digest: [DIGEST: 
        fn digest(
            self, 
            context: &mut Scope, 
            op: OperationData
        ) 
            -> Result<Self::Outputs> 
        {
            let idtype = IdType::Operation("Switch");
            
            let dtype = add_new_op!(INPUT0 self);
            let shape = {
                let g = context.graph.borrow();
                g.tensor_shape(
                        Output {
                            operation: op.clone(),
                            index: 0,
                        },
                    )?
            };
        
            let ident0 = NodeIdent::new();
            let full_name0 = context.resolve_tensor_name(self.get_op_name(), idtype, false)?;
            let tensor0 = Tensor {
                ident: ident0,
                idtype,
                dtype,
                idx: 0,
                initializer: None,
            };

            let ident1 = NodeIdent::new();
            let full_name1 = context.resolve_tensor_name(self.get_op_name(), idtype, false)?;
            let tensor1 = Tensor {
                ident: ident1,
                idtype,
                dtype,
                idx: 1,
                initializer: None,
            };

            match context.control_context {
                ControlFlow::CondContext(ref mut cond) => {
                    cond.values.insert(ident0);
                    cond.external_values.insert(ident0, tensor0);
                    cond.values.insert(ident1);
                    cond.external_values.insert(ident1, tensor1);
                }
                ControlFlow::WhileContext(ref mut cond) => {
                    cond.values.insert(ident0);
                    cond.external_values.insert(ident0, tensor0);
                    cond.values.insert(ident1);
                    cond.external_values.insert(ident1, tensor1);
                }
                ControlFlow::None => {}
            }

            let reg = &mut *context.registry.borrow_mut();
            // OUTPUT FALSE:
            context.own_scope.ops.push((full_name0.clone(), ident0));
            reg.insert(
                ident0,
                TensorData {
                    full_name: full_name0,
                    dtype,
                    idtype,
                    data_origin: (op.clone(), 0),
                    shape: shape.clone(),
                },
            );
            // OUTPUT TRUE:
            context.own_scope.ops.push((full_name1.clone(), ident1));
            reg.insert(
                ident1,
                TensorData {
                    full_name: full_name1,
                    dtype,
                    idtype,
                    data_origin: (op, 1),
                    shape,
                },
            );
            
            Ok((tensor0, tensor1))
        } 
    ],
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
    Tx: Into<Tensor>,
    Ty: Into<Tensor>,
    S: AsRef<Path>,
{
    context.install(RefSwitch::new(data.into(), pred.into(), name)?)
}

/// Forwards `data` to the output port determined by `pred`.
///
/// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
/// the data goes to `output_false`.
add_new_op!(RefSwitch,
    constructor: [add_new_op!(BIN CONSTRUCTOR: RefSwitch, Init: []);],
    digest: [DIGEST: 
        fn digest(
            self, 
            context: &mut Scope, 
            op: OperationData
        ) 
            -> Result<Self::Outputs> 
        {
            let idtype = IdType::Operation("RefSwitch");
            
            let dtype = add_new_op!(INPUT0 self);
            let shape = {
                let g = context.graph.borrow();
                g.tensor_shape(
                        Output {
                            operation: op.clone(),
                            index: 0,
                        },
                    )?
            };
        
            let ident0 = NodeIdent::new();
            let full_name0 = context.resolve_tensor_name(self.get_op_name(), idtype, false)?;
            let tensor0 = Tensor {
                ident: ident0,
                idtype,
                dtype,
                idx: 0,
                initializer: None,
            };

            let ident1 = NodeIdent::new();
            let full_name1 = context.resolve_tensor_name(self.get_op_name(), idtype, false)?;
            let tensor1 = Tensor {
                ident: ident1,
                idtype,
                dtype,
                idx: 1,
                initializer: None,
            };

            match context.control_context {
                ControlFlow::CondContext(ref mut cond) => {
                    cond.values.insert(ident0);
                    cond.external_values.insert(ident0, tensor0);
                    cond.values.insert(ident1);
                    cond.external_values.insert(ident1, tensor1);
                }
                ControlFlow::WhileContext(ref mut cond) => {
                    cond.values.insert(ident0);
                    cond.external_values.insert(ident0, tensor0);
                    cond.values.insert(ident1);
                    cond.external_values.insert(ident1, tensor1);
                }
                ControlFlow::None => {}
            }

            let reg = &mut *context.registry.borrow_mut();
            context.own_scope.ops.push((full_name0.clone(), ident0));
            reg.insert(
                ident0,
                TensorData {
                    full_name: full_name0,
                    dtype,
                    idtype,
                    data_origin: (op.clone(), 0),
                    shape: shape.clone(),
                },
            );
            context.own_scope.ops.push((full_name1.clone(), ident1));
            reg.insert(
                ident1,
                TensorData {
                    full_name: full_name1,
                    dtype,
                    idtype,
                    data_origin: (op, 1),
                    shape,
                },
            );
            
            Ok((tensor0, tensor1))
        } 
    ],
    extra_funcs: [], 
    extra_attr: [],
    output: [(Tensor, Tensor)],
);


///// Merge /////

pub fn merge<S>(
    context: &mut Scope,
    values: Vec<Tensor>,
    name: S,
) -> Result<(Tensor, Tensor)>
where
    S: AsRef<Path>,
{
    context.install(Merge::new(values, name)?)
}

/// Forwards the value of an available tensor from `inputs` to `output`.
///
/// `Merge` waits for at least one of the tensors in `inputs` to become available.
/// It is usually combined with `Switch` to implement branching.
///
/// `Merge` forwards the first tensor to become available to `output`, and sets
/// `value_index` to its index in `inputs`.
add_new_op!(Merge,
    constructor: [
        fn new<S: AsRef<Path>>(values: Vec<Tensor>, name: S) -> Result<Merge<'a>> {
            let output_type = values[0].dtype;
            for x in &values {
                if &x.dtype != &output_type {
                    return Err(Error::from(ErrorKind::Stub));
                }
            }

            Ok(
                Merge {
                    ident: NodeIdent::new(),
                    name: generate_name!(is_none: name),
                    attributes: Vec::with_capacity(0),
                    elements: vec![],
                    input_lists: vec![(0, values)],
                    output_type: output_type,
                },
            )
        }
    ],
    digest: [DIGEST:
        fn digest(
            self, 
            context: &mut Scope, 
            op: OperationData
        ) 
            -> Result<Self::Outputs> 
        {
            let idtype = IdType::Operation("Merge");
            
            let ident0 = NodeIdent::new();
            let dtype0 = add_new_op!(DTYPE_ATTR self);
            let full_name0 = context.resolve_tensor_name(self.get_op_name(), idtype, false)?;
            let shape0 = {
                let g = context.graph.borrow();
                g.tensor_shape(
                        Output {
                            operation: op.clone(),
                            index: 0,
                        },
                    )?
            };
            let tensor0 = Tensor {
                ident: ident0,
                idtype,
                dtype: dtype0,
                idx: 0,
                initializer: None,
            };

            let ident1 = NodeIdent::new();
            let full_name1 = context.resolve_tensor_name(self.get_op_name(), idtype, false)?;
            let shape1 = {
                let g = context.graph.borrow();
                g.tensor_shape(
                        Output {
                            operation: op.clone(),
                            index: 0,
                        },
                    )?
            };
            let tensor1 = Tensor {
                ident: ident1,
                idtype,
                dtype: DataType::Int32,
                idx: 1,
                initializer: None,
            };

            match context.control_context {
                ControlFlow::CondContext(ref mut cond) => {
                    cond.values.insert(ident0);
                    cond.external_values.insert(ident0, tensor0);
                    cond.values.insert(ident1);
                    cond.external_values.insert(ident1, tensor1);
                }
                ControlFlow::WhileContext(ref mut cond) => {
                    cond.values.insert(ident0);
                    cond.external_values.insert(ident0, tensor0);
                    cond.values.insert(ident1);
                    cond.external_values.insert(ident1, tensor1);
                }
                ControlFlow::None => {}
            }

            let reg = &mut *context.registry.borrow_mut();
            context.own_scope.ops.push((full_name0.clone(), ident0));
            reg.insert(
                ident0,
                TensorData {
                    full_name: full_name0,
                    dtype: dtype0,
                    idtype,
                    data_origin: (op.clone(), 0),
                    shape: shape0,
                },
            );
            context.own_scope.ops.push((full_name1.clone(), ident1));
            reg.insert(
                ident1,
                TensorData {
                    full_name: full_name1,
                    dtype: DataType::Int32,
                    idtype,
                    data_origin: (op, 1),
                    shape: shape1,
                },
            );
            
            Ok((tensor0, tensor1))
        }
    ],
    extra_funcs: [], 
    extra_attr: [output_type: DataType],
    output: [(Tensor, Tensor)],
);


///// RefMerge /////

pub fn ref_merge<S>(
    context: &mut Scope,
    values: Vec<Tensor>,
    name: S,
) -> Result<(Tensor, Tensor)>
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
            for x in &values {
                if &x.dtype != &output_type {
                    return Err(Error::from(ErrorKind::Stub));
                }
            }

            Ok(
                RefMerge {
                    ident: NodeIdent::new(),
                    name: generate_name!(is_none: name),
                    attributes: Vec::with_capacity(0),
                    elements: vec![],
                    input_lists: vec![(0, values)],
                    output_type: output_type,
                },
            )
        }
    ],
    digest: [DIGEST:
        fn digest(
            self, 
            context: &mut Scope, 
            op: OperationData
        ) 
            -> Result<Self::Outputs> 
        {
            let idtype = IdType::Operation("RefMerge");
            
            let ident0 = NodeIdent::new();
            let dtype0 = add_new_op!(DTYPE_ATTR self);
            let full_name0 = context.resolve_tensor_name(self.get_op_name(), idtype, false)?;
            let shape0 = {
                let g = context.graph.borrow();
                g.tensor_shape(
                        Output {
                            operation: op.clone(),
                            index: 0,
                        },
                    )?
            };
            let tensor0 = Tensor {
                ident: ident0,
                idtype,
                dtype: dtype0,
                idx: 0,
                initializer: None,
            };

            let ident1 = NodeIdent::new();
            let full_name1 = context.resolve_tensor_name(self.get_op_name(), idtype, false)?;
            let shape1 = {
                let g = context.graph.borrow();
                g.tensor_shape(
                        Output {
                            operation: op.clone(),
                            index: 0,
                        },
                    )?
            };
            let tensor1 = Tensor {
                ident: ident1,
                idtype,
                dtype: DataType::Int32,
                idx: 1,
                initializer: None,
            };

            match context.control_context {
                ControlFlow::CondContext(ref mut cond) => {
                    cond.values.insert(ident0);
                    cond.external_values.insert(ident0, tensor0);
                    cond.values.insert(ident1);
                    cond.external_values.insert(ident1, tensor1);
                }
                ControlFlow::WhileContext(ref mut cond) => {
                    cond.values.insert(ident0);
                    cond.external_values.insert(ident0, tensor0);
                    cond.values.insert(ident1);
                    cond.external_values.insert(ident1, tensor1);
                }
                ControlFlow::None => {}
            }

            let reg = &mut *context.registry.borrow_mut();
            context.own_scope.ops.push((full_name0.clone(), ident0));
            reg.insert(
                ident0,
                TensorData {
                    full_name: full_name0,
                    dtype: dtype0,
                    idtype,
                    data_origin: (op.clone(), 0),
                    shape: shape0,
                },
            );
            context.own_scope.ops.push((full_name1.clone(), ident1));
            reg.insert(
                ident1,
                TensorData {
                    full_name: full_name1,
                    dtype: DataType::Int32,
                    idtype,
                    data_origin: (op, 1),
                    shape: shape1,
                },
            );
            
            Ok((tensor0, tensor1))
        }
    ],
    extra_funcs: [], 
    extra_attr: [output_type: DataType],
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
fn while_loop<'a, S>(
    context: &mut Scope,
    pred: WhileCondGraph,
    body: WhileBodyGraph,
    loop_vars: &mut [Tensor],
    name: S,
) -> Result<Vec<Tensor>>
where
    S: AsRef<Path>,
{
    use self::WhileContextInterface;

    let name = if name_cmp!(name, "") {
        Path::new("while")
    } else {
        name.as_ref()
    };

    let scope = &mut context.loop_scope(WhileContext::new(name.to_str().unwrap().to_owned()), name);
    scope.build_loop(pred, body, loop_vars)
}

#[derive(Debug, Clone)]
pub(crate) struct WhileContext {
    pub name: String,
    /// Values considered to have been already seen in this context.
    pub values: HashSet<NodeIdent>,
    /// Values referenced by but external to this context.
    pub external_values: HashMap<NodeIdent, Tensor>,
    /// The boolean tensor for loop termination condition.
    pub pivot: Option<Tensor>,
    /// We use this node to control constants created by the pred lambda.
    pub pivot_for_pred: Option<Tensor>,
    /// We use this node to control constants created by the body lambda.
    pub pivot_for_body: Option<Tensor>,
    /// The list of exit tensors for loop variables.
    pub loop_exits: Vec<Tensor>,
    /// The list of enter tensors for loop variables.
    pub loop_enters: Vec<Tensor>,
}

impl WhileContext {
    #[allow(dead_code)]
    fn new(name: String) -> WhileContext {
        WhileContext {
            name,
            pivot: None,
            pivot_for_body: None,
            pivot_for_pred: None,
            values: HashSet::new(),
            external_values: HashMap::new(),
            loop_exits: vec![],
            loop_enters: vec![],
        }
    }
}

pub(crate) trait WhileContextInterface {
    fn loop_scope<S: AsRef<Path>>(&mut self, cond_context: WhileContext, name: S) -> Scope;
    fn build_loop(
        &mut self,
        pred: WhileCondGraph,
        body: WhileBodyGraph,
        loop_vars: &mut [Tensor],
    ) -> Result<Vec<Tensor>>;

    fn initialize_values(&mut self, values: &[Tensor]);
    fn exit_result(&mut self, result: &[Tensor]);
}

macro_rules! while_context {
    (mut $ctx:ident) => ($ctx.control_context.get_mut_while_loop().unwrap());
    ($ctx:ident) => ($ctx.control_context.get_while_loop().unwrap());
}

impl WhileContextInterface for Scope {
    fn loop_scope<S: AsRef<Path>>(&mut self, mut cond_context: WhileContext, name: S) -> Scope {
        self.allow_writes();
        let name = self.resolve_new_scope_name(name, "cond");
        let mut context = self.as_new_child(name);
        match context.control_context {
            ControlFlow::CondContext(ref cond) => cond_context.values.extend(cond.values.iter()),
            ControlFlow::WhileContext(ref whileloop) => {
                cond_context.values.extend(whileloop.values.iter())
            }
            _ => {}
        }
        context.control_context = ControlFlow::WhileContext(cond_context);
        context
    }

    fn build_loop(
        &mut self,
        mut pred: WhileCondGraph,
        mut body: WhileBodyGraph,
        loop_vars: &mut [Tensor],
    ) -> Result<Vec<Tensor>> {
        // Let the context know the loop variables so the loop variables
        // would be added in the outer contexts properly.
        self.initialize_values(loop_vars);
        //let real_vars = loop_vars;
        let enter_vars;
        {
            let scope = &mut self.clear_control_dependencies();
            let name = self.own_scope.name.to_str().unwrap();
            enter_vars = loop_vars
                .iter()
                .map(|x| enter(scope, *x, name, ""))
                .collect::<Result<Vec<_>>>()?;
            for x in &enter_vars {
                scope.prevent_feeding(x);
            }
        }

        self.initialize_values(&enter_vars);
        while_context!(mut self).loop_enters = enter_vars.clone();

        /*
        let mut merge_vars = Vec::with_capacity(loop_vars.len());
        for x in loop_vars {
            merge_vars.push(merge(self, vec![*x, *x], "")?.0);
        }
        while_context!(mut self).pivot_for_pred = Some(merge_vars[0]);
        */

        // Build the graph for pred.
        let c = pred(self, loop_vars)?;
        while_context!(mut self).pivot = Some(loop_cond(self, c, "LoopCond")?);
        let switch_vars = loop_vars
            .iter()
            .map(|x| {
                let pivot = while_context!(self).pivot.unwrap();
                switch_ref_or_tensor(self, *x, pivot)
            })
            .collect::<Result<Vec<_>>>()?;

        // Build the graph for the body.
        let mut vars_for_body =
            switch_vars.iter().map(|&(_, x)| self.identity(x, "")).collect::<Result<Vec<_>>>()?;
        while_context!(mut self).pivot_for_body = Some(vars_for_body[0]);
        let body_result = body(self, &mut vars_for_body)?;

        // Add NextIteration and the back edges to complete the loop.
        let _next_vars = enter_vars
            .iter()
            .zip(body_result.iter())
            .map(|(i, v)| add_next_and_back_edge(self, *i, *v))
            .collect::<Result<Vec<_>>>()?;

        // Add the exit ops.
        let exit_vars =
            switch_vars.iter().map(|&(x, _)| exit(self, x, "")).collect::<Result<Vec<_>>>()?;
        while_context!(mut self).loop_exits = exit_vars.clone();

        // Exit the loop.
        self.exit_result(&exit_vars);
        Ok(exit_vars)
    }

    /// Makes the values known to this context.
    fn initialize_values(&mut self, values: &[Tensor]) {
        let outside_values = &mut while_context!(mut self).values;
        *outside_values = HashSet::new();
        for x in values {
            outside_values.insert(x.ident);
        }
    }

    /// Make a list of tensors available in the outer context.
    fn exit_result(&mut self, result: &[Tensor]) {
        let ctxt = while_context!(mut self);
        for e in result {
            ctxt.values.insert(e.ident);
        }
    }
}

fn switch_ref_or_tensor(
    scope: &mut Scope,
    data: Tensor,
    pred: Tensor,
) -> Result<(Tensor, Tensor)> {
    // TODO: add "colocate_with(data)"
    if data.is_ref() {
        ref_switch(scope, data, pred, "RefSwitch")
    } else {
        switch(scope, data, pred, "Switch")
    }
}

fn add_next_and_back_edge(
    scope: &mut Scope,
    m: Tensor,
    v: Tensor,
) -> Result<(Tensor, Tensor)> {
    let next_iter = if v.is_ref() {
        scope.install(RefNextIteration::new(v, "")?)?
    } else {
        scope.install(NextIteration::new(v, "")?)?
    };
    //m.op._update_input(1, v)
    merge(scope, vec![m, next_iter], "")
}

fn loop_cond<S: AsRef<Path>>(
    context: &mut Scope,
    pred: Tensor,
    name: S,
) -> Result<Tensor> {
    if pred.dtype != DataType::Bool {
        return Err(Error::from(ErrorKind::Stub));
    }
    if pred.get_shape(context) != Shape::from(Some(vec![])) {
        let msg = format!(
            "tf: expected shape `[]` for pred Tensor on `cond` op call, found shape: `{:?}`",
            pred.get_shape(context)
        );
        return Err(Error::from(msg));
    }
    context.install(LoopCond::new(pred, name)?)
}

/// Forwards the input to the output.
///
/// This operator represents the loop termination condition used by the
/// "pivot" switches of a loop.
add_new_op!(LoopCond,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: LoopCond, Init: []);],
    digest: [DEFAULT_DIGEST: LoopCond, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

/// Creates or finds a child frame, and makes `data` available to it.
///
/// The unique `frame_name` is used by the `Executor` to identify frames. If
/// `is_constant` is true, `data` is a constant in the child frame; otherwise
/// it may be changed in the child frame. At most `parallel_iterations`
/// iterations are run in parallel in the child frame.
fn enter<S>(scope: &mut Scope, data: Tensor, frame_name: &str, name: S) -> Result<Tensor>
where
    S: AsRef<Path>,
{
    if data.is_ref() {
        let fname = &[frame_name];
        let enter = RefEnter::new(data, name)?.frame_name(fname);
        scope.install(enter)
    } else {
        let fname = &[frame_name];
        let enter = Enter::new(data, name)?.frame_name(fname);
        scope.install(enter)
    }
}

add_new_op!(Enter,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: Enter, Init: []);],
    digest: [DEFAULT_DIGEST: Enter, INPUT0],
    extra_funcs: [
        fn frame_name(mut self, val: &'a [&str]) -> Self {
            self.attributes.push(("frame_name", false, Attribute::String(val)));
            self
        }

        #[allow(dead_code)]
        fn is_constant(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("is_constant", false, Attribute::Bool(val)));
            self
        }

        #[allow(dead_code)]
        fn parallel_iterations(mut self, val: &'a [i64]) -> Self {
            self.attributes.push(("parallel_iterations", false, Attribute::Int(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);

add_new_op!(RefEnter,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: RefEnter, Init: []);],
    digest: [DEFAULT_DIGEST: RefEnter, INPUT0],
    extra_funcs: [
        fn frame_name(mut self, val: &'a [&str]) -> Self {
            self.attributes.push(("frame_name", false, Attribute::String(val)));
            self
        }

        #[allow(dead_code)]
        fn is_constant(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("is_constant", false, Attribute::Bool(val)));
            self
        }

        #[allow(dead_code)]
        fn parallel_iterations(mut self, val: &'a [i64]) -> Self {
            self.attributes.push(("parallel_iterations", false, Attribute::Int(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);

/// Exits the current frame to its parent frame.
///
/// Exit makes its input `data` available to the parent frame.
fn exit<S>(scope: &mut Scope, data: Tensor, name: S) -> Result<Tensor>
where
    S: AsRef<Path>,
{
    if data.is_ref() {
        scope.install(RefExit::new(data, name)?)
    } else {
        scope.install(Exit::new(data, name)?)
    }
}

add_new_op!(Exit,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: Exit, Init: []);],
    digest: [DEFAULT_DIGEST: Exit, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

add_new_op!(RefExit,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: RefExit, Init: []);],
    digest: [DEFAULT_DIGEST: RefExit, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

add_new_op!(NextIteration,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: NextIteration, Init: []);],
    digest: [DEFAULT_DIGEST: NextIteration, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

add_new_op!(RefNextIteration,
    constructor: [add_new_op!(UNARY CONSTRUCTOR: RefNextIteration, Init: []);],
    digest: [DEFAULT_DIGEST: RefNextIteration, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

/// Create an op that groups multiple operations.
///
/// When this op finishes, all ops in input have finished. This op has no output.
/// This is useful for control flow in conjunction with `control_dependencies` for example.
#[derive(Debug, Clone, Copy)]
pub struct Group(NodeIdent);

impl Group {
    pub fn new<Id, S>(scope: &mut Scope, ops: &[Id], name: S) -> Result<Group>
    where
        Id: GetIdent,
        S: AsRef<str>,
    {
        let graph = &mut *scope.graph.borrow_mut();
        let registry = &*scope.registry.borrow();
        let mut ctrl_ops = Vec::with_capacity(ops.len());
        for x in ops {
            let ident: NodeIdent = x.get_ident();
            let r = &registry[&ident].data_origin.0;
            ctrl_ops.push(r);
        }

        const OP: IdType = IdType::Operation("Group");
        let name = scope.resolve_tensor_name(Some(Path::new(name.as_ref())), OP, false)?;
        let finished = no_op_(graph, name.to_str().unwrap(), ctrl_ops)?;

        let ident = NodeIdent::new();
        let ops_reg = &mut *scope.ops.borrow_mut();
        ops_reg.insert(ident, finished.clone());

        Ok(Group(ident))
    }
}

impl Into<NodeIdent> for Group {
    fn into(self) -> NodeIdent {
        self.0
    }
}

impl GetIdent for Group {
    fn get_ident(&self) -> NodeIdent {
        self.0
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

#[cfg(test)]
mod test {
    #![allow(unused_imports)]
    use super::*;

    #[test]
    fn test_where_cond() {
        let mut context = Scope::new();
        let x = context.constant(&[4_i32, 2, 4], &[3], "x").unwrap();
        let y = context.constant(&[2_i32, 4, 2], &[3], "y").unwrap();
        let cond = greater(&mut context, x, y, "").unwrap();
        let op = where_cond(&mut context, cond, None, None, "").unwrap();
        let results = test_suite!(run_op: [op]; context, input: {});
        test_suite!(results; assert: {[0;Int64] == [0_i64, 2]});
        test_suite!(results; assert_len: {[0;Int64] == 2});
        //println!("{:?}", Vec::from(&*results.pop().unwrap().unwrap_i64()));
    }

    #[test]
    #[ignore]
    fn test_assert_eq() {
        let mut context = Scope::new();
        let x = context.constant(&[2_i32], &[] as &[i32] as &[i32], "x").unwrap();
        let y = context.constant(&[2_i32], &[] as &[i32] as &[i32], "y").unwrap();
        let assert = assert_eq(&mut context, x, y, None, None, "").unwrap();
        //context.install(assert.clone()).unwrap();
        let results = test_suite!(run_op: [assert]; context, input: {});
    }

    #[test]
    #[ignore]
    fn test_assert_greater() {
        let mut context = Scope::new();
        let x = context.constant(&[3_i32], &[] as &[i32], "x").unwrap();
        let y = context.constant(&[2_i32], &[] as &[i32], "y").unwrap();
        let assert = assert_greater(&mut context, x, y, None, None, "").unwrap();
        //context.install(assert.clone()).unwrap();
        let results = test_suite!(run_op: [assert]; context, input: {});
    }

    #[test]
    fn test_cond() {
        use super::assign;
        let mut context = Scope::new();
        let var: Tensor =
            context.get_variable(Some(DataType::Int32), Some(&[] as &[i32]), "").unwrap().into();
        let x = context.constant(&[2_i32], &[] as &[i32], "").unwrap();
        let y = context.constant(&[5_i32], &[] as &[i32], "").unwrap();

        let f1 = Box::new(
            move |scope: &mut Scope| -> Result<Vec<Tensor>> {
                Ok(vec![assign(scope, var, x, true, "")?])
            },
        );
        let f2 = Box::new(
            move |scope: &mut Scope| -> Result<Vec<Tensor>> {
                Ok(vec![assign(scope, var, y, true, "")?])
            },
        );

        let pred = less(&mut context, y, x, "").unwrap();
        let op = cond(&mut context, pred, f1, f2, "").unwrap()[0];
        let r = test_suite!(run_op: [op, var]; context, input: {});
        test_suite!(r; assert_len: {[0;Int32] == 1});
        test_suite!(r; assert: {[0;Int32] == [5_i32], [1;Int32] == [5_i32]});

        let f1 = Box::new(
            move |mut scope: &mut Scope| -> Result<Vec<Tensor>> {
                let mult_x = scope.constant(&[10_i32], &[] as &[i32], "")?;
                let add_x = scope.constant(&[20_i32], &[] as &[i32], "")?;
                let v = multiply(&mut scope, x, mult_x, "")?;
                let v = add(&mut scope, v, add_x, "")?;
                Ok(vec![assign(scope, var, v, true, "")?])
            },
        );
        let f2 = Box::new(
            move |scope: &mut Scope| -> Result<Vec<Tensor>> {
                Ok(vec![assign(scope, var, y, true, "")?])
            },
        );

        let pred = less(&mut context, x, y, "").unwrap();
        let op = cond(&mut context, pred, f1, f2, "").unwrap()[0];
        let r = test_suite!(run_op: [op, var]; context, input: {});
        test_suite!(r; assert_len: {[0;Int32] == 1});
        test_suite!(r; assert: {[0;Int32] == [40_i32], [1;Int32] == [40_i32]});
    }

    #[ignore]
    #[test]
    fn test_while_loop() {
        let mut context = Scope::new();
        let x = context.constant(&[0_i32], &[] as &[i32], "").unwrap();

        let pred = Box::new(move |scope: &mut Scope, loop_vars: &mut [Tensor]| {
            let y = scope.constant(&[10_i32], &[] as &[i32], "").unwrap();
            let x = loop_vars[0];
            less(scope, x, y, "")
        });

        let body = Box::new(move |scope: &mut Scope,
              loop_vars: &mut [Tensor]|
              -> Result<Vec<Tensor>> {
            let y = scope.constant(&[1_i32], &[] as &[i32], "").unwrap();
            let x = loop_vars[0];
            Ok(vec![add(scope, x, y, "")?])
        });

        let op = while_loop(&mut context, pred, body, &mut [x.into()], "").unwrap()[0];
        let r = test_suite!(run_op: [op]; context, input: {});
        test_suite!(r; assert_len: {[0;Int32] == 1});
        test_suite!(r; assert: {[0;Int32] == [10_i32]});
    }
}

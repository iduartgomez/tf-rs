use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use tf::TensorType;

use {Graph, Output};
use super::{add_control_input, Attribute, Constant, ControlOp, ControlOpKind, DataType, Error,
            ErrorKind, Function, GetOp, GradFunc, IdType, NodeIdent, Operation, OperationData,
            Result, Shape, ShapeOps, Tensor, TensorContent, TensorData, TypedTensor, Variable};
use ops::{array_ops, control_flow_ops, init_ops, ControlFlow};

const DEFAULT_GRAPH_SEED: i64 = 87_654_321;

/// Master context manager for building TensorFlow graphs and managing session execution.
#[derive(Debug)]
pub struct Scope {
    /// tensors of tensors and ops
    pub(crate) tensors: Rc<RefCell<HashMap<NodeIdent, TensorData>>>,
    pub(crate) ops: Rc<RefCell<HashMap<NodeIdent, OperationData>>>,
    /// owned graph for building
    pub(crate) graph: Rc<RefCell<Graph>>,
    pub(crate) scopes: Rc<RefCell<InternScope>>,
    pub(crate) own_scope: InternScope,
    pub(crate) control_context: ControlFlow,
    reuse_variable: bool,
    not_variable_scope: bool,
    ignore_deps: bool,
    locked: Rc<RefCell<bool>>,
    parent_lock: Option<Rc<RefCell<bool>>>,
    seed: Option<i64>,
}

impl Scope {
    pub fn new() -> Scope {
        let own_scope = InternScope::new(PathBuf::new());
        Scope {
            scopes: Rc::new(RefCell::new(own_scope.clone())),
            own_scope,
            tensors: Rc::new(RefCell::new(HashMap::new())),
            ops: Rc::new(RefCell::new(HashMap::new())),
            graph: Rc::new(RefCell::new(Graph::new())),
            control_context: ControlFlow::None,
            reuse_variable: false,
            not_variable_scope: false,
            ignore_deps: false,
            locked: Rc::new(RefCell::new(false)),
            parent_lock: None,
            seed: None,
        }
    }

    /// If `reuse`, don't return error on name collision with an existing identifier.
    pub(crate) fn resolve_name(
        &self,
        name: Option<&Path>,
        kind: IdType,
        reuse: bool,
    ) -> Result<PathBuf> {
        let name = if let Some(given_name) = name {
            if given_name.to_str().unwrap() == "" {
                return self.resolve_name(None, kind, reuse);
            }
            let mut name = self.own_scope.name.join(given_name);
            if !reuse {
                match kind {
                    // check both constant and var containers in case a 'variable' was named
                    // like a constant and viceversa
                    IdType::Constant | IdType::Variable => {
                        if self.own_scope.name_exists(&name) {
                            return Err(Error::from(ErrorKind::Stub));
                        }
                    }
                    IdType::Placeholder => {
                        if self.tensors
                            .borrow()
                            .values()
                            .find(|x| &x.full_name == &name)
                            .is_some()
                        {
                            return Err(Error::from(ErrorKind::Stub));
                        }
                    }
                    IdType::Operation(_) => {
                        if self.own_scope
                            .ops
                            .iter()
                            .find(|&&(ref x, _)| x == &name)
                            .is_some()
                        {
                            name = self.own_scope.name.join(format!(
                                "{}_{}",
                                given_name.display(),
                                self.own_scope.ops.len()
                            ))
                        }
                    }
                }
            }
            name
        } else {
            let name = match kind {
                IdType::Constant => format!("Constant_{}", self.own_scope.constants.len()),
                IdType::Operation(name) => format!("{}_{}", name, self.own_scope.ops.len()),
                IdType::Placeholder => format!("Placeholder_{}", self.tensors.borrow().len()),
                IdType::Variable => format!("Variable_{}", self.own_scope.variables.len()),
            };
            self.own_scope.name.join(name)
        };
        Ok(name)
    }

    pub(crate) fn resolve_scope_name<S: AsRef<Path>>(
        &self,
        name: S,
        default_prefix: &str,
    ) -> PathBuf {
        let new_name;
        if name_cmp!(name, "") {
            new_name = self.own_scope.name.join(format!(
                "{}_{}",
                default_prefix,
                self.own_scope.inner_scopes.len()
            ));
            new_name
        } else {
            self.own_scope.name.join(format!(
                "{}_{}",
                name.as_ref().display(),
                self.own_scope.inner_scopes.len()
            ))
        }
    }

    pub(crate) fn as_new_child(&mut self, name: PathBuf) -> Scope {
        // FIXME: use is_op to avoid duplicates
        let own_scope = if name == self.own_scope.name {
            // it's just a mirror of self, clone all the scope data,
            // writes on self will be locked, but after drop of the new scope,
            // they will be added from the mirror.
            self.own_scope.clone()
        } else {
            // find if this scope already exists
            let global = &mut self.scopes.borrow_mut().inner_scopes;
            let child = if let Some(scope) = find_parent_scope(global, &name, 0) {
                // an inmediate parent for this scope exists
                let mut add_new = false;
                let child;
                {
                    // initialize the new scope:
                    if let Some(found_child) = scope.get_child(&name) {
                        // this scope already existed, just return a copy of it
                        child = found_child.clone();
                    } else {
                        // this scope didn't exist previously, create it
                        child = InternScope::new(name.clone());
                        add_new = true;
                    };
                }
                if add_new {
                    // it was new, so add it to it's parent
                    scope.add_scope(child.clone());
                }
                Some(child)
            } else {
                // there was no inmediate parent for this scope
                None
            };
            if let Some(child) = child {
                child
            } else {
                // there was no inmediate parent for this scope, we are at root
                // create a new scope, add it to to 'scopes' and add it to 'self'
                let child = InternScope::new(name);
                global.push(Box::new(child.clone()));
                self.own_scope.add_scope(child.clone());
                child
            }
        };
        // lock self, new writes are only allowed on the returned subscope
        // if pertinent, when the new subscope is dropped, changes will be pulled
        // back to 'self' and the lock will be dropped.
        *self.locked.borrow_mut() = true;

        Scope {
            scopes: self.scopes.clone(),
            own_scope,
            tensors: self.tensors.clone(),
            graph: self.graph.clone(),
            control_context: self.control_context.clone(),
            reuse_variable: false,
            not_variable_scope: false,
            ignore_deps: false,
            locked: Rc::new(RefCell::new(false)),
            parent_lock: Some(self.locked.clone()),
            ops: self.ops.clone(),
            seed: self.seed.clone(),
        }
    }

    /// If this scope is locked, don't allow any write operations and panic at runtime.
    pub(crate) fn allow_writes(&self) {
        if *self.locked.borrow() {
            let scope_name = format!("{}", self.own_scope.name.display());
            let scope_name = match scope_name.as_str() {
                "" => format!("{}", "root"),
                _ => scope_name,
            };
            panic!(format!(
                "tried to write at scope `{}` while an other child scope was open",
                scope_name
            ))
        }
    }

    pub(crate) fn locked(&self) -> bool {
        self.parent_lock.is_some()
    }

    /// Install an operation within the current context.
    ///
    /// Returns the output of the operation.
    #[doc(hidden)]
    pub fn install<'a, T>(&mut self, op: T) -> Result<T::Outputs>
    where
        T: Operation<'a>,
    {
        self.allow_writes();
        let processed_inputs = self.process_op_inputs(&op);
        let new_op = {
            let graph = &mut *self.graph.borrow_mut();
            let root = &*self.scopes.borrow();

            let op_name = IdType::Operation(op.get_op_type_name());
            let name = if let Some(name) = op.get_op_name() {
                self.resolve_name(Some(name), op_name, false)?
            } else {
                self.resolve_name(None, op_name, false)?
            };
            let mut new_op = graph.new_operation(op.get_op_type_name(), name.to_str().unwrap())?;

            for &(name, is_list, ref attribute) in op.fetch_attributes() {
                match *attribute {
                    Attribute::String(ref val) => {
                        if val.len() > 1 || is_list {
                            new_op.set_attr_string_list(name, val)?;
                        } else if !val.is_empty() {
                            new_op.set_attr_string(name, val[0])?;
                        }
                    }
                    Attribute::Int(ref val) => {
                        if val.len() > 1 || is_list {
                            new_op.set_attr_int_list(name, val)?;
                        } else if !val.is_empty() {
                            new_op.set_attr_int(name, val[0])?;
                        }
                    }
                    Attribute::Float(ref val) => {
                        if val.len() > 1 || is_list {
                            new_op.set_attr_float_list(name, val)?;
                        } else if !val.is_empty() {
                            new_op.set_attr_float(name, val[0])?;
                        }
                    }
                    Attribute::Bool(ref val) => {
                        if val.len() > 1 || is_list {
                            new_op.set_attr_bool_list(name, val)?;
                        } else if !val.is_empty() {
                            new_op.set_attr_bool(name, val[0])?;
                        }
                    }
                    Attribute::Type(ref val) => {
                        if val.len() > 1 || is_list {
                            new_op.set_attr_type_list(name, val)?;
                        } else if !val.is_empty() {
                            new_op.set_attr_type(name, val[0])?;
                        }
                    }
                    Attribute::Shape(ref val) => {
                        if val.len() > 1 || is_list {
                            new_op.set_attr_shape_list(name, val)?;
                        } else if !val.is_empty() {
                            new_op.set_attr_shape(name, &val[0])?;
                        }
                    }
                    Attribute::Tensor(ref val) => {
                        if val.len() > 1 || is_list {
                            TensorContent::set_tensor_list_attr(&mut new_op, name, val)?;
                        } else if !val.is_empty() {
                            TensorContent::set_tensor_attr(&mut new_op, name, val)?;
                        }
                    }
                }
            }
            for input in processed_inputs {
                match input {
                    OpInput::Single(val) => new_op.add_input(val),
                    OpInput::List(val) => new_op.add_input_list(&val),
                }
            }
            let control_inputs = root.control_dependencies.iter().map(|x| &x.finished);
            add_control_input(&mut new_op, control_inputs);
            new_op.finish()?
        };
        op.digest(self, new_op)
    }

    #[cfg(test)]
    pub(crate) fn get_src_op<Op: GetOp>(&self, op: Op) -> (OperationData, i32) {
        let &TensorData {
            data_origin: (ref op, idx),
            ..
        } = &self.tensors.borrow()[op.get_op()];
        (op.clone(), idx)
    }

    fn process_op_inputs<'a, T>(&mut self, op: &T) -> Vec<OpInput>
    where
        T: Operation<'a>,
    {
        let reg_c = self.tensors.clone();
        let mut inputs = vec![];

        fn input_ls<'a>(
            input_lists: &mut ::std::slice::Iter<'a, (usize, Vec<Tensor>)>,
        ) -> (Option<usize>, Option<&'a [Tensor]>) {
            if let Some(&(ref idx, ref list)) = input_lists.next() {
                (Some(*idx), Some(list))
            } else {
                (None, None)
            }
        }

        fn iter_input_ls<'a>(
            input_lists: &mut ::std::slice::Iter<'a, (usize, Vec<Tensor>)>,
            reg: &HashMap<NodeIdent, TensorData>,
            all_inputs: &mut Vec<OpInput>,
            args_index: &mut usize,
            ls_idx: &mut Option<usize>,
            current_list: &mut Option<&'a [Tensor]>,
        ) {
            while ls_idx.is_some() && (ls_idx.as_ref().unwrap() == args_index) {
                let mut inputs = vec![];
                for tensor in current_list.unwrap() {
                    let data = &reg[&tensor.ident];
                    inputs.push(Output {
                        operation: data.data_origin.0.clone(),
                        index: data.data_origin.1,
                    })
                }
                all_inputs.push(OpInput::List(inputs));
                let (i, ls) = input_ls(input_lists);
                *ls_idx = i;
                *current_list = ls;
                *args_index += 1;
            }
        }

        let mut args_index = 0_usize;
        let input_lists = &mut op.fetch_input_lists().iter();
        let (mut ls_idx, mut current_list) = input_ls(input_lists);
        for input in op.fetch_inputs().into_iter() {
            {
                iter_input_ls(
                    input_lists,
                    &*reg_c.borrow(),
                    &mut inputs,
                    &mut args_index,
                    &mut ls_idx,
                    &mut current_list,
                )
            }
            let (operation, index) = match self.control_context {
                ControlFlow::CondContext(_) => {
                    if !self.control_context.get_cond().unwrap().new_switch {
                        use self::control_flow_ops::CondContextInterface;
                        let output = self.process_output_tensor(input);
                        let data = &reg_c.borrow()[&output.ident];
                        (data.data_origin.0.clone(), data.data_origin.1)
                    } else {
                        let data = &reg_c.borrow()[&input.ident];
                        (data.data_origin.0.clone(), data.data_origin.1)
                    }
                }
                ControlFlow::None | ControlFlow::WhileContext(_) => {
                    let data = &reg_c.borrow()[&input.ident];
                    (data.data_origin.0.clone(), data.data_origin.1)
                }
            };
            inputs.push(OpInput::Single(Output { operation, index }));
            args_index += 1;
        }
        if current_list.is_some() {
            iter_input_ls(
                input_lists,
                &*reg_c.borrow(),
                &mut inputs,
                &mut args_index,
                &mut ls_idx,
                &mut current_list,
            );
        }
        inputs
    }

    /// Returns a context manager for defining ops that creates variables (layers).
    ///
    /// If the name argument is an empty string the name will be autogenerated.
    pub fn variable_scope<S>(
        &mut self,
        name: S,
        default_name: Option<S>,
        reuse: Option<bool>,
    ) -> Result<Scope>
    where
        S: AsRef<Path>,
    {
        self.allow_writes();
        if name_cmp!(name, "") && default_name.is_none() {
            return Err(Error::from(
                "If default_name is None then name is required not be empty.",
            ));
        }
        let name = if let Some(default_name) = default_name {
            self.resolve_scope_name(name, default_name.as_ref().to_str().unwrap())
        } else {
            self.resolve_scope_name(name, "")
        };
        let mut scope = self.as_new_child(name);
        if let Some(value) = reuse {
            scope.reuse_variable = value;
        }
        Ok(scope)
    }

    /// Returns a context manager for use when defining an op.
    ///
    /// If the name argument is an empty string the name will be autogenerated.
    pub fn name_scope<S>(&mut self, name: S, default: Option<S>) -> Scope
    where
        S: AsRef<Path>,
    {
        self.allow_writes();
        let name = if let Some(default) = default {
            self.resolve_scope_name(name, default.as_ref().to_str().unwrap())
        } else {
            self.resolve_scope_name(name, "op_scope")
        };
        let mut scope = self.as_new_child(name);
        scope.not_variable_scope = true;
        scope
    }

    /// Gets an existing variable with these parameters or create a new one.
    ///
    /// This function prefixes the name with the current variable scope and performs reuse checks.
    ///
    /// Returns an error when creating a new variable and shape is not declared or when violating
    /// reuse during variable creation.
    ///
    /// Reuse is set during `variable_scope` creation.
    pub fn get_variable<Sh, S>(
        &mut self,
        dtype: Option<DataType>,
        shape: Option<Sh>,
        name: S,
    ) -> Result<Variable>
    where
        S: AsRef<Path>,
        Sh: ShapeOps,
    {
        fn get_initial_value(
            g: &mut Graph,
            dtype: DataType,
            n: &str,
            shape: &[u64],
        ) -> Result<OperationData> {
            let op_data = match dtype {
                DataType::Bool => array_ops::constant(g, n, TypedTensor::<bool>::new(shape), &[])?,
                DataType::Double => array_ops::constant(g, n, TypedTensor::<f64>::new(shape), &[])?,
                DataType::Float => array_ops::constant(g, n, TypedTensor::<f32>::new(shape), &[])?,
                DataType::Int32 => array_ops::constant(g, n, TypedTensor::<i32>::new(shape), &[])?,
                DataType::UInt8 => array_ops::constant(g, n, TypedTensor::<u8>::new(shape), &[])?,
                DataType::Int16 => array_ops::constant(g, n, TypedTensor::<i16>::new(shape), &[])?,
                DataType::Int8 => array_ops::constant(g, n, TypedTensor::<i8>::new(shape), &[])?,
                DataType::Int64 => array_ops::constant(g, n, TypedTensor::<i64>::new(shape), &[])?,
                DataType::String => {
                    array_ops::constant(g, n, TypedTensor::<String>::new(shape), &[])?
                }
                DataType::QUInt8 => {
                    array_ops::constant(g, n, TypedTensor::<::QUInt8>::new(shape), &[])?
                }
                DataType::QInt16 => {
                    array_ops::constant(g, n, TypedTensor::<::QInt16>::new(shape), &[])?
                }
                DataType::QUInt16 => {
                    array_ops::constant(g, n, TypedTensor::<::QUInt16>::new(shape), &[])?
                }
                DataType::QInt32 => {
                    array_ops::constant(g, n, TypedTensor::<::QInt32>::new(shape), &[])?
                }
                DataType::BFloat16 => {
                    array_ops::constant(g, n, TypedTensor::<::BFloat16>::new(shape), &[])?
                }
                DataType::Complex64 => {
                    array_ops::constant(g, n, TypedTensor::<::Complex32>::new(shape), &[])?
                }
                DataType::Complex128 => {
                    array_ops::constant(g, n, TypedTensor::<::Complex64>::new(shape), &[])?
                }
                _ => return Err(Error::from(ErrorKind::Stub)),
            };
            Ok(op_data)
        }

        self.allow_writes();
        let new_var = self.resolve_name(Some(name.as_ref()), IdType::Variable, false)?;

        let var = if self.not_variable_scope {
            // use scopes/root
            // FIXME: don't default to root
            self.scopes
                .borrow()
                .variables
                .binary_search_by(|&(ref name, _)| name.cmp(&new_var))
        } else {
            (&self.own_scope.variables).binary_search_by(|&(ref name, _)| name.cmp(&new_var))
        };
        if self.reuse_variable {
            // find and return an existing variable or return error
            if let Ok(idx) = var {
                Ok(self.own_scope.variables[idx].1)
            } else {
                Err(Error::from(ErrorKind::Stub))
            }
        } else if var.is_err() && !self.reuse_variable {
            // try making a new variable
            let rank_info = if let Some(shape) = shape {
                shape.to_shape()
            } else {
                // shape for a new variable must be specified
                return Err(Error::from(ErrorKind::Stub));
            };
            let dtype = if let Some(dtype) = dtype {
                dtype
            } else {
                return Err(Error::from("dtype not specified"));
            };

            let ident = NodeIdent::new();
            let init;
            let init_ident;
            let var;
            {
                let graph = &mut *self.graph.borrow_mut();
                let tensors = &mut *self.tensors.borrow_mut();
                let ops = &mut self.ops.borrow_mut();

                // variable op, not initialized!
                var = {
                    let deps = match self.control_context {
                        ControlFlow::CondContext(ref cond) => {
                            vec![&tensors[&cond.pivot.ident].data_origin.0]
                        }
                        ControlFlow::WhileContext(ref cond) => {
                            if cond.pivot_for_body.is_some() {
                                vec![
                                    &tensors[&cond.pivot_for_body.as_ref().unwrap().ident]
                                        .data_origin
                                        .0,
                                ]
                            } else {
                                vec![
                                    &tensors[&cond.pivot_for_pred.as_ref().unwrap().ident]
                                        .data_origin
                                        .0,
                                ]
                            }
                        }
                        ControlFlow::None => vec![],
                    };
                    init_ops::variable_(graph, new_var.to_str().unwrap(), dtype, &rank_info, deps)?
                };
                let var_op_id = NodeIdent::new();
                ops.insert(var_op_id.clone(), var.clone());

                // initializer
                init = {
                    let initial_value = {
                        let init_name = new_var.join("init_value");
                        get_initial_value(
                            graph,
                            dtype,
                            init_name.to_str().unwrap(),
                            &rank_info.definition_u64().unwrap(),
                        )?
                    };
                    let data_origin_id = NodeIdent::new();
                    ops.insert(data_origin_id.clone(), initial_value.clone());

                    init_ident = NodeIdent::new();
                    tensors.insert(
                        init_ident.clone(),
                        TensorData::new(
                            TensorData::name_builder(new_var.join("init_value"), 0),
                            dtype,
                            IdType::Constant,
                            (initial_value.clone(), 0),
                            data_origin_id,
                            rank_info.clone(),
                        ),
                    );

                    let init = &[
                        init_ops::assign_(
                            graph,
                            new_var.join("init").to_str().unwrap(),
                            var.clone(),
                            (initial_value, 0),
                            true,
                        )?,
                    ];
                    // get previous existing control dependencies
                    let cd = &self.scopes.borrow().control_dependencies;
                    let control_inputs = cd.iter().map(|x| &x.finished).chain(init);
                    control_flow_ops::no_op_(
                        graph,
                        new_var.join("init_ctrl").to_str().unwrap(),
                        control_inputs,
                    )?
                };
                // Register variable data.
                tensors.insert(
                    ident,
                    TensorData::new(
                        new_var.clone(),
                        dtype,
                        IdType::Variable,
                        (var, 0),
                        var_op_id,
                        rank_info,
                    ),
                );
                self.scopes
                    .borrow_mut()
                    .control_dependencies
                    .push_front(ControlOp {
                        ident: NodeIdent::new(),
                        finished: init,
                        kind: ControlOpKind::VarInitializer,
                    });
            }
            Ok(self._make_var_handle(
                ident,
                init_ident,
                TensorData::name_builder(new_var.clone(), 0),
                dtype,
            ))
        } else {
            Err(Error::from(ErrorKind::Stub))
        }
    }

    /// Create a new_variable with the given initializer.
    pub fn get_variable_with_initializer<S, Op>(
        &mut self,
        initializer: Op,
        validate_shape: bool,
        name: S,
    ) -> Result<Variable>
    where
        S: AsRef<Path>,
        Op: GetOp,
    {
        self.allow_writes();
        let new_var = self.resolve_name(Some(name.as_ref()), IdType::Variable, false)?;
        //let initializer = initializer.get_op();

        let var = if self.not_variable_scope {
            // use scopes/root
            // FIXME: don't default to root
            self.scopes
                .borrow()
                .variables
                .binary_search_by(|&(ref name, _)| name.cmp(&new_var))
        } else {
            (&self.own_scope.variables).binary_search_by(|&(ref name, _)| name.cmp(&new_var))
        };

        if self.reuse_variable {
            // find and return an existing variable or return error
            if let Ok(idx) = var {
                Ok(self.own_scope.variables[idx].1)
            } else {
                Err(Error::from(ErrorKind::Stub))
            }
        } else if var.is_err() && !self.reuse_variable {
            let ident = NodeIdent::new();
            let init;
            let var;
            let dtype;
            let rank_info;
            {
                let graph = &mut *self.graph.borrow_mut();
                let tensors = &mut *self.tensors.borrow_mut();
                let ops = &mut self.ops.borrow_mut();

                let initializer = {
                    if let Some(index) = initializer.source_index() {
                        // is an op output tensor ident
                        let initializer = tensors
                            .get(&initializer.get_op())
                            .ok_or(Error::from(ErrorKind::OpNotFound))?;
                        rank_info = graph.tensor_shape(Output {
                            operation: initializer.data_origin.0.clone(),
                            index,
                        })?;
                        dtype = initializer.dtype;
                        initializer.data_origin.clone()
                    } else {
                        // is an op ident, try find its output
                        let initializer = ops.get(&initializer.get_op())
                            .ok_or(Error::from(ErrorKind::OpNotFound))?;
                        if initializer.num_outputs() != 1 {
                            return Err(Error::from(ErrorKind::from("var initializer op has more than one output and index is unspecified")));
                        }
                        rank_info = graph.tensor_shape(Output {
                            operation: initializer.clone(),
                            index: 0,
                        })?;
                        dtype = initializer.output_type(0);
                        (initializer.clone(), 0)
                    }
                };

                // variable op, not initialized!
                var = {
                    let deps = match self.control_context {
                        ControlFlow::CondContext(ref cond) => {
                            vec![&tensors[&cond.pivot.ident].data_origin.0]
                        }
                        ControlFlow::WhileContext(ref cond) => {
                            if cond.pivot_for_body.is_some() {
                                vec![
                                    &tensors[&cond.pivot_for_body.as_ref().unwrap().ident]
                                        .data_origin
                                        .0,
                                ]
                            } else {
                                vec![
                                    &tensors[&cond.pivot_for_pred.as_ref().unwrap().ident]
                                        .data_origin
                                        .0,
                                ]
                            }
                        }
                        ControlFlow::None => vec![],
                    };
                    init_ops::variable_(graph, new_var.to_str().unwrap(), dtype, &rank_info, deps)?
                };
                let var_op_id = NodeIdent::new();
                ops.insert(var_op_id.clone(), var.clone());

                // assign op
                init = {
                    let init = &[
                        init_ops::assign_(
                            graph,
                            new_var.join("init").to_str().unwrap(),
                            var.clone(),
                            initializer,
                            validate_shape,
                        )?,
                    ];

                    // get previous existing control dependencies
                    let cd = &self.scopes.borrow().control_dependencies;
                    let control_inputs = cd.iter().map(|x| &x.finished).chain(init);
                    control_flow_ops::no_op_(
                        graph,
                        new_var.join("init_ctrl").to_str().unwrap(),
                        control_inputs,
                    )?
                };

                // Register variable data
                tensors.insert(
                    ident,
                    TensorData::new(
                        new_var.clone(),
                        dtype,
                        IdType::Variable,
                        (var, 0),
                        var_op_id,
                        rank_info,
                    ),
                );
                self.scopes
                    .borrow_mut()
                    .control_dependencies
                    .push_front(ControlOp {
                        ident: NodeIdent::new(),
                        finished: init,
                        kind: ControlOpKind::VarInitializer,
                    });
            }
            Ok(self._make_var_handle(
                ident,
                *initializer.get_op(),
                TensorData::name_builder(new_var, 0),
                dtype,
            ))
        } else {
            Err(Error::from(ErrorKind::Stub))
        }
    }

    fn _make_var_handle(
        &mut self,
        ident: NodeIdent,
        initializer: NodeIdent,
        new_var: PathBuf,
        dtype: DataType,
    ) -> Variable {
        // make handle
        let var = Variable {
            ident,
            dtype,
            initializer,
            idx: 0,
        };
        if !self.not_variable_scope {
            // use local
            self.own_scope.variables.push((new_var, var));
            var
        } else {
            // use scopes
            self.scopes.borrow_mut().variables.push((new_var, var));
            var
        }
    }

    /// Create a new 'constant' tensor with given values and shape.
    pub fn constant<Sh, T, S>(&mut self, value: &[T], shape: Sh, name: S) -> Result<Constant>
    where
        S: AsRef<Path>,
        T: TensorType,
        Sh: ShapeOps,
    {
        self.allow_writes();
        let graph = &mut *self.graph.borrow_mut();
        let tensors = &mut *self.tensors.borrow_mut();
        let ops = &mut *self.ops.borrow_mut();

        let full_name = self.resolve_name(Some(name.as_ref()), IdType::Constant, false)?;
        let ident = NodeIdent::new();

        let shape = &shape
            .definition_u64()
            .ok_or(Error::from(ErrorKind::UndefinedTensorShape))?;

        let data_origin = {
            let cd = &self.scopes.borrow().control_dependencies;
            match self.control_context {
                ControlFlow::CondContext(ref cond) => {
                    let pivot = vec![&tensors[&cond.pivot.ident].data_origin.0];
                    array_ops::constant(
                        graph,
                        full_name.to_str().unwrap(),
                        to_typed_tensor![value; shape],
                        cd.iter().map(|x| &x.finished).chain(pivot),
                    )?
                }
                ControlFlow::WhileContext(ref cond) => {
                    let pivot = if cond.pivot_for_body.is_some() {
                        vec![
                            &tensors[&cond.pivot_for_body.as_ref().unwrap().ident]
                                .data_origin
                                .0,
                        ]
                    } else {
                        vec![
                            &tensors[&cond.pivot_for_pred.as_ref().unwrap().ident]
                                .data_origin
                                .0,
                        ]
                    };
                    array_ops::constant(
                        graph,
                        full_name.to_str().unwrap(),
                        to_typed_tensor![value; shape],
                        cd.iter().map(|x| &x.finished).chain(pivot),
                    )?
                }
                ControlFlow::None => array_ops::constant(
                    graph,
                    full_name.to_str().unwrap(),
                    to_typed_tensor![value; shape],
                    cd.iter().map(|x| &x.finished),
                )?,
            }
        };
        let data_origin_id = NodeIdent::new();
        ops.insert(data_origin_id.clone(), data_origin.clone());

        let dtype = data_origin.output_type(0);
        let data = TensorData::new(
            full_name.clone(),
            dtype,
            IdType::Constant,
            (data_origin.clone(), 0),
            data_origin_id.clone(),
            graph.tensor_shape(Output {
                operation: data_origin,
                index: 0,
            })?,
        );
        let t_name = data.get_name();
        tensors.insert(ident, data);

        self.own_scope.constants.push((t_name, ident));
        Ok(Constant {
            ident,
            origin_op: data_origin_id,
            dtype,
        })
    }

    /// Inserts a placeholder for a tensor that will be always fed.
    ///
    /// _Important:_ This tensor will produce an error if evaluated. Its value must be fed to the
    /// client session.
    pub fn placeholder(&mut self, dtype: DataType) -> Tensor {
        self.allow_writes();

        let ident = NodeIdent::new();
        let full_name = self.resolve_name(None, IdType::Placeholder, false).unwrap();

        let graph = &mut *self.graph.borrow_mut();
        let tensors = &mut *self.tensors.borrow_mut();
        let ops = &mut *self.ops.borrow_mut();

        let data_origin = (
            array_ops::placeholder(graph, full_name.to_str().unwrap(), dtype).unwrap(),
            0,
        );
        let data_origin_id = NodeIdent::new();
        ops.insert(data_origin_id.clone(), data_origin.0.clone());

        tensors.insert(
            ident,
            TensorData::new(
                TensorData::name_builder(full_name, 0),
                dtype,
                IdType::Placeholder,
                data_origin,
                data_origin_id.clone(),
                Shape::from(None),
            ),
        );

        Tensor {
            ident,
            idtype: IdType::Placeholder,
            dtype,
            idx: 0,
            initializer: None,
            origin_op: Some(data_origin_id),
        }
    }

    /// Returns a scope that specifies control dependencies.
    pub fn control_dependencies<'a, I, T: 'a>(&mut self, control_inputs: I) -> Scope
    where
        I: IntoIterator<Item = &'a T>,
        T: GetOp,
    {
        self.allow_writes();
        let name = self.own_scope.name.clone();
        let mut context = self.as_new_child(name);
        let op_name = self.resolve_name(None, IdType::Operation("NoOp"), false)
            .unwrap();

        let tensors = &*self.tensors.borrow();
        let existing_ops = &*self.ops.borrow();
        let global = &mut self.scopes.borrow_mut().control_dependencies;

        let mut ops = vec![];
        for control_input in control_inputs.into_iter() {
            let ident = control_input.get_op();
            let ctrl = if let Some(op) = existing_ops.get(ident) {
                ops.push(op);
                ControlOp {
                    ident: *ident,
                    finished: op.clone(),
                    kind: ControlOpKind::Ops,
                }
            } else {
                let finished = &tensors[&ident].data_origin.0;
                ops.push(finished);
                ControlOp {
                    ident: *ident,
                    finished: finished.clone(),
                    kind: ControlOpKind::Other,
                }
            };
            context
                .own_scope
                .control_dependencies
                .push_back(ctrl.clone());
            global.push_back(ctrl);
        }

        let ctrl_group = {
            let graph = &mut *self.graph.borrow_mut();
            ControlOp {
                ident: NodeIdent::new(),
                finished: control_flow_ops::no_op_(graph, op_name.to_str().unwrap(), ops).unwrap(),
                kind: ControlOpKind::Ops,
            }
        };
        global.push_front(ctrl_group);
        context
    }

    /// Returns a scope which ignores all previously set up control dependencies.
    pub fn clear_control_dependencies(&mut self) -> Scope {
        let name = self.own_scope.name.clone();
        let mut context = self.as_new_child(name);
        context.ignore_deps = true;
        context
    }

    /// Returns a copy of the tensor, with the same shape and content.
    pub fn identity<S, Tx>(&mut self, tensor: Tx, name: S) -> Result<Tensor>
    where
        S: AsRef<Path>,
        Tx: GetOp,
    {
        self.allow_writes();

        let graph = &mut *self.graph.borrow_mut();
        let tensors = &mut *self.tensors.borrow_mut();
        let ops = &mut *self.ops.borrow_mut();
        let global = &self.scopes.borrow().control_dependencies;

        let (dtype, idtype, data_origin, full_name) = {
            if tensor.source_index().is_none() {
                return Err(Error::from(ErrorKind::from(
                    "provided type for `identity` op is not from a tensor",
                )));
            }
            let src = tensors
                .get(&tensor.get_op())
                .ok_or(Error::from(ErrorKind::TensorNotFound))?;
            let full_name = self.resolve_name(Some(name.as_ref()), src.idtype, false)?;
            let data_origin = (
                array_ops::identity(
                    graph,
                    full_name.to_str().unwrap(),
                    src.data_origin.clone(),
                    global.iter().map(|x| &x.finished),
                )?,
                0,
            );
            (src.dtype, src.idtype, data_origin, full_name)
        };
        let data_origin_id = NodeIdent::new();
        ops.insert(data_origin_id.clone(), data_origin.0.clone());

        let ident = NodeIdent::new();
        tensors.insert(
            ident,
            TensorData::new(
                full_name,
                dtype,
                IdType::Operation("Identity"),
                data_origin,
                data_origin_id.clone(),
                Shape::from(None),
            ),
        );

        Ok(Tensor {
            ident,
            dtype: dtype,
            idtype: idtype,
            idx: 0,
            initializer: None,
            origin_op: Some(data_origin_id),
        })
    }

    /// Sets the graph-level random seed. Can only be set at root scope context,
    /// panics otherwise.
    ///
    /// Operations that rely on a random seed actually derive it from two seeds:
    /// the graph-level and operation-level seeds. This sets the graph-level seed.
    ///
    /// Its interactions with operation-level seeds is as follows:
    ///     1. If neither the graph-level nor the operation seed is set:
    ///       A random seed is used for this op.
    ///     2. If the graph-level seed is set, but the operation seed is not:
    ///       The system deterministically picks an operation seed in conjunction
    ///       with the graph-level seed so that it gets a unique random sequence.
    ///     3. If the graph-level seed is not set, but the operation seed is set:
    ///       A default graph-level seed and the specified operation seed are used to
    ///       determine the random sequence.
    ///     4. If both the graph-level and the operation seed are set:
    ///       Both seeds are used in conjunction to determine the random sequence.
    pub fn set_random_seed(&mut self, value: Option<i64>) {
        if self.parent_lock.is_some() {
            panic!("random seed can only be set at the root scope");
        }
        self.seed = value;
    }

    /// Returns the local seeds an operation should use given an op-specific seed.
    ///
    /// Given operation-specific seed, op_seed, this helper function returns two seeds
    /// derived from graph-level and op-level seeds. Many random operations internally
    /// use the two seeds to allow user to change the seed globally for a graph, or
    /// for only specific operations.
    pub fn get_seed(&self, op_seed: Option<i64>) -> (Option<i64>, Option<i64>) {
        let seeds;
        if let Some(g_seed) = self.seed {
            let op_seed = if let Some(seed) = op_seed {
                seed
            } else {
                // op_seed = ops.get_default_graph()._last_id
                0
            };
            seeds = (Some(g_seed), Some(op_seed));
        } else if op_seed.is_some() {
            seeds = (Some(DEFAULT_GRAPH_SEED), op_seed);
        } else {
            seeds = (None, None);
        }
        if let (Some(0), Some(0)) = seeds {
            (Some(0), Some(::std::i64::MAX))
        } else {
            seeds
        }
    }

    #[doc(hidden)]
    /// Marks the given op as unfetchable in this graph.
    pub fn prevent_fetching<Op: GetOp>(&mut self, op: Op) {
        self.scopes.borrow_mut().unfetchable.insert(*op.get_op());
    }

    #[doc(hidden)]
    /// Marks the given op as unfeedable in this graph.
    pub fn prevent_feeding<Op: GetOp>(&mut self, op: Op) {
        self.scopes.borrow_mut().unfeedable.insert(*op.get_op());
    }

    /// Consumes self and returns underlying graph if it's a unique reference, otherwise
    /// will return a Rc pointer to it.
    pub fn unwrap_graph(mut self) -> ::std::result::Result<Graph, Rc<RefCell<Graph>>> {
        let mut graph = Rc::new(RefCell::new(Graph::new()));
        ::std::mem::swap(&mut graph, &mut self.graph);
        match Rc::try_unwrap(graph) {
            Ok(cell) => Ok(cell.into_inner()),
            Err(rc) => Err(rc),
        }
    }

    /// Returns this scope unique name.
    pub fn name(&self) -> &str {
        self.own_scope.name.to_str().unwrap()
    }

    pub(crate) fn is_function(&self, id: &NodeIdent) -> bool {
        unimplemented!()
    }

    pub(crate) fn get_function(&self, id: &NodeIdent) -> Result<Function> {
        unimplemented!()
    }

    pub(crate) fn get_gradient_function<N: GetOp>(&self, func: N) -> Option<GradFunc> {
        unimplemented!()
    }
}

impl ::std::ops::Drop for Scope {
    fn drop(&mut self) {
        use std::mem::swap;
        let mut new_scope = InternScope::new(PathBuf::new());
        swap(&mut new_scope, &mut self.own_scope);
        let global = &mut *self.scopes.borrow_mut();
        let InternScope {
            name,
            variables,
            constants,
            ops,
            inner_scopes,
            control_dependencies,
            unfetchable,
            unfeedable,
        } = new_scope;
        // find if this scope already exists
        if let Some(parent) = find_parent_scope(&mut global.inner_scopes, &name, 0) {
            // merge changes into it
            parent.inner_scopes.extend(inner_scopes);
            parent.variables.extend(variables);
            parent.ops.extend(ops);
            parent.constants.extend(constants);
        }
        // pop dependencies in this scope, if there are any
        let original_deps = global.control_dependencies.len() - control_dependencies.len();
        global.control_dependencies.truncate(original_deps);
        // add to global scope:
        global.unfetchable.extend(unfetchable);
        global.unfeedable.extend(unfeedable);
        if let Some(lock) = self.parent_lock.as_ref() {
            *lock.borrow_mut() = false;
        }
    }
}

enum OpInput {
    List(Vec<Output>),
    Single(Output),
}

#[derive(Debug, Clone)]
pub(crate) struct InternScope {
    /// Full path for this scope.
    pub name: PathBuf,
    /// Variables only available in this scope.
    variables: Vec<(PathBuf, Variable)>,
    /// Constants only available in this scope.
    constants: Vec<(PathBuf, NodeIdent)>,
    /// Ops declared in this scope.
    pub ops: Vec<(PathBuf, NodeIdent)>,
    /// Children scopes.
    inner_scopes: Vec<Box<InternScope>>,
    /// Control dependencies in this scope
    pub control_dependencies: VecDeque<ControlOp>,
    /// Unfetchable tensors.
    unfetchable: HashSet<NodeIdent>,
    /// Unfeedable tensors.
    unfeedable: HashSet<NodeIdent>,
}

impl InternScope {
    fn new(name: PathBuf) -> InternScope {
        InternScope {
            name,
            variables: vec![],
            constants: vec![],
            inner_scopes: vec![],
            ops: vec![],
            control_dependencies: VecDeque::new(),
            unfetchable: HashSet::new(),
            unfeedable: HashSet::new(),
        }
    }

    fn add_scope(&mut self, scope: InternScope) {
        self.inner_scopes.push(Box::new(scope));
        self.inner_scopes
            .sort_unstable_by(|a, b| a.name.cmp(&b.name));
    }

    fn get_child(&self, name: &Path) -> Option<&InternScope> {
        let f = name.file_name().unwrap();
        for scope in &self.inner_scopes {
            if let Ok(rest) = name.strip_prefix(&self.name) {
                if rest == f {
                    return Some(self);
                } else {
                    return scope.get_child(name);
                }
            }
        }
        None
    }

    fn name_exists(&self, name: &Path) -> bool {
        for &(ref var_name, _) in &self.variables {
            if var_name == name {
                return true;
            }
        }
        for &(ref const_name, _) in &self.constants {
            if const_name == name {
                return true;
            }
        }
        false
    }
}

fn find_parent_scope<'a>(
    scopes: &'a mut [Box<InternScope>],
    name: &Path,
    comp: usize,
) -> Option<&'a mut InternScope> {
    for scope in scopes {
        let rest = if let Some((_, prefix)) = scope
            .name
            .iter()
            .enumerate()
            .skip_while(|&(ref i, _)| i < &comp)
            .find(|&(_, p)| name.starts_with(p))
        {
            name.strip_prefix(prefix).unwrap().to_owned()
        } else {
            continue;
        };
        if rest.parent().is_none() {
            return Some(scope);
        }
        return find_parent_scope(&mut scope.inner_scopes, &rest, comp + 1);
    }
    None
}

impl PartialEq for InternScope {
    fn eq(&self, rhs: &InternScope) -> bool {
        self.name == rhs.name
    }
}

/*
#[cfg(test)]
mod test {
    #![allow(unused_imports)]
    use super::*;

    #[ignore]
    #[test]
    fn scope_management() {
        let mut root = Scope::new();
        /*
        let scopes =
            context.get_variable("glob_01", DataType::Double, Some(&[1, 1, 1])).unwrap();
        let foo_01;
        {
            let mut foo = context.variable_scope("foo", None);
            foo_01 = foo.get_variable("foo_01", DataType::Int32, Some(&[2, 2])).unwrap();
        }
        */
        {
            let foo = &mut root.name_scope("foo", None);
            {
                let bar = &mut foo.name_scope("bar", None);
            }
        }
    }
}
*/

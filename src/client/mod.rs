use std::cell::RefCell;
use std::rc::Rc;

use super::{DataType, Graph, Session, SessionOptions, StepWithGraph};
use super::framework::*;

/// A ClientSession object lets the caller drive the evaluation of the TensorFlow graph
/// constructed with the Rust API.
#[derive(Debug)]
pub struct ClientSession<'g> {
    fetch: Vec<Ident>,
    feed: Vec<(Ident, Vec<TensorContent>)>,
    context: &'g mut Scope,
    reinit_vars: bool,
    initialized: bool,
}

impl<'g> ClientSession<'g> {
    pub fn new(context: &'g mut Scope) -> Result<ClientSession<'g>, ::Error> {
        if context.locked() {
            // can only create sessions out of root scopes
            return Err(::Error::Stub);
        }
        Ok(ClientSession {
               fetch: vec![],
               feed: vec![],
               context,
               reinit_vars: false,
               initialized: false,
           })
    }

    /// If variables had already been initialized, reinitialize them again.
    ///
    /// Default for this setting is false.
    pub fn reinitialize_vars(&mut self) -> &mut Self {
        self.reinit_vars = true;
        self
    }

    /// Output to fetch from the given operation.
    pub fn fetch<I, V>(&mut self, fetches: I) -> &mut Self
        where I: IntoIterator<Item = V>,
              V: Into<Ident>
    {
        for t in fetches.into_iter() {
            self.fetch.push(t.into());
        }
        self
    }

    /// Input to feed to a graph node.
    pub fn feed<In, I, Id>(&mut self, inputs: I) -> &mut Self
        where Id: Into<Ident>,
              I: IntoIterator<Item = (Id, Vec<TensorContent>)>
    {
        for (id, inputs) in inputs.into_iter() {
            self.feed.push((id.into(), inputs));
        }
        self
    }

    /// Prune the session of previous feed/fetch inputs and results.
    pub fn prune(&mut self) -> &mut Self {
        self.fetch.clear();
        self.feed.clear();
        self
    }

    /// Consumes self and returns underlying graph if it's a unique reference, otherwise
    /// will return a Rc pointer to it.
    pub fn unwrap_graph(self) -> Result<Graph, Rc<RefCell<Graph>>> {
        let mut graph = Rc::new(RefCell::new(Graph::new()));
        ::std::mem::swap(&mut graph, &mut self.context.graph);
        match Rc::try_unwrap(graph) {
            Ok(cell) => Ok(cell.into_inner()),
            Err(rc) => Err(rc),
        }
    }

    /// Run a TensorFlow session with the currently built-in graph in context.
    /// All variables are initialized beforehand.
    ///
    /// Evaluates the tensors provided during the session construction to fetch.
    /// The number and order of outputs will match the construction phase.
    ///
    /// Returns error if any of the session input was inadequate.
    pub fn run(&mut self, options: Option<SessionOptions>) -> Result<Vec<TensorContent>, ::Error> {
        fn control_ops<'a, I>(init_steep: &mut StepWithGraph,
                              steep: &mut StepWithGraph,
                              control_ops: I)
                              -> bool
            where I: Iterator<Item = &'a ControlOp>
        {
            let mut any_init = false;
            for init in control_ops {
                if init.kind == ControlOpKind::VarInitializer {
                    any_init = true;
                    init_steep.add_target(&init.finished);
                } else if init.kind == ControlOpKind::Group {
                    steep.add_target(&init.finished);
                }
            }
            any_init
        }

        let graph = &*self.context.graph.borrow();
        let registry = &*self.context.registry.borrow();

        let mut session = if let Some(opts) = options {
            Session::new(&opts, graph)
        } else {
            Session::new(&SessionOptions::new(), graph)
        }?;

        let root_deps = &self.context.scopes.borrow().control_dependencies;
        let this_deps = self.context.own_scope.control_dependencies.iter();

        let steep0 = &mut StepWithGraph::new();
        let steep1 = &mut StepWithGraph::new();

        let any_init0 = control_ops(steep0, steep1, this_deps);
        let any_init1 = control_ops(steep0, steep1, root_deps.iter());
        // initialize variables
        if (!self.initialized || self.reinit_vars) && (any_init0 || any_init1) {
            session.run(steep0)?;
        }

        // take output tokens
        let mut output_tokens = Vec::with_capacity(self.fetch.len());
        for output in &self.fetch {
            let info = &registry[&output];
            output_tokens.push((steep1.request_output(&info.data_origin.0, info.data_origin.1),
                                info.dtype));
        }
        // feed input
        for &(ref token, ref inputs) in &self.feed {
            let info = &registry[&token];
            let op = &info.data_origin.0;
            let idx = info.data_origin.1;
            for input in inputs {
                tensor_output_op!(input; StepWithGraph::add_input[steep1, op, idx,]);
            }
        }

        session.run(steep1)?;
        // fetch the outputs
        let mut results = Vec::with_capacity(self.fetch.len());
        for (token, dtype) in output_tokens {
            let res = match dtype {
                DataType::Bool => TensorContent::from(steep1.take_output::<bool>(token)?),
                DataType::Double => TensorContent::from(steep1.take_output::<f64>(token)?),
                DataType::Float => TensorContent::from(steep1.take_output::<f32>(token)?),
                DataType::Int32 => TensorContent::from(steep1.take_output::<i32>(token)?),
                DataType::UInt8 => TensorContent::from(steep1.take_output::<u8>(token)?),
                DataType::Int16 => TensorContent::from(steep1.take_output::<i16>(token)?),
                DataType::Int8 => TensorContent::from(steep1.take_output::<i8>(token)?),
                DataType::Int64 => TensorContent::from(steep1.take_output::<i64>(token)?),
                _ => unimplemented!(),
            };
            results.push(res);
        }

        Ok(results)
    }
}

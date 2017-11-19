use super::*;

add_new_op!(SparseSoftmaxCrossEntropyWithLogits,
    constructor: [
        add_new_op!(
            BIN CONSTRUCTOR: SparseSoftmaxCrossEntropyWithLogits, 
            Init: []);
    ],
    digest: [DIGEST:
        fn digest(
            self, 
            context: &mut Scope, 
            op: OperationData
        ) 
            -> Result<Self::Outputs> 
        {
            let idtype = IdType::Operation("SparseSoftmaxCrossEntropyWithLogits");
            
            let dtype = add_new_op!(INPUT0 self);
            let shape0 = {
                let g = context.graph.borrow();
                g.tensor_shape(
                        Output {
                            operation: op.clone(),
                            index: 0,
                        },
                    )?
            };
            let shape1 = {
                let g = context.graph.borrow();
                g.tensor_shape(
                        Output {
                            operation: op.clone(),
                            index: 1,
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
                    shape: shape0,
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
                    shape: shape1,
                },
            );
            
            Ok((tensor0, tensor1))
        }
    ],
    extra_funcs: [], 
    extra_attr: [],
    output: [(Tensor, Tensor)],
);

add_new_op!(InTopKV2, 
    constructor: [
        pub(crate) fn new<S: AsRef<Path>>(
            predictions: Tensor, 
            targets: Tensor, 
            k: Tensor, 
            name: S
        ) -> Result<InTopKV2<'a>> {
            Ok(
                InTopKV2 {
                    ident: NodeIdent::new(),
                    elements: vec![predictions, targets, k],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: vec![],
                    output_type: DataType::Bool,
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: InTopKV2, DTYPE_ATTR],
    extra_funcs: [], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);

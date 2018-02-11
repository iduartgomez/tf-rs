use super::*;

add_new_op!(SparseSoftmaxCrossEntropyWithLogits,
    constructor: [
        add_new_op!(
            BIN CONSTRUCTOR: SparseSoftmaxCrossEntropyWithLogits, 
            Init: []);
    ],
    digest: [DIGEST_BIN_OUT: SparseSoftmaxCrossEntropyWithLogits, INPUT0, INPUT0],
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
                    input_lists: Vec::with_capacity(0),
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

add_new_op!(L2Loss, 
    constructor: [add_new_op!(UNARY CONSTRUCTOR: L2Loss, Init: []);],
    digest: [DEFAULT_DIGEST: L2Loss, INPUT0],
    extra_funcs: [], 
    extra_attr: [],
    output: [Tensor],
);

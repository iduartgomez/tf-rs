use super::*;

///  Update '*var' by subtracting 'alpha' * 'delta' from it.
///
///  ### Args:
///    * var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
///      Should be from a Variable().
///    * alpha: A `Tensor`. Must have the same type as `var`.
///      Scaling factor. Must be a scalar.
///    * delta: A `Tensor`. Must have the same type as `var`. The change.
///    * use_locking: An optional `bool`. Defaults to `False`.
///      If `True`, the subtraction will be protected by a lock;
///      otherwise the behavior is undefined, but may exhibit less contention.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    A mutable `Tensor`. Has the same type as `var`. Same as "var".
pub fn apply_gradient_descent<Tx, Ty, S>(
    context: &mut Scope,
    var: Variable,
    alpha: Tx,
    beta: Ty,
    use_locking: bool,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let alpha = alpha.into_tensor(context);
    let beta = beta.into_tensor(context);
    context.install(
        ApplyGradientDescent::new(var.into(), alpha, beta, name)?.use_locking(&[use_locking]),
    )
}

add_new_op!(ApplyGradientDescent, 
    constructor: [
        pub(crate) fn new<S: AsRef<Path>>(
            var: Tensor, 
            alpha: Tensor, 
            beta: Tensor, 
            name: S
        ) 
            -> Result<ApplyGradientDescent<'a>> 
        {
            Ok(
                ApplyGradientDescent {
                    ident: NodeIdent::new(),
                    elements: vec![var, alpha, beta],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: Vec::with_capacity(0),
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: ApplyGradientDescent, INPUT0],
    extra_funcs: [
        /// Default is false, must be an slice of len == 1.
        fn use_locking(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("use_locking", false, Attribute::Bool(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);

///  Update '*var' by subtracting 'alpha' * 'delta' from it.
///
///  ### Args:
///    * var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
///      Should be from a Variable().
///    * alpha: A `Tensor`. Must have the same type as `var`.
///      Scaling factor. Must be a scalar.
///    * delta: A `Tensor`. Must have the same type as `var`. The change.
///    * use_locking: An optional `bool`. Defaults to `False`.
///      If `True`, the subtraction will be protected by a lock;
///      otherwise the behavior is undefined, but may exhibit less contention.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    A mutable `Tensor`. Has the same type as `var`. Same as "var".
pub fn resource_apply_gradient_descent<Tx, Ty, S>(
    context: &mut Scope,
    var: Variable,
    alpha: Tx,
    beta: Ty,
    use_locking: bool,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let alpha = alpha.into_tensor(context);
    let beta = beta.into_tensor(context);
    context.install(
        ResourceApplyGradientDescent::new(var.into(), alpha, beta, name)?
            .use_locking(&[use_locking]),
    )
}

add_new_op!(ResourceApplyGradientDescent, 
    constructor: [
        pub(crate) fn new<S: AsRef<Path>>(
            var: Tensor, 
            alpha: Tensor, 
            beta: Tensor, 
            name: S
        ) 
            -> Result<ResourceApplyGradientDescent<'a>> 
        {
            Ok(
                ResourceApplyGradientDescent {
                    ident: NodeIdent::new(),
                    elements: vec![var, alpha, beta],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: Vec::with_capacity(0),
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: ResourceApplyGradientDescent, INPUT0],
    extra_funcs: [
        /// Default is false, must be an slice of len == 1.
        fn use_locking(mut self, val: &'a [bool]) -> Self {
            self.attributes.push(("use_locking", false, Attribute::Bool(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Tensor],
);

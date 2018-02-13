use super::*;

///  Adds sparse updates to the variable referenced by `resource`.
///
///  Duplicate entries are handled correctly: if multiple `indices` reference
///  the same location, their contributions add.
///
///  Requires `updates.shape = indices.shape + ref.shape[1:]`.
///
///  ### Args:
///    * resource: A `Tensor` of type `resource`. Should be from a `Variable` node.
///    * indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
///      A tensor of indices into the first dimension of `ref`.
///    * updates: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
///      A tensor of updated values to add to `ref`.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    The created Operation.
pub fn resource_scatter_add<S>(
    scope: &mut Scope,
    resource: Tensor,
    indices: Tensor,
    updates: Tensor,
    name: S,
) -> Result<ResourceScatterAdd>
where
    S: AsRef<Path>,
{
    let op = ResourceScatterAdd::new(resource, indices, updates, name)?;
    scope.install(op.clone())?;
    Ok(op)
}

#[derive(Debug, Clone)]
pub struct ResourceScatterAdd<'a> {
    ident: NodeIdent,
    elements: [Tensor; 3],
    name: Option<PathBuf>,
    attributes: Vec<(&'a str, bool, Attribute<'a>)>,
    input_lists: Vec<(usize, Vec<Tensor>)>,
}

impl<'a> ResourceScatterAdd<'a> {
    fn new<S: AsRef<Path>>(
        resource: Tensor,
        indices: Tensor,
        update: Tensor,
        name: S,
    ) -> Result<ResourceScatterAdd<'a>> {
        Ok(ResourceScatterAdd {
            ident: NodeIdent::new(),
            elements: [resource, indices, update],
            name: generate_name!(is_none: name),
            attributes: Vec::with_capacity(0),
            input_lists: Vec::with_capacity(0),
        })
    }
}

impl<'a> Operation<'a> for ResourceScatterAdd<'a> {
    type Outputs = ();

    add_new_op!(CORE_FN: ResourceScatterAdd);

    fn digest(self, context: &mut Scope, op: OperationData) -> Result<Self::Outputs> {
        add_new_op!(REGISTER_AS_OP: (self, context, op); ResourceScatterAdd);
        Ok(())
    }
}

impl_into_ident!(ResourceScatterAdd);

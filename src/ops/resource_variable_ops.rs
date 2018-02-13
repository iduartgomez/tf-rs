use super::*;

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

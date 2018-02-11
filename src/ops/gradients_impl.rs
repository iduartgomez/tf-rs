use super::*;

pub fn gradients<Tys, Txs, S>(
    scope: &mut Scope,
    ys: Vec<Tys>,
    xs: Vec<Txs>,
    grad_ys: Option<Vec<Tensor>>,
    colocate_gradients_with_ops: bool,
    gate_gradients: bool,
    aggregation_method: &str,
    stop_gradients: Option<Vec<Tensor>>,
    name: S,
) -> Result<Vec<Option<Tensor>>>
where
    Txs: TensorOps,
    Tys: TensorOps,
    S: AsRef<Path>
{
    unimplemented!()
}

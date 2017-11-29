use super::*;

///  Partitions `data` into `num_partitions` tensors using indices from `partitions`.
///
///  For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
///  becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
///  are placed in `outputs[i]` in lexicographic order of `js`, and the first
///  dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
///  In detail,
///
///  ```python
///      outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]
///
///      outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
///  ```
///
///  `data.shape` must start with `partitions.shape`.
///
///  For example:
///
///  ```python
///      # Scalar partitions.
///      partitions = 1
///      num_partitions = 2
///      data = [10, 20]
///      outputs[0] = []  # Empty with shape [0, 2]
///      outputs[1] = [[10, 20]]
///
///      # Vector partitions.
///      partitions = [0, 0, 1, 1, 0]
///      num_partitions = 2
///      data = [10, 20, 30, 40, 50]
///      outputs[0] = [10, 20, 50]
///      outputs[1] = [30, 40]
///  ```
///
///  See `dynamic_stitch` for an example on how to merge partitions back.
///
///  ### Args:
///    * data: A `Tensor`.
///    * partitions: A `Tensor` of type `int32`.
///      Any shape.  Indices in the range `[0, num_partitions)`.
///    * num_partitions: An `int` that is `>= 1`.
///      The number of partitions to output.
///    * name: A name for the operation (optional).
///
///  ### Returns:
///    A list of `num_partitions` `Tensor` objects with the same type as `data`.
pub fn dynamic_partition<Tx, Ty, S>(
    scope: &mut Scope,
    data: Tx,
    partitions: Ty,
    num_partitions: i32,
    name: S,
) -> Result<Vec<Tensor>>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let data = data.into_tensor(scope);
    let partitions = partitions.into_tensor(scope);

    scope.install(
        DynamicPartition::new(data, partitions, name)?.num_partitions(&[num_partitions as i64]),
    )
}

add_new_op!(DynamicPartition, 
    constructor: [add_new_op!(BIN CONSTRUCTOR: DynamicPartition, Init: []);],
    digest: [DIGEST:
        fn digest(
            self, 
            context: &mut Scope, 
            op: OperationData,
        ) 
            -> Result<Self::Outputs> 
        {
            let idtype = IdType::Operation("DynamicPartition");
            let dtype = add_new_op!(INPUT0 self);
        
            let g = &*context.graph.borrow();
            let reg = &mut *context.registry.borrow_mut();

            let output_len = op.output_list_length("outputs").unwrap();
            let mut outputs = Vec::with_capacity(output_len);
            for output_num in 0..(output_len as i32) {
                let shape0 = {
                g.tensor_shape(
                            Output {
                                operation: op.clone(),
                                index: output_num,
                            },
                        )?
                };
            
                let ident0 = NodeIdent::new();
                let full_name0 = context.resolve_tensor_name(self.get_op_name(), idtype, false)?;
                let tensor0 = Tensor {
                    ident: ident0,
                    idtype,
                    dtype,
                    idx: output_num,
                    initializer: None,
                };
                outputs.push(tensor0);
                
                match context.control_context {
                    ControlFlow::CondContext(ref mut cond) => {
                        cond.values.insert(ident0);
                        cond.external_values.insert(ident0, tensor0);
                    }
                    ControlFlow::WhileContext(ref mut cond) => {
                        cond.values.insert(ident0);
                        cond.external_values.insert(ident0, tensor0);
                    }
                    ControlFlow::None => {}
                }

                context.own_scope.ops.push((full_name0.clone(), ident0));
                reg.insert(
                    ident0,
                    TensorData {
                        full_name: full_name0,
                        dtype,
                        idtype,
                        data_origin: (op.clone(), output_num),
                        shape: shape0,
                    },
                );
            }
            
            Ok(outputs)
        }
    ],
    extra_funcs: [
        fn num_partitions(mut self, val: &'a [i64]) -> Self {
            self.attributes.push(("num_partitions", false, Attribute::Int(val)));
            self
        }
    ], 
    extra_attr: [],
    output: [Vec<Tensor>],
);

#[test]
#[cfg(test)]
fn test_dynamic_partition() {
    let mut context = Scope::new();
    let x = context.constant(&[10_i32, 20, 30, 40, 50], &[5], "x").unwrap();

    let op = dynamic_partition(&mut context, x, [0_i32, 0, 1, 1, 0].as_ref(), 2, "").unwrap();
    assert_eq!(op.len(), 2);

    let a = &op[0];
    let results = test_suite!(run_op: [a]; context, input: {});
    test_suite!(results; assert_len: {[0;Int32] == 3});
    test_suite!(results; assert: {[0;Int32] == [10_i32, 20, 50]});

    let b = &op[1];
    let results = test_suite!(run_op: [b]; context, input: {});
    test_suite!(results; assert_len: {[0;Int32] == 2});
    test_suite!(results; assert: {[0;Int32] == [30_i32, 40]});
}


/// Interleave the values from the `data` tensors into a single tensor.
///
/// Builds a merged tensor such that
///
/// ```python
///     merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
/// ```
///
/// For example, if each `indices[m]` is scalar or vector, we have
///
/// ```python
///     # Scalar indices:
///     merged[indices[m], ...] = data[m][...]
///
///     # Vector indices:
///     merged[indices[m][i], ...] = data[m][i, ...]
/// ```
///
/// Each `data[i].shape` must start with the corresponding `indices[i].shape`,
/// and the rest of `data[i].shape` must be constant w.r.t. `i`. That is, we
/// must have `data[i].shape = indices[i].shape + constant`. In terms of this `constant`,
/// the output shape is
///
/// ```python
///     merged.shape = [max(indices)] + constant
/// ```
///
/// Values are merged in order, so if an index appears in both 
/// ```python
/// indices[m][i] and indices[n][j] 
/// for (m,i) < (n,j) the slice data[n][j]
/// ``` 
/// will appear in the merged result.
///
/// For example:
///
/// ```python
///     indices[0] = 6
///     indices[1] = [4, 1]
///     indices[2] = [[5, 2], [0, 3]]
///     data[0] = [61, 62]
///     data[1] = [[41, 42], [11, 12]]
///     data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
///     merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
///               [51, 52], [61, 62]]
/// ```
///
/// ### Args:
///     * indices: A list of at least 1 Tensor objects of type int32.
///     * data: A list with the same number of Tensor objects as indices of Tensor objects of the same type.
///     * name: A name for the operation (optional).
///
/// ### Returns:
///     A Tensor. Has the same type as data.
pub fn dynamic_stitch<Tx, Ty, S>(
    scope: &mut Scope,
    indices: Vec<Ty>,
    mut data: Vec<Tx>,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let indices = indices
        .into_iter()
        .map(|x| {
            let x = x.into_tensor(scope);
            if let DataType::Int32 = x.dtype {
                Ok(x)
            } else {
                return Err(Error::from(ErrorKind::Stub));
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let last = if let Some(last) = data.pop() {
        last.into_tensor(scope)
    } else {
        return Err(Error::from(ErrorKind::Stub));
    };
    let mut data = data.into_iter()
        .map(|x| {
            let x = x.into_tensor(scope);
            if last.dtype == x.dtype {
                Ok(x)
            } else {
                return Err(Error::from(ErrorKind::Stub));
            }
        })
        .collect::<Result<Vec<_>>>()?;
    data.push(last);
    let input_len = data.len() as i64;

    scope.install(DynamicStitch::new(indices, data, name)?.input_len(&[input_len]))
}

add_new_op!(DynamicStitch, 
    constructor: [
        fn new<S: AsRef<Path>>(
            indices: Vec<Tensor>, 
            data: Vec<Tensor>, 
            name: S,
        ) -> Result<DynamicStitch<'a>> {
            if data.len() != indices.len() {
                return Err(Error::from(format!(
                    "List argument 'data' to 'dynamic_stitch' Op with length {}
                    must match length {} of argument 'indices'.",
                    data.len(), indices.len())));
            }

            let output_type = data[0].dtype;
            Ok(
                DynamicStitch {
                    ident: NodeIdent::new(),
                    input_lists: vec![(0, indices), (0, data)],
                    elements: Vec::with_capacity(0),
                    name: generate_name!(is_none: name),
                    attributes: Vec::with_capacity(1),
                    output_type,
                },
            )
        }
    ],
    digest: [DEFAULT_DIGEST: Pack, DTYPE_ATTR],
    extra_funcs: [
        fn input_len(mut self, val: &'a [i64]) -> Self {
            self.attributes.push(("N", false, Attribute::Int(val)));
            self
        }
    ], 
    extra_attr: [output_type: DataType],
    output: [Tensor],
);

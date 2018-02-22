use super::*;

///// Embedding Lookup /////

///  Looks up `ids` in a list of embedding tensors.
///
///  This function is used to perform parallel lookups on the list of
///  tensors in `params`.  It is a generalization of
///  `gather`, where `params` is
///  interpreted as a partitioning of a large embedding tensor.
///
///  If `len(params) > 1`, each element `id` of `ids` is partitioned between
///  the elements of `params` according to the `partition_strategy`.
///  In all strategies, if the id space does not evenly divide the number of
///  partitions, each of the first `(max_id + 1) % len(params)` partitions will
///  be assigned one more id.
///
///  If `partition_strategy` is `"mod"`, we assign each id to partition
///  `p = id % len(params)`. For instance,
///  13 ids are split across 5 partitions as:
///  `[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]`
///
///  If `partition_strategy` is `"div"`, we assign ids to partitions in a
///  contiguous manner. In this case, 13 ids are split across 5 partitions as:
///  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`
///
///  The results of the lookup are concatenated into a dense
///  tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.
///
///  ### Args:
///    * params: A single tensor representing the complete embedding tensor,
///      or a list of P tensors all of same shape except for the first dimension,
///      representing sharded embedding tensors.  Alternatively, a
///      `PartitionedVariable`, created by partitioning along dimension 0. Each
///      element must be appropriately sized for the given `partition_strategy`.
///    * ids: A `Tensor` with type `int32` or `int64` containing the ids to be looked
///      up in `params`.
///    * partition_strategy: A string specifying the partitioning strategy, relevant
///      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
///      is `"mod"`.
///    * name: A name for the operation (optional).
///    * max_norm: If provided, embedding values are l2-normalized to the value of
///      max_norm.
///
///  ### Returns:
///    A `Tensor` with the same type as the tensors in `params`.
pub fn embedding_lookup<Ty, Tx, S>(
    context: &mut Scope,
    params: &[Tensor],
    ids: Tx,
    partition_strategy: &str,
    max_norm: Option<Ty>,
    name: S,
) -> Result<Tensor>
where
    Tx: TensorOps,
    Ty: TensorOps,
    S: AsRef<Path>,
{
    let ids = ids.into_tensor(context);
    let max_norm = if let Some(max_norm) = max_norm {
        Some(max_norm.into_tensor(context))
    } else {
        None
    };
    embedding_lookup_and_transform(context, params, ids, partition_strategy, max_norm, None, name)
}

fn embedding_lookup_and_transform<S>(
    context: &mut Scope,
    params: &[Tensor],
    ids: Tensor,
    partition_strategy: &str,
    max_norm: Option<Tensor>,
    mut transform_fn: Option<Box<FnMut(&mut Scope, Tensor) -> Result<Tensor>>>,
    name: S,
) -> Result<Tensor>
where
    S: AsRef<Path>,
{
    if params.is_empty() {
        return Err(Error::from(ErrorKind::Stub));
    }

    let scope = &mut context.name_scope(name.as_ref().to_str().unwrap(), Some("embedding_lookup"));
    let np = params.len() as i32; // Number of partitions
                                  // Preserve the resource variable status to avoid accidental dense reads.
    let dims = if let Some(dims) = ids.get_shape(scope).dims() {
        if dims == 1 {
            true
        } else {
            false
        }
    } else {
        false
    };
    if np == 1 && (transform_fn.is_none() || dims) {
        let s = gather(scope, params[0], ids, name)?;
        let mut result = clip(scope, s, ids, max_norm)?;
        if let Some(mut transform_fn) = transform_fn {
            result = transform_fn(scope, result)?;
        }
        return Ok(result);
    }

    // Flatten the ids. There are two cases where we need to do this.
    // - There is more than one params tensor.
    // - There is a transform_fn and ids is not statically known to be 1-D.
    //   We must flatten in this case because transform_fn expects a flat
    //   tensor of embeddings.
    let flat_ids = array_ops::reshape(scope, ids, [-1].as_ref(), "")?;
    let original_indices = {
        let s = array_ops::size(scope, flat_ids, "")?;
        math_ops::range(scope, 0, s, 1, "")?
    };
    let np_t = {
        let np = np.into_tensor(scope);
        math_ops::cast(scope, np, flat_ids.dtype, "")?
    };

    // Create p_assignments and set new_ids depending on the strategy.
    let mut p_assignments;
    let new_ids;
    if partition_strategy == "mod" {
        p_assignments = math_ops::floor_mod(scope, flat_ids, np_t, "")?;
        new_ids = math_ops::floor_div(scope, flat_ids, np_t, "")?;
    } else if partition_strategy == "div" {
        // Compute num_total_ids as the sum of dim-0 of params, then assign to
        // partitions based on a constant number of ids per partition. Optimize
        // if we already know the full shape statically.
        let mut dim_0_size = params[0].get_shape(scope).get_dim_size(0).unwrap();
        let mut sum = true;
        for p in 1..np {
            if let Some(val) = params[p as usize].get_shape(scope).get_dim_size(p as usize) {
                dim_0_size += val;
            } else {
                sum = false;
                break;
            }
        }

        let mut num_total_ids;
        if sum {
            num_total_ids = dim_0_size.into_tensor(scope);
            num_total_ids = math_ops::cast(scope, num_total_ids, flat_ids.dtype, "")?;
        } else {
            let mut dim_0_sizes = vec![];
            for (i, param) in params.iter().enumerate() {
                if let Some(val) = param.get_shape(scope).get_dim_size(0) {
                    dim_0_sizes.push(val.into_tensor(scope));
                } else {
                    let dim_size = {
                        let s = array_ops::shape(scope, *param, None, "")?;
                        array_ops::strided_slice(
                            scope,
                            s,
                            0,
                            0,
                            1,
                            None,
                            None,
                            None,
                            None,
                            None,
                            "",
                        )?
                    };
                    dim_0_sizes.push(dim_size);
                }
            }
            num_total_ids = {
                let s = array_ops::stack(scope, dim_0_sizes, 0, "")?;
                let c = math_ops::cast(scope, s, flat_ids.dtype, "")?;
                math_ops::reduce_sum(scope, c, &[] as &[i32], false, "")?
            };
        }
        let ids_per_partition = math_ops::floor_div(scope, num_total_ids, np_t, "")?;
        let extras = math_ops::floor_mod(scope, num_total_ids, np_t, "")?;

        let c_1 = {
            let c = (1_i32).into_tensor(scope);
            math_ops::cast(scope, c, ids_per_partition.dtype, "")?
        };
        p_assignments = {
            let mut a = math_ops::add(scope, ids_per_partition, c_1, "")?;
            a = math_ops::floor_div(scope, flat_ids, a, "")?;

            let mut b = math_ops::sub(scope, flat_ids, extras, "")?;
            b = math_ops::floor_div(scope, b, ids_per_partition, "")?;

            math_ops::maximum(scope, a, b, "")?
        };

        // Emulate a conditional using a boolean indicator tensor
        let is_in_first_extras_partitions = {
            let a = math_ops::minimum(scope, p_assignments, extras, "")?;
            math_ops::cast(scope, a, flat_ids.dtype, "")?
        };
        new_ids = {
            let mut a = math_ops::add(scope, ids_per_partition, c_1, "")?;
            a = math_ops::floor_mod(scope, flat_ids, a, "")?;
            a = math_ops::multiply(scope, is_in_first_extras_partitions, a, "")?;

            let mut b = math_ops::sub(scope, c_1, is_in_first_extras_partitions, "")?;
            let mut c = math_ops::sub(scope, flat_ids, extras, "")?;
            c = math_ops::floor_mod(scope, c, ids_per_partition, "")?;
            b = math_ops::multiply(scope, b, c, "")?;

            math_ops::add(scope, a, b, "")?
        };
    } else {
        return Err(Error::from(format!(
            "Unrecognized partition strategy: {}",
            partition_strategy
        )));
    }

    // Cast partition assignments to int32 for use in dynamic_partition.
    // There really should not be more than 2^32 partitions.
    p_assignments = math_ops::cast(scope, p_assignments, DataType::Int32, "")?;
    // Partition list of ids based on assignments into np separate lists
    let gather_ids = data_flow_ops::dynamic_partition(scope, new_ids, p_assignments, np, "")?;
    // Similarly, partition the original indices.
    let pindices =
        data_flow_ops::dynamic_partition(scope, original_indices, p_assignments, np, "")?;
    // Do np separate lookups, finding embeddings for plist[p] in params[p]
    let mut partitioned_result = vec![];
    for (p, pids) in gather_ids.into_iter().enumerate() {
        let mut result = gather(scope, params[p], pids, "")?;
        if let Some(ref mut transform_fn) = transform_fn {
            // If transform_fn is provided, the clip_by_norm precedes
            // the transform and hence must be co-located. See below
            // for the counterpart if transform_fn is not proveded.
            let clipped = clip(scope, result, pids, max_norm)?;
            result = transform_fn(scope, result)?;
        }
        partitioned_result.push(result);
    }
    // Stitch these back together
    let mut ret = data_flow_ops::dynamic_stitch(scope, pindices, partitioned_result, name)?;

    // Determine the static element shape.
    let element_shape_s = if transform_fn.is_none() {
        let mut element_shape_s = params[0].get_shape(scope);
        for p in &params[1..] {
            let s = p.get_shape(scope).slice(1, None)?;
            element_shape_s = element_shape_s.merge_with(&s)?;
        }
        element_shape_s
    } else {
        let s = ret.get_shape(scope);
        s.slice(1, None)?
    };

    // Compute the dynamic element shape.
    let element_shape_d = if element_shape_s.is_fully_defined() {
        let s = element_shape_s.definition_i64().unwrap();
        scope.constant(&s, [s.len() as i64].as_ref(), "")?.into()
    } else if transform_fn.is_none() {
        // TODO: It's important that we compute params[0].shape on the right device
        // to avoid data motion.
        // with ops.colocate_with(params[0]):
        let params_shape = array_ops::shape(scope, params[0], None, "")?;
        array_ops::strided_slice(
            scope,
            params_shape,
            1_32,
            ::std::i32::MAX,
            1_32,
            None,
            None,
            None,
            None,
            None,
            "",
        )?
    } else {
        let params_shape = array_ops::shape(scope, ret, None, "")?;
        array_ops::strided_slice(
            scope,
            params_shape,
            1_32,
            ::std::i32::MAX,
            1_32,
            None,
            None,
            None,
            None,
            None,
            "",
        )?
    };

    // Reshape to reverse the flattening of ids.
    ret = {
        let mut shape = array_ops::shape(scope, ids, None, "")?;
        shape = array_ops::concat(scope, vec![shape, element_shape_d], 0, "")?;
        array_ops::reshape(scope, ret, shape, "")?
    };

    /*
    // Normally the reshape is sufficient, but setting shape explicitly
    // teaches shape inference that params[1:].get_shape() matters
    // (in the case that transform_fn is None).
    ret.set_shape(ids.get_shape().concatenate(element_shape_s))
    */
    if transform_fn.is_none() {
        // If transform_fn was provided, the clip_by_norm was done above.
        ret = clip(scope, ret, ids, max_norm)?;
    }
    Ok(ret)
}

/// Helper function for _embedding_lookup_and_transform.
///
/// This function gathers embeddings from a single tensor. The gather deals with
/// resource variables specially.
fn gather<S: AsRef<Path>>(
    scope: &mut Scope,
    params: Tensor,
    ids: Tensor,
    name: S,
) -> Result<Tensor> {
    array_ops::gather(scope, params, ids, name)
}

///  Helper function for _embedding_lookup_and_transform.
///
///  This function optionally clips embeddings to an l2-norm of max_norm.
fn clip(
    scope: &mut Scope,
    params: Tensor,
    ids: Tensor,
    max_norm: Option<Tensor>,
) -> Result<Tensor> {
    if max_norm.is_none() {
        return Ok(params);
    }
    let (ids_rank, ids_static) = rank(scope, ids)?;
    let (params_rank, params_static) = rank(scope, params)?;
    let axes = if ids_static && params_static {
        let ids_rank = match ids_rank {
            RankVal::S(t) => t as i64,
            _ => return Err(Error::from(ErrorKind::Stub)),
        };
        let params_rank = match params_rank {
            RankVal::S(t) => t as i64,
            _ => return Err(Error::from(ErrorKind::Stub)),
        };
        let r: Vec<_> = (ids_rank..params_rank).collect();
        (&r).into_tensor(scope)
    } else {
        let ids_rank = match ids_rank {
            RankVal::T(t) => t,
            _ => return Err(Error::from(ErrorKind::Stub)),
        };
        let params_rank = match params_rank {
            RankVal::T(t) => t,
            _ => return Err(Error::from(ErrorKind::Stub)),
        };
        math_ops::range(scope, ids_rank, params_rank, 1, "")?
    };
    clip_by_norm(scope, params, max_norm.unwrap(), Some(axes), "")
}

enum RankVal {
    S(usize),
    T(Tensor),
}

/// Helper function to retrieve the rank of a tensor.
fn rank(scope: &mut Scope, x: Tensor) -> Result<(RankVal, bool)> {
    let rank = x.get_shape(scope).dims();
    if let Some(rank) = rank {
        Ok((RankVal::S(rank), true))
    } else {
        Ok((RankVal::T(array_ops::rank(scope, x, "")?), false))
    }
}

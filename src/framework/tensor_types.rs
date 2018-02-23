use std::iter::IntoIterator;

use TensorShape;
use tf::TensorType;

use super::*;

pub trait TensorOps {
    fn into_tensor(self, context: &mut Scope) -> Tensor;
}

//pub type TensorDef<'a, T, TeS> = (&'a [T], &'a [TeS]);

macro_rules! impl_tensor_ops {
    ($T:ty) => {
        impl TensorOps for $T
        {
            fn into_tensor(self, context: &mut Scope) -> Tensor {
                context.constant(&[self], &[] as &[i32], "").unwrap().into()
            }
        }

        impl<'a> TensorOps for &'a [$T] {
            fn into_tensor(self, context: &mut Scope) -> Tensor {
                context.constant(self, [self.len() as i64].as_ref(), "").unwrap().into()
            }
        }

        impl<'a, TeS: ShapeSize> TensorOps for (&'a [$T], &'a [TeS]) {
            fn into_tensor(self, context: &mut Scope) -> Tensor {
                context.constant(self.0, self.1, "").unwrap().into()
            }
        }
    };

    (Trait: $TR:tt) => {
        impl<T: $TR> TensorOps for T
        {
            fn into_tensor(self, context: &mut Scope) -> Tensor {
                context.constant(&[self], &[] as &[i32], "").unwrap().into()
            }
        }

        impl<'a, T: $TR> TensorOps for &'a [T] {
            fn into_tensor(self, context: &mut Scope) -> Tensor {
                context.constant(self, [self.len() as i64].as_ref(), "").unwrap().into()
            }
        }

        impl<'a, T: $TR, TeS: ShapeSize> TensorOps for (&'a [T], &'a [TeS]) {
            fn into_tensor(self, context: &mut Scope) -> Tensor {
                context.constant(self.0, self.1, "").unwrap().into()
            }
        }
    };

    (Gen: $T:ty) => {
        impl TensorOps for $T {
            fn into_tensor(self, _scope: &mut Scope) -> Tensor {
                self.into()
            }
        }

        impl<'a> TensorOps for &'a $T {
            fn into_tensor(self, _scope: &mut Scope) -> Tensor {
                (*self).into()
            }
        }
    }
}

impl_tensor_ops!(f32);
impl_tensor_ops!(f64);
impl_tensor_ops!(i16);
impl_tensor_ops!(Trait: ShapeSize);
impl_tensor_ops!(u8);
impl_tensor_ops!(i8);
impl_tensor_ops!(String);
impl_tensor_ops!(::Complex32);
impl_tensor_ops!(::Complex64);
impl_tensor_ops!(::QUInt8);
impl_tensor_ops!(::QUInt16);
impl_tensor_ops!(::QInt16);
impl_tensor_ops!(::QInt32);
impl_tensor_ops!(::BFloat16);
impl_tensor_ops!(Gen: Tensor);
impl_tensor_ops!(Gen: Constant);
impl_tensor_ops!(Gen: Variable);

///// Shape related traits /////

/// Marker for types which can be used to define a tensor shape.
pub trait ShapeSize: TensorType + Copy {
    fn as_i32(self) -> i32;
    fn as_i64(self) -> i64;
    fn as_u64(self) -> u64;
}

impl ShapeSize for i32 {
    fn as_i32(self) -> i32 {
        self
    }

    fn as_i64(self) -> i64 {
        self as i64
    }

    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl ShapeSize for i64 {
    fn as_i32(self) -> i32 {
        self as i32
    }

    fn as_i64(self) -> i64 {
        self
    }

    fn as_u64(self) -> u64 {
        self as u64
    }
}

/// Methods to determine and get the shape of a tensor if it's actually defined.
pub trait ShapeOps
where
    Self: Sized,
{
    /// Returns True iff self is compatible with other.
    ///
    /// Two possibly-partially-defined shapes are compatible if there exists a fully-defined shape
    /// that both shapes can represent. Thus, compatibility allows the shape inference code
    /// to reason about partially-defined shapes. For example:
    ///
    /// Shape(None) is compatible with all shapes.
    ///
    /// Shape([None, None]) is compatible with all two-dimensional shapes,
    /// such as Shape([32, 784]), and also Shape(None).
    /// It is not compatible with, for example, Shape([None])
    /// or Shape([None, None, None]).
    ///
    /// Shape([32, None]) is compatible with all two-dimensional shapes with size 32
    /// in the 0th dimension, and also Shape([None, None]) and Shape(None).
    /// It is not compatible with, for example, Shape([32]), Shape([32, None, 1])
    /// or Shape([64, None]).
    ///
    /// Shape([32, 784]) is compatible with itself, and also Shape([32, None]),
    /// Shape([None, 784]), Shape([None, None]) and Shape(None).
    /// It is not compatible with, for example, Shape([32, 1, 784]) or Shape([None]).
    ///
    /// The compatibility relation is reflexive and symmetric, but not transitive.
    /// For example, Shape([32, 784]) is compatible with Shape(None),
    /// and Shape(None) is compatible with Shape([4, 4]),
    /// but Shape([32, 784]) is not compatible with Shape([4, 4]).
    fn is_compatible_with(&self, other: &TensorShape) -> bool;
    fn merge_with(&self, other: &TensorShape) -> Result<TensorShape>;
    fn is_fully_defined(&self) -> bool;

    fn get_dim_size(&self, idx: usize) -> Option<i64>;
    fn definition_i64(&self) -> Option<Vec<i64>>;
    fn definition_u64(&self) -> Option<Vec<u64>>;
    fn to_shape(&self) -> TensorShape;
    fn slice(&self, start: usize, end: Option<usize>) -> Result<TensorShape>;
}

impl ShapeOps for TensorShape {
    fn is_compatible_with(&self, other: &TensorShape) -> bool {
        let o_dnum = other.dims();
        let s_dnum = self.dims();
        if o_dnum.is_none() || s_dnum.is_none() {
            return true;
        }

        let o_def = o_dnum.unwrap();
        let s_def = s_dnum.unwrap();
        if o_def != s_def {
            return false;
        }

        let any_diff = (0..s_def).into_iter().any(|idx| {
            let s_dim = self[idx];
            let o_dim = other[idx];
            if s_dim.is_none() || o_dim.is_none() {
                false
            } else {
                let s_dim = s_dim.unwrap();
                let o_dim = o_dim.unwrap();
                s_dim != o_dim
            }
        });
        !any_diff
    }

    fn merge_with(&self, other: &TensorShape) -> Result<TensorShape> {
        let s_def: Option<Vec<_>> = self.clone().into();
        let o_def: Option<Vec<_>> = other.clone().into();
        if s_def.is_none() {
            return Ok(other.clone());
        } else if o_def.is_none() {
            return Ok(self.clone());
        }

        let s_def = s_def.unwrap();
        let o_def = o_def.unwrap();
        let def: Result<_> = s_def
            .into_iter()
            .zip(o_def.into_iter())
            .map(|(s_dim, o_dim)| {
                if let Some(s_dim) = s_dim {
                    if let Some(o_dim) = o_dim {
                        if s_dim == o_dim {
                            Ok(Some(s_dim))
                        } else {
                            Err(Error::from(
                                "cannot merge dimensions of a shape with unequal values.",
                            ))
                        }
                    } else {
                        Ok(Some(s_dim))
                    }
                } else if let Some(o_dim) = o_dim {
                    Ok(Some(o_dim))
                } else {
                    Ok(None)
                }
            })
            .collect();
        Ok(TensorShape::from(Some(def?)))
    }

    fn is_fully_defined(&self) -> bool {
        if let Some(dim_num) = self.dims() {
            for dim in 0..dim_num {
                if self[dim].is_none() {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    fn get_dim_size(&self, idx: usize) -> Option<i64> {
        if let Some(dim_num) = self.dims() {
            if let Some(n) = self[idx] {
                Some(n)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn definition_i64(&self) -> Option<Vec<i64>> {
        let mut def = vec![];
        if let Some(dim_num) = self.dims() {
            for dim in 0..dim_num {
                if let Some(n) = self[dim] {
                    def.push(n);
                } else {
                    return None;
                }
            }
            Some(def)
        } else {
            None
        }
    }

    fn definition_u64(&self) -> Option<Vec<u64>> {
        let mut def = vec![];
        if let Some(dim_num) = self.dims() {
            for dim in 0..dim_num {
                if let Some(n) = self[dim] {
                    def.push(n as u64);
                } else {
                    return None;
                }
            }
            Some(def)
        } else {
            None
        }
    }

    fn to_shape(&self) -> TensorShape {
        self.clone()
    }

    fn slice(&self, start: usize, end: Option<usize>) -> Result<TensorShape> {
        let def: Option<Vec<Option<i64>>> = self.clone().into();
        if let Some(def) = def {
            if let Some(end) = end {
                Ok(TensorShape::from(Some((&def[start..]).to_vec())))
            } else {
                Ok(TensorShape::from(Some((&def[start..]).to_vec())))
            }
        } else {
            Err(Error::from(ErrorKind::UndefinedTensorShape))
        }
    }
}

impl<'a, TeS> ShapeOps for &'a [TeS]
where
    TeS: ShapeSize,
{
    fn is_compatible_with(&self, other: &TensorShape) -> bool {
        let self_shape = self.to_shape();
        self_shape.is_compatible_with(other)
    }

    fn merge_with(&self, other: &TensorShape) -> Result<TensorShape> {
        let s_shape: TensorShape = self.to_shape();
        s_shape.merge_with(other)
    }

    fn is_fully_defined(&self) -> bool {
        true
    }

    fn get_dim_size(&self, idx: usize) -> Option<i64> {
        if let Some(dim) = self.get(idx) {
            Some(dim.as_i64())
        } else {
            None
        }
    }

    fn definition_i64(&self) -> Option<Vec<i64>> {
        Some(self.iter().map(|x| x.as_i64()).collect())
    }

    fn definition_u64(&self) -> Option<Vec<u64>> {
        Some(self.iter().map(|x| x.as_u64()).collect())
    }

    fn to_shape(&self) -> TensorShape {
        TensorShape::from(Some(self.iter().map(|x| Some(x.as_i64())).collect()))
    }

    fn slice(&self, start: usize, end: Option<usize>) -> Result<TensorShape> {
        if let Some(end) = end {
            let def = self[start..end].iter().map(|x| Some(x.as_i64())).collect();
            Ok(TensorShape::from(Some(def)))
        } else {
            let def = self[start..].iter().map(|x| Some(x.as_i64())).collect();
            Ok(TensorShape::from(Some(def)))
        }
    }
}

pub trait DTypeOps {
    /// Returns whether this is a (non-quantized, real) floating point type.
    fn is_floating(&self) -> bool;
    /// Returns whether this is a (non-quantized) integer type.
    fn is_integer(&self) -> bool;
    /// Returns whether this is a complex floating point type.
    fn is_complex(&self) -> bool;
}

impl DTypeOps for DataType {
    fn is_floating(&self) -> bool {
        match *self {
            DataType::Float | DataType::Double | DataType::BFloat16 => true,
            _ => false,
        }
    }

    fn is_integer(&self) -> bool {
        match *self {
            DataType::Int32
            | DataType::UInt8
            | DataType::Int16
            | DataType::Int8
            | DataType::Int64
            | DataType::UInt16 => true,
            _ => false,
        }
    }

    fn is_complex(&self) -> bool {
        match *self {
            DataType::Complex64 | DataType::Complex128 => true,
            _ => false,
        }
    }
}

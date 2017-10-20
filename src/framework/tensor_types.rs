use Shape;
use tf::TensorType;

use super::*;

pub trait TensorOps {
    fn into_tensor<S: AsRef<Path>>(self, scope: &mut Scope, name: S) -> Tensor;
}

macro_rules! impl_tensor_ops {
    ($T:ty) => {
        impl TensorOps for $T
        {
            fn into_tensor<S: AsRef<Path>>(self, scope: &mut Scope, name: S) -> Tensor {
                scope.constant(&[self], &[] as &[i32], name).unwrap().into()
            }
        }

        impl<'a> TensorOps for &'a [$T] {
            fn into_tensor<S: AsRef<Path>>(self, scope: &mut Scope, name: S) -> Tensor {
                scope.constant(self, &[self.len() as i32] as &[i32], name).unwrap().into()
            }
        }
    }
}

impl_tensor_ops!(i32);
impl_tensor_ops!(i64);
impl_tensor_ops!(f32);
impl_tensor_ops!(f64);

impl TensorOps for Tensor {
    fn into_tensor<S: AsRef<Path>>(self, _scope: &mut Scope, _name: S) -> Tensor {
        self
    }
}

impl TensorOps for Constant {
    fn into_tensor<S: AsRef<Path>>(self, _scope: &mut Scope, _name: S) -> Tensor {
        self.into()
    }
}

impl TensorOps for Variable {
    fn into_tensor<S: AsRef<Path>>(self, _scope: &mut Scope, _name: S) -> Tensor {
        self.into()
    }
}


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
pub trait DefinedShape {
    fn is_fully_defined(&self) -> bool;
    fn definition_u64(&self) -> Option<Vec<u64>>;
    fn definition_i64(&self) -> Option<Vec<i64>>;
}

impl DefinedShape for Shape {
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
}

/// Return a tensor shape from self.
pub trait IntoShape {
    fn to_shape(&self) -> Shape;
}

impl<'a, TeS> IntoShape for &'a [TeS]
where
    TeS: ShapeSize,
{
    fn to_shape(&self) -> Shape {
        Shape::from(Some(self.iter().map(|x| Some(x.as_i64())).collect()))
    }
}

impl IntoShape for Shape {
    fn to_shape(&self) -> Shape {
        self.clone()
    }
}

pub(crate) fn shape_as_u64<T: ShapeSize>(dims: &[T]) -> Vec<u64> {
    dims.iter().map(|x| x.as_u64()).collect()
}

pub(crate) fn shape_as_i64<T: ShapeSize>(dims: &[T]) -> Vec<i64> {
    dims.iter().map(|x| x.as_i64()).collect()
}

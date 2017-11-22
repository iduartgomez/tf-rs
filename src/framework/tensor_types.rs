use Shape;
use tf::TensorType;

use super::*;

pub trait TensorOps {
    fn into_tensor(self, scope: &mut Scope) -> Tensor;
}

//pub type TensorDef<'a, T, TeS> = (&'a [T], &'a [TeS]);

macro_rules! impl_tensor_ops {
    ($T:ty) => {
        impl TensorOps for $T
        {
            fn into_tensor(self, scope: &mut Scope) -> Tensor {
                scope.constant(&[self], &[] as &[i32], "").unwrap().into()
            }
        }

        impl<'a> TensorOps for &'a [$T] {
            fn into_tensor(self, scope: &mut Scope) -> Tensor {
                scope.constant(self, &[self.len() as i64], "").unwrap().into()
            }
        }

        impl<'a, TeS: ShapeSize> TensorOps for (&'a [$T], &'a [TeS]) {
            fn into_tensor(self, scope: &mut Scope) -> Tensor {
                scope.constant(self.0, self.1, "").unwrap().into()
            }
        }
    };
    (Trait: $TR:tt) => {
        impl<T: $TR> TensorOps for T
        {
            fn into_tensor(self, scope: &mut Scope) -> Tensor {
                scope.constant(&[self], &[] as &[i32], "").unwrap().into()
            }
        }

        impl<'a, T: $TR> TensorOps for &'a [T] {
            fn into_tensor(self, scope: &mut Scope) -> Tensor {
                scope.constant(self, &[self.len() as i64], "").unwrap().into()
            }
        }

        impl<'a, T: $TR, TeS: ShapeSize> TensorOps for (&'a [T], &'a [TeS]) {
            fn into_tensor(self, scope: &mut Scope) -> Tensor {
                scope.constant(self.0, self.1, "").unwrap().into()
            }
        }
    };
}

/*
#[test]
fn tensor_def() {
    use framework::tensor_types::TensorOps;
    let ctxt = &mut Scope::new();
    let def0 = ([0_i32, 1].as_ref(), [2_i32].as_ref()); // as TensorDef<i32, i32>;
    let def1 = 3_i32;
    def0.into_tensor(ctxt);
    def1.into_tensor(ctxt);
}
*/

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

impl TensorOps for Tensor {
    fn into_tensor(self, _scope: &mut Scope) -> Tensor {
        self
    }
}

impl TensorOps for Constant {
    fn into_tensor(self, _scope: &mut Scope) -> Tensor {
        self.into()
    }
}

impl TensorOps for Variable {
    fn into_tensor(self, _scope: &mut Scope) -> Tensor {
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
pub trait ShapeOps
where
    Self: Sized,
{
    fn is_compatible_with(&self, other: &Self) -> bool;
    fn merge_with<Idx>(&self, other: &Shape, range: Option<::std::ops::Range<Idx>>)
        -> Result<Self>;
    fn is_fully_defined(&self) -> bool;
    fn get_dim_size(&self, idx: usize) -> Option<i64>;
    fn definition_i64(&self) -> Option<Vec<i64>>;
    fn definition_u64(&self) -> Option<Vec<u64>>;
}

impl ShapeOps for Shape {
    fn is_compatible_with(&self, other: &Shape) -> bool {
        unimplemented!()
    }

    fn merge_with<Idx>(
        &self,
        other: &Shape,
        range: Option<::std::ops::Range<Idx>>,
    ) -> Result<Self> {
        unimplemented!()
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

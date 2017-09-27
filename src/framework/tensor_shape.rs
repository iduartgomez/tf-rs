use Shape;
use tf::TensorType;

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

pub trait IntoShape {
    fn into_shape(&self) -> Shape;
}

impl<'a, TeS> IntoShape for &'a [TeS] 
    where TeS: ShapeSize
{
    fn into_shape(&self) -> Shape {
        Shape::from(Some(self.iter().map(|x| Some(x.as_i64())).collect()))
    }
}

impl IntoShape for Shape {
    fn into_shape(&self) -> Shape {
        self.clone()
    }
}

pub trait ShapeSize: TensorType + Copy {
    fn as_i64(self) -> i64;
    fn as_u64(self) -> u64;
}

impl ShapeSize for i32 {
    fn as_i64(self) -> i64 {
        self as i64
    }

    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl ShapeSize for i64 {
    fn as_i64(self) -> i64 {
        self
    }

    fn as_u64(self) -> u64 {
        self as u64
    }
}

/*
pub fn shape_from_dims<T: ShapeSize>(dims: &[T]) -> Shape {
    Shape::from(Some(dims.iter().map(|x| Some(x.as_i64())).collect::<Vec<_>>()),)
}
*/

pub fn shape_as_u64<T: ShapeSize>(dims: &[T]) -> Vec<u64> {
    dims.iter().map(|x| x.as_u64()).collect()
}

pub fn shape_as_i64<T: ShapeSize>(dims: &[T]) -> Vec<i64> {
    dims.iter().map(|x| x.as_i64()).collect()
}

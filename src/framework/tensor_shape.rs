use Shape;
use tf::TensorType;

pub trait DefinedShape {
    fn is_fully_defined(&self) -> bool;
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

pub fn shape_from_dims<T: ShapeSize>(dims: &[T]) -> Shape {
    Shape::from(Some(dims.iter().map(|x| Some(x.as_i64())).collect::<Vec<_>>()),)
}

pub fn shape_as_u64<T: ShapeSize>(dims: &[T]) -> Vec<u64> {
    dims.iter().map(|x| x.as_u64()).collect()
}

pub fn shape_as_i64<T: ShapeSize>(dims: &[T]) -> Vec<i64> {
    dims.iter().map(|x| x.as_i64()).collect()
}


// TODO: this is very inefficient, implement efficient Clone for type
// at the very least don't initialize with zeros
/*
macro_rules! clone_tensor {
    ($val:ident) => {{
        let mut copy = TypedTensor::new($val.dims());
        for (i, x) in $val.iter().enumerate() { 
            copy[i] = *x;
        }
        copy
    }}
}
*/

macro_rules! name_cmp {
    ($name:ident, $cmp:expr) => ( $name.as_ref().to_str().unwrap() == $cmp )
}

macro_rules! tensor_output_op {
    ($val:expr; $exec:path[$($args:tt,)*]) => (
        match *$val {
            TensorContent::Float(ref val) => $exec($($args,)* val),
            TensorContent::Double(ref val) => $exec($($args,)* val),
            TensorContent::Int32(ref val) => $exec($($args,)* val),
            TensorContent::UInt8(ref val) => $exec($($args,)* val),
            TensorContent::Int16(ref val) => $exec($($args,)* val),
            TensorContent::Int8(ref val) => $exec($($args,)* val),
            TensorContent::Int64(ref val) => $exec($($args,)* val),
            TensorContent::Bool(ref val) => $exec($($args,)* val),
            _ => unimplemented!()
        }
    );
}

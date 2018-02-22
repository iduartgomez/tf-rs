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
            TensorContent::String(ref val) => $exec($($args,)* val),
            TensorContent::QUInt8(ref val) => $exec($($args,)* val),
            TensorContent::QUInt16(ref val) => $exec($($args,)* val),
            TensorContent::QInt16(ref val) => $exec($($args,)* val),
            TensorContent::QInt32(ref val) => $exec($($args,)* val),
            TensorContent::BFloat16(ref val) => $exec($($args,)* val),
            TensorContent::Complex64(ref val) => $exec($($args,)* val),
            TensorContent::Complex128(ref val) => $exec($($args,)* val),
        }
    );
}

macro_rules! dtype_to_const {
    ($context:ident; $dtype:expr; $val:expr; $shape:expr; $name:expr) => (
        match $dtype {
            DataType::Bool => $context.constant($val, $shape, $name),
            DataType::Double => $context.constant($val, $shape, $name),
            DataType::Float => $context.constant($val, $shape, $name),
            DataType::Int32 => $context.constant($val, $shape, $name),
            DataType::UInt8 => $context.constant($val, $shape, $name),
            DataType::Int16 => $context.constant($val, $shape, $name),
            DataType::Int8 => $context.constant($val, $shape, $name),
            DataType::Int64 => $context.constant($val, $shape, $name),
            DataType::String => $context.constant($val, $shape, $name),
            DataType::QUInt8 => $context.constant($val, $shape, $name),
            DataType::QInt16 => $context.constant($val, $shape, $name),
            DataType::QUInt16 => $context.constant($val, $shape, $name),
            DataType::QInt32 => $context.constant($val, $shape, $name),
            DataType::BFloat16 => $context.constant($val, $shape, $name),
            DataType::Complex64 => $context.constant($val, $shape, $name),
            DataType::Complex128 => $context.constant($val, $shape, $name),
            _ => return Err(Error::from(ErrorKind::Stub)),
        }
    )
}

//! Additional ops and models ported from TensorFlow Python lib.

use std::path::{Path, PathBuf};

use tf::TensorType;

use super::{DataType, Graph, OperationData, Output, Status, TypedTensor};
pub use super::framework::*;
pub(crate) use self::dtype_traits::*;

pub(crate) mod dtype_traits {
    use tf::TensorType;

    pub trait Float: TensorType {
        fn as_float(self) -> f32;
        fn as_double(self) -> f64;
    }

    impl Float for f32 {
        fn as_float(self) -> f32 {
            self
        }

        fn as_double(self) -> f64 {
            self as f64
        }
    }

    impl Float for f64 {
        fn as_float(self) -> f32 {
            self as f32
        }

        fn as_double(self) -> f64 {
            self
        }
    }
}

////// Macros //////

macro_rules! generate_name {
    (is_none: $name:ident) => (
        if name_cmp!($name, "") {
            None
        } else {
            Some($name.as_ref().to_owned())
        }
    );
    (ret: $name:expr) => (
        if let Some(ref name) = *$name {
            Some(name.as_path())
        } else {
            None
        }
    )
}

macro_rules! impl_into_ident {
    ($name:ident) => {
        impl<'a> Into<NodeIdent> for $name<'a> {
            fn into(self) -> NodeIdent {
                self.ident
            }
        }
        
        impl<'a> Into<NodeIdent> for &'a $name<'a> {
            fn into(self) -> NodeIdent {
                self.ident
            }
        }

        impl<'a> GetIdent for $name<'a> {
            fn get_ident(&self) -> NodeIdent {
                self.ident
            }
        }
    }
}

/// Macro for creating new operations.
///
/// __BINARY__: Define a binary (takes two inputs, and returns one output) operation,
/// with _n_ => 0 extra attributes, and _m_ => 0 extra functions.
///
/// Provide a custom constructor or use the default one calling `BIN CONSTRUCTOR`
/// variant of this same macro.
///
/// __UNARY__: Define a unary (takes one input, returns one output) operation,
/// with _n_ => 0 extra attributes, and _m_ => 0 extra functions.
///
/// Provide a custom constructor or use the default one calling `BIN CONSTRUCTOR`
/// variant of this same macro.
macro_rules! add_new_op {
    (
        $name:tt, 
        constructor: [$($constructor:tt)*],
        digest: [$($digest:tt)*],
        extra_funcs: [$($funcs:tt)*], 
        extra_attr: [$($attr_name:ident: $attr_ty:ty),*],
        output: [$($output:tt)*],
    ) => {
        #[derive(Debug, Clone)]
        struct $name<'a> {
            ident: NodeIdent,
            elements: Vec<Tensor>,
            name: Option<PathBuf>,
            /// attr_name, is_list, attr_value
            attributes: Vec<(&'a str, bool, Attribute<'a>)>,
            input_lists: Vec<(usize, Vec<Tensor>)>,
            $( $attr_name: $attr_ty, )*
        }

        impl<'a> $name<'a> {
            $($constructor)*
            $($funcs)*
        }

        impl<'a> Operation<'a> for $name<'a> {
            type Outputs = $($output)*;
            add_new_op!(CORE_FN: $name);
            add_new_op!($($digest)*);
        }

        impl_into_ident!($name);
    };
    // Generic constructor for unary ops.
    (UNARY CONSTRUCTOR: $name:ident, Init: [$($attr_name:ident: $attr_init:expr),*]) => {
        fn new<S: AsRef<Path>>(x: Tensor, name: S) -> Result<$name<'a>, ::Error> {
            Ok(
                $name {
                    ident: NodeIdent::new(),
                    elements: vec![x],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: vec![],
                    $($attr_name: $attr_init),*
                },
            )
        }
    };
    // Generic constructor for binary ops.
    (BIN CONSTRUCTOR: $name:ident, Init: [$($attr_name:ident: $attr_init:expr),*]) => {
        fn new<S: AsRef<Path>>(x: Tensor, y: Tensor, name: S) -> Result<$name<'a>, ::Error> {
            Ok(
                $name {
                    ident: NodeIdent::new(),
                    elements: vec![x, y],
                    name: generate_name!(is_none: name),
                    attributes: vec![],
                    input_lists: vec![],
                    $($attr_name: $attr_init),*
                },
            )
        }
    };

    // digest fn for context
    (DEFAULT_DIGEST: $name:tt, $infer_dtype:tt) => {
        #[doc(hidden)]
        fn digest(
            self,
            context: &mut Scope,
            op: OperationData,
        ) -> Result<Self::Outputs, ::Error> {
            let (ident, idtype, dtype) = add_new_op!(
                REGISTER_TENSOR: (self, context, op); $name, $infer_dtype);
            let tensor = Tensor {
                ident,
                idtype,
                dtype,
                idx: 0,
                initializer: None,
            };
            match context.control_context {
                ControlFlow::CondContext(ref mut cond) => {
                    cond.values.insert(ident);
                    cond.external_values.insert(ident, tensor); 
                }
                ControlFlow::WhileContext(ref mut cond) => {
                    cond.values.insert(ident);
                    cond.external_values.insert(ident, tensor); 
                }
                ControlFlow::None => {}
            }
            Ok(tensor)
        }
    };
    (DIGEST: $($digest:tt)*) => { $($digest)* };

    (REGISTER_TENSOR: ($SELF:ident, $context:ident, $op:ident); $name:tt, $infer_dtype:tt) => {{
        let ident = NodeIdent::new();
        let dtype = add_new_op!($infer_dtype $SELF);
        let idtype = IdType::Operation(stringify!($name));
        let full_name = $context.resolve_tensor_name($SELF.get_op_name(), idtype, false)?;
        let shape = {
            let g = &*$context.graph.borrow();
            g.tensor_shape(
                    Output {
                        operation: $op.clone(),
                        index: 0,
                    },
                )?
        };
        {
            let reg = &mut *$context.registry.borrow_mut();
            $context.own_scope.ops.push((full_name.clone(), ident));
            reg.insert(
                ident,
                TensorData {
                    full_name,
                    dtype,
                    idtype,
                    data_origin: ($op, 0),
                    shape,
                },
            );
        }
        (ident, idtype, dtype)
    }};

    (REGISTER_AS_OP: ($SELF:ident, $context:ident, $op:ident); $name:tt) => {{
        let idtype = IdType::Operation(stringify!($name));
        let full_name = $context.resolve_tensor_name($SELF.get_op_name(), idtype, false)?;
        {
            let reg = &mut *$context.ops.borrow_mut();
            $context.own_scope.ops.push((full_name.clone(), $SELF.ident));
            reg.insert($SELF.ident, $op);
        }
    }};

    // extra funcs:
    (CORE_FN: $op_name:tt) => {
        fn get_op_type_name(&self) -> &'static str {
            stringify!($op_name)
        }

        fn get_op_name(&self) -> Option<&Path> {
            generate_name!(ret: &self.name)
        }

        fn fetch_inputs(&self) -> &[Tensor] {
            &self.elements
        }

        fn fetch_input_lists(&self) -> &[(usize, Vec<Tensor>)] {
            &self.input_lists
        }

        fn fetch_attributes<'s>(&'s self) 
            -> &'s [(&str, bool, Attribute<'a>)] 
        {
            &self.attributes
        }
    };
    // DataType inference:
    (INPUT0 $s:ident) => ($s.elements[0].dtype);
    (DTYPE_ATTR $s:ident) => ($s.output_type);
    (NONE $s:ident) => (DataType::Resource)
}

#[allow(unused_macros)]
macro_rules! test_suite {
    (run_op: [$($op:ident),+]; $context:ident, input: {$($arg:tt),*}) => {{
        use client::ClientSession;
        let mut session = ClientSession::new(&mut $context).unwrap();
        session.fetch(vec![$($op),+]);
        $( session.feed(vec![$arg]) )*
        session.run(None).unwrap()
    }};
    ($res:ident; assert: {$([$res_idx:expr; $ty:ident] == $cmp:expr),+}) => {{
        $(
            match $res[$res_idx] {
                TensorContent::$ty(ref val) => {
                    for (i, n) in (&$cmp).iter().enumerate() {
                        assert_eq!(val[i], *n);
                    }
                }
                _ => panic!("wrong type specified for this test")
            }
        )+
    }};
    ($res:ident; assert_len: {$([$res_idx:expr; $ty:ident] == $cmp:expr),+}) => {{
        $(
            match $res[$res_idx] {
                TensorContent::$ty(ref val) => {
                    assert_eq!(val.len(), $cmp)
                }
                _ => panic!("wrong type specified for this test")
            }
        )+
    }};
    (out: $op:expr, $idx:expr) => (
        Output {
            operation: $op,
            index: $idx,
        }
    )
}

pub(crate) mod array_ops;
pub use self::array_ops::*;

pub(crate) mod control_flow_ops;
pub use self::control_flow_ops::*;

pub(crate) mod init_ops;
pub use self::init_ops::*;

pub(crate) mod math_ops;
pub use self::math_ops::*;

pub(crate) mod state_ops;
pub use self::state_ops::*;

trait DTypeOps {
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
            DataType::Float |
            DataType::Double |
            DataType::BFloat16 => true,
            _ => false,
        }
    }

    fn is_integer(&self) -> bool {
        match *self {
            DataType::Int32 |
            DataType::UInt8 |
            DataType::Int16 |
            DataType::Int8 |
            DataType::Int64 |
            DataType::UInt16 => true,
            _ => false,
        }
    }

    fn is_complex(&self) -> bool {
        match *self {
            DataType::Complex64 |
            DataType::Complex128 => true,
            _ => false,
        }
    }
}

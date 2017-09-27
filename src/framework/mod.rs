//! Built-in helpers and utilities for interfacing with TensorFlow Rust and the C API.

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use uuid;
use tf::TensorType;

use super::{DataType, OperationData, OperationDescription, Shape, TypedTensor};

// Macros:

macro_rules! to_tf_tensor {
    [$values:expr; $shape:expr] => {{
        let mut tensor = TypedTensor::<T>::new($shape);
        for (i, v) in $values.iter().enumerate() {
            tensor[i] = v.clone();
        }
        tensor
    }};
}

// TODO: this is very inefficient, implement efficient Clone for type
// at the very least don't initialize with zeros
macro_rules! clone_tensor {
    ($val:ident) => {{
        let mut copy = TypedTensor::new($val.dims());
        for (i, x) in $val.iter().enumerate() { 
            copy[i] = x.clone();
        }
        copy
    }}
}

/////////////////////

mod scope;
pub use self::scope::*;

mod tensor_shape;
pub(crate) use self::tensor_shape::*;


#[doc(hidden)]
/// An interface to add and manipulate operations in the computation graph.
pub trait Operation<'a>
    where Self: Sized + Into<NodeIdent>
{
    type Outputs;

    fn get_op_type_name(&self) -> &'static str;
    /// Return the full qualified path name for this op, including scopes, if any.
    fn get_op_name(&self) -> Option<&Path>;
    fn fetch_inputs(&self) -> &[Tensor];
    /// List arguments are checked first, they include their position info in part of the return tuple.
    ///
    /// If there are any they are inserted during the operation construction when appropiate.
    fn fetch_input_lists(&self) -> &[(usize, Vec<Tensor>)];
    /// Get the attributes for this operation. Used while 'digesting' it.
    fn fetch_attributes<'s>(&'s self) -> &'s [(&str, bool, Attribute<'a>)];
    #[doc(hidden)]
    /// Consumes self and returns output. Used when installing the op into the context.
    fn digest(self, context: &mut Scope, op: OperationData) -> Result<Self::Outputs, ::Error>;
}

pub(crate) fn add_control_input<I, T>(op: &mut OperationDescription, control_inputs: I)
    where I: IntoIterator<Item = T>,
          T: ::std::ops::Deref<Target = OperationData>
{
    for ctrl in control_inputs.into_iter() {
        op.add_control_input(&*ctrl);
    }
}

/// This is a token to identify computation elements in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIdent(uuid::Uuid);

impl NodeIdent {
    pub(crate) fn new() -> NodeIdent {
        NodeIdent(uuid::Uuid::new_v4())
    }
}

impl GetIdent for NodeIdent {
    fn get_ident(&self) -> NodeIdent {
        self.clone()
    }
}

/// Get the identity token of an object.
pub trait GetIdent {
    fn get_ident(&self) -> NodeIdent;
}


#[derive(Debug, Clone)]
pub(crate) struct TensorData {
    /// fully qualified name, including the scope path
    pub full_name: PathBuf,
    pub dtype: DataType,
    pub idtype: IdType,
    /// operation & operation's output index
    pub data_origin: (OperationData, i32),
    pub shape: Shape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum IdType {
    Constant,
    Variable,
    Operation(&'static str),
    Placeholder,
}

#[derive(Debug, Clone)]
pub(crate) struct ControlOp {
    pub ident: NodeIdent,
    pub finished: OperationData,
    pub kind: ControlOpKind,
}

impl Hash for ControlOp {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ident.hash(state);
    }
}

impl PartialEq for ControlOp {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident
    }
}

impl Eq for ControlOp {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ControlOpKind {
    VarInitializer,
    Ops,
    Other,
}


/// Tensor
#[derive(Debug, Clone, Copy)]
pub struct Tensor {
    pub(crate) ident: NodeIdent,
    pub(crate) idtype: IdType,
    pub(crate) dtype: DataType,
    /// Index of this tensor in the source operation output.
    pub(crate) idx: i32,
}

impl Tensor {
    /// Performs an assign operation for this tensor.
    pub fn new<TeS, T>(context: &mut Scope, values: &[T], shape: &[TeS]) -> Tensor
        where T: TensorType,
              TeS: ShapeSize
    {
        unimplemented!()
    }

    pub fn get_initializer(&self, context: &Scope) -> Result<Tensor, ::Error> {
        if !self.is_ref() {
            return Err(::Error::Stub);
        }
        let registry = &*context.registry.borrow();
        let tensor_data = &registry[&self.ident];

        unimplemented!()
    }

    pub fn get_name(&self, context: &Scope) -> String {
        let registry = &*context.registry.borrow();
        let (ref op, _) = registry[&self.ident].data_origin;
        op.name().unwrap()
    }

    pub fn get_shape(&self, context: &Scope) -> Shape {
        let registry = &*context.registry.borrow();
        registry[&self.ident].shape.clone()
    }

    pub fn is_ref(&self) -> bool {
        if self.idtype == IdType::Variable {
            true
        } else {
            false
        }
    }

    pub fn write(&mut self, context: &mut Scope, tensor: Tensor) -> Result<(), ::Error> {
        unimplemented!()
    }
}

impl GetIdent for Tensor {
    fn get_ident(&self) -> NodeIdent {
        self.ident
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident
    }
}

impl Eq for Tensor {}

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ident.hash(state);
    }
}

impl Into<NodeIdent> for Tensor {
    fn into(self) -> NodeIdent {
        self.ident
    }
}

impl<'a> Into<NodeIdent> for &'a Tensor {
    fn into(self) -> NodeIdent {
        self.ident
    }
}


/// Constant
#[derive(Debug, Clone, Copy)]
pub struct Constant {
    ident: NodeIdent,
    dtype: DataType,
}

impl Constant {
    pub fn new<TeS, T>(context: &mut Scope, value: &[T], shape: &[TeS]) -> Constant
        where T: TensorType,
              TeS: ShapeSize
    {
        let name = context.resolve_tensor_name(None, IdType::Constant, false).unwrap();
        context.constant(name, value, shape).unwrap()
    }

    pub fn get_name(&self, context: &Scope) -> String {
        let registry = &*context.registry.borrow();
        let (ref op, _) = registry[&self.ident].data_origin;
        op.name().unwrap()
    }

    pub fn get_shape(&self, shape: &Scope) -> Shape {
        let registry = &*shape.registry.borrow();
        registry[&self.ident].shape.clone()
    }
}

/// Treat a constant as an initilized variable. Useful for operating with the context manager.
impl Into<Tensor> for Constant {
    fn into(self) -> Tensor {
        let Constant { ident, dtype, .. } = self;
        Tensor {
            ident,
            dtype,
            idtype: IdType::Constant,
            idx: 0,
        }
    }
}

impl Hash for Constant {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ident.hash(state);
    }
}

impl Into<NodeIdent> for Constant {
    fn into(self) -> NodeIdent {
        self.ident
    }
}

impl<'a> Into<NodeIdent> for &'a Constant {
    fn into(self) -> NodeIdent {
        self.ident
    }
}


/// Variable
#[derive(Debug, Clone, Copy)]
pub struct Variable {
    ident: NodeIdent,
    pub(crate) dtype: DataType,
    initializer: NodeIdent,
    /// index of the output source operation
    idx: i32,
}

impl Variable {
    pub fn new<TeS, T>(context: &mut Scope, initial_value: &[T], shape: &[TeS]) -> Variable
        where T: TensorType,
              TeS: ShapeSize
    {
        let name = context.resolve_tensor_name(None, IdType::Variable, false).unwrap();
        unimplemented!()
    }

    pub fn get_name(&self, context: &Scope) -> String {
        let registry = &*context.registry.borrow();
        let (ref op, _) = registry[&self.ident].data_origin;
        op.name().unwrap()
    }

    pub fn get_shape(&self, shape: &Scope) -> Shape {
        let registry = &*shape.registry.borrow();
        registry[&self.ident].shape.clone()
    }

    pub fn from_tensor(self, context: &Scope) -> Result<Tensor, ::Error> {
        unimplemented!()
    }
}

impl GetIdent for Variable {
    fn get_ident(&self) -> NodeIdent {
        self.ident
    }
}

impl Hash for Variable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ident.hash(state);
    }
}

impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident
    }
}

impl Eq for Variable {}

impl Into<Tensor> for Variable {
    fn into(self) -> Tensor {
        let Variable { ident, dtype, idx, .. } = self;
        Tensor {
            ident,
            dtype,
            idtype: IdType::Variable,
            idx,
        }
    }
}

impl Into<NodeIdent> for Variable {
    fn into(self) -> NodeIdent {
        self.ident
    }
}

impl<'a> Into<NodeIdent> for &'a Variable {
    fn into(self) -> NodeIdent {
        self.ident
    }
}


#[derive(Debug, Clone)]
pub struct TensorArray {
    pub flow: Tensor,
}

impl TensorArray {
    pub fn size(&self) -> usize {
        unimplemented!()
    }

    pub fn write(&mut self, idx: usize, value: Tensor) -> Self {
        unimplemented!()
    }

    pub fn from_flow(scope: &Scope, flow: &Tensor) -> Self {
        unimplemented!()
    }
}

impl Into<NodeIdent> for TensorArray {
    fn into(self) -> NodeIdent {
        unimplemented!()
    }
}


#[derive(Debug)]
pub enum TensorContent {
    Float(TypedTensor<f32>),
    Double(TypedTensor<f64>),
    Int32(TypedTensor<i32>),
    UInt8(TypedTensor<u8>),
    Int16(TypedTensor<i16>),
    Int8(TypedTensor<i8>),
    String(TypedTensor<String>),
    Int64(TypedTensor<i64>),
    Bool(TypedTensor<bool>),
}

impl Clone for TensorContent {
    fn clone(&self) -> TensorContent {
        match *self {
            TensorContent::Float(ref val) => TensorContent::Float(clone_tensor!(val)),
            TensorContent::Double(ref val) => TensorContent::Double(clone_tensor!(val)),
            TensorContent::Int32(ref val) => TensorContent::Int32(clone_tensor!(val)),
            TensorContent::UInt8(ref val) => TensorContent::UInt8(clone_tensor!(val)),
            TensorContent::Int16(ref val) => TensorContent::Int16(clone_tensor!(val)),
            TensorContent::Int8(ref val) => TensorContent::Int8(clone_tensor!(val)),
            TensorContent::String(ref val) => TensorContent::String(clone_tensor!(val)), 
            TensorContent::Int64(ref val) => TensorContent::Int64(clone_tensor!(val)),
            TensorContent::Bool(ref val) => TensorContent::Bool(clone_tensor!(val)),
        }
    }
}

macro_rules! unwrap_tfinput {
    ($variant:ident, $name:tt, $type:ty) => {
        pub fn $name(self) -> TypedTensor<$type> {
            match self {
                TensorContent::$variant(val) => val,
                _ => panic!()
            }
        } 
    }
}

macro_rules! tf_tensor_obj {
    ($type:ty, $id:ident) => {
        impl From<TypedTensor<$type>> for TensorContent {
            fn from(tensor: TypedTensor<$type>) -> Self {
                TensorContent::$id(tensor)
            }
        }
    }
}

tf_tensor_obj!(f32, Float);
tf_tensor_obj!(f64, Double);
tf_tensor_obj!(i32, Int32);
tf_tensor_obj!(u8, UInt8);
tf_tensor_obj!(i16, Int16);
tf_tensor_obj!(i8, Int8);
tf_tensor_obj!(i64, Int64);
tf_tensor_obj!(bool, Bool);


impl TensorContent {
    fn get_datatype(&self) -> DataType {
        match *self {
            TensorContent::Float(_) => DataType::Float,
            TensorContent::Double(_) => DataType::Double,
            TensorContent::Int32(_) => DataType::Int32,
            TensorContent::UInt8(_) => DataType::UInt8,
            TensorContent::Int16(_) => DataType::Int16,
            TensorContent::Int8(_) => DataType::Int8,
            TensorContent::Int64(_) => DataType::Int64,
            TensorContent::Bool(_) => DataType::Bool,
            TensorContent::String(_) => DataType::String,
        }
    }

    fn set_tensor_list_attr(
        new_op: &mut OperationDescription,
        name: &str,
        val: &[TensorContent],
    ) -> Result<(), ::Error> {
        match val[0].get_datatype() {
            DataType::Bool => new_op.set_attr_tensor_list(name, collect_bool_tensor(val))?,
            DataType::Double => new_op.set_attr_tensor_list(name, collect_double_tensor(val))?,
            DataType::Float => new_op.set_attr_tensor_list(name, collect_float_tensor(val))?,
            DataType::Int32 => new_op.set_attr_tensor_list(name, collect_i32_tensor(val))?,
            DataType::UInt8 => new_op.set_attr_tensor_list(name, collect_u8_tensor(val))?,
            DataType::Int16 => new_op.set_attr_tensor_list(name, collect_i16_tensor(val))?,
            DataType::Int8 => new_op.set_attr_tensor_list(name, collect_i8_tensor(val))?,
            DataType::Int64 => new_op.set_attr_tensor_list(name, collect_i64_tensor(val))?,
            _ => unimplemented!(),
        }
        Ok(())
    }

    fn set_tensor_attr(
        new_op: &mut OperationDescription,
        name: &str,
        val: &[TensorContent],
    ) -> Result<(), ::Error> {
        match val[0].get_datatype() {
            DataType::Bool => {
                new_op.set_attr_tensor(name, collect_bool_tensor(val).pop().unwrap())?
            }
            DataType::Double => {
                new_op.set_attr_tensor(name, collect_double_tensor(val).pop().unwrap())?
            }
            DataType::Float => {
                new_op.set_attr_tensor(name, collect_float_tensor(val).pop().unwrap())?
            }
            DataType::Int32 => {
                new_op.set_attr_tensor(name, collect_i32_tensor(val).pop().unwrap())?
            }
            DataType::UInt8 => new_op.set_attr_tensor(name, collect_u8_tensor(val).pop().unwrap())?,
            DataType::Int16 => {
                new_op.set_attr_tensor(name, collect_i16_tensor(val).pop().unwrap())?
            }
            DataType::Int8 => new_op.set_attr_tensor(name, collect_i8_tensor(val).pop().unwrap())?,
            DataType::Int64 => {
                new_op.set_attr_tensor(name, collect_i64_tensor(val).pop().unwrap())?
            }
            _ => unimplemented!(),
        }
        Ok(())
    }

    unwrap_tfinput!(Float, unwrap_float, f32);
    unwrap_tfinput!(Double, unwrap_double, f64);
    unwrap_tfinput!(Int32, unwrap_i32, i32);
    unwrap_tfinput!(UInt8, unwrap_u8, u8);
    unwrap_tfinput!(Int16, unwrap_i16, i16);
    unwrap_tfinput!(Int8, unwrap_i8, i8);
    unwrap_tfinput!(Int64, unwrap_i64, i64);
    unwrap_tfinput!(Bool, unwrap_bool, bool);
}

/// Enumerates possible operation attributes.
#[derive(Debug, Clone)]
pub enum Attribute<'a> {
    String(&'a [&'a str]),
    Int(&'a [i64]),
    Float(&'a [f32]),
    Bool(&'a [bool]),
    Type(&'a [DataType]),
    Shape(&'a [Shape]),
    Tensor(Vec<TensorContent>),
}

macro_rules! collect_tensors {
    ($variant:ident, $name:tt, $type:ty) => {
        fn $name(tensors: &[TensorContent]) -> Vec<TypedTensor<$type>> {
            tensors.iter().map(|x| {
                match *x {
                    TensorContent::$variant(ref val) => clone_tensor!(val),
                    _ => panic!()
                }
            }).collect::<Vec<_>>()
        }
    }
}

collect_tensors!(Float, collect_float_tensor, f32);
collect_tensors!(Double, collect_double_tensor, f64);
collect_tensors!(Int32, collect_i32_tensor, i32);
collect_tensors!(UInt8, collect_u8_tensor, u8);
collect_tensors!(Int16, collect_i16_tensor, i16);
collect_tensors!(Int8, collect_i8_tensor, i8);
collect_tensors!(Int64, collect_i64_tensor, i64);
collect_tensors!(Bool, collect_bool_tensor, bool);

impl<'a, T> From<Vec<TypedTensor<T>>> for Attribute<'a>
    where T: TensorType,
          TensorContent: From<TypedTensor<T>>
{
    fn from(src: Vec<TypedTensor<T>>) -> Attribute<'a> {
        Attribute::Tensor(src.into_iter().map(|x| TensorContent::from(x)).collect())
    }
}

macro_rules! attr_from_ty {
    ($variant:ident, $type:ty) => {
        impl<'a> From<&'a [$type]> for Attribute<'a> {
            fn from(src: &'a [$type]) -> Attribute<'a> {
                Attribute::$variant(src)
            }
        }
    };
}

attr_from_ty!(String, &'a str);
attr_from_ty!(Int, i64);
attr_from_ty!(Float, f32);
attr_from_ty!(Bool, bool);
attr_from_ty!(Type, DataType);
attr_from_ty!(Shape, Shape);

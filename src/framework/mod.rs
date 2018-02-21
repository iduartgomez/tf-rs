//! Built-in helpers and utilities for interfacing with TensorFlow Rust and the C API.

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use uuid;
use tf;

use super::{DataType, OperationData, OperationDescription, Shape, TypedTensor};
use errors::*;

// Macros:

macro_rules! to_typed_tensor {
    [$val:expr; $shape:expr] => {{
        TypedTensor::<T>::new($shape).with_values(&$val).unwrap()
    }}
}

macro_rules! clone_tensor {
    ($val:ident) => {{
        TypedTensor::new($val.dims()).with_values(&$val).unwrap()
    }}
}

/////////////////////

mod scope;
pub use self::scope::*;

mod tensor_types;
pub use self::tensor_types::{ShapeOps, ShapeSize, TensorOps};

#[doc(hidden)]
/// An interface to add and manipulate operations in the computation graph.
pub trait Operation<'a>
where
    Self: Sized + GetOp,
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
    fn digest(self, context: &mut Scope, op: OperationData) -> Result<Self::Outputs>;
}

pub(crate) fn add_control_input<I, T>(op: &mut OperationDescription, control_inputs: I)
where
    I: IntoIterator<Item = T>,
    T: ::std::ops::Deref<Target = OperationData>,
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

    #[allow(dead_code)]
    pub(crate) fn get_hash(&self) -> usize {
        use std::collections::hash_map::DefaultHasher;
        let hasher = &mut DefaultHasher::default();
        self.0.hash(hasher);
        hasher.finish() as usize
    }

    pub(crate) fn get_attr(&self, context: &Scope, name: &str) -> Option<FuncAttr> {
        unimplemented!()
    }

    pub fn get_outputs(&self, context: &Scope) -> Result<Vec<Tensor>> {
        let reg = &*context.registry.borrow();
        if context.ops.borrow().contains_key(self) {
            // first check if it's an operation
            Ok(reg.iter()
                .filter(|&(_, t)| &t.data_origin_id == self)
                .map(|(id, t)| {
                    let mut input = Tensor {
                        ident: *id,
                        dtype: t.dtype,
                        idtype: t.idtype.clone(),
                        initializer: None,
                        origin_op: None,
                        idx: t.data_origin.1,
                    };
                    if t.idtype.is_initialized() {
                        input.initializer = Some(t.data_origin_id);
                    } else {
                        input.origin_op = Some(t.data_origin_id);
                    }
                    input
                })
                .collect())
        } else {
            // is not an op, is a tensor, return err
            Err(Error::from(ErrorKind::OpNotFound))
        }
    }

    pub fn get_inputs(&self, context: &Scope) -> Result<Vec<Tensor>> {
        let ops = &*context.ops.borrow();
        let reg = &*context.registry.borrow();
        if let Some(op) = ops.get(self) {
            // first check if it's an operation
            let mut inputs = Vec::with_capacity(op.num_inputs());
            for (source_op, idx) in (0..op.num_inputs()).into_iter().map(|i| op.input(i)) {
                let source_op_name = source_op.name().unwrap();
                let input_op = ops.iter()
                    .find(|&(_, op)| source_op_name == op.name().unwrap())
                    .map(|(id, _)| id)
                    .ok_or(Error::from(ErrorKind::OpNotFound))?;
                inputs.extend(
                    reg.iter()
                        .filter(|&(_, t)| &t.data_origin_id == input_op)
                        .map(|(id, t)| {
                            let mut input = Tensor {
                                ident: *id,
                                dtype: t.dtype,
                                idtype: t.idtype.clone(),
                                initializer: None,
                                origin_op: None,
                                idx: t.data_origin.1,
                            };
                            if t.idtype.is_initialized() {
                                input.initializer = Some(t.data_origin_id);
                            } else {
                                input.origin_op = Some(t.data_origin_id);
                            }
                            input
                        }),
                );
            }
            Ok(inputs)
        } else {
            // is not an op, is a tensor, return err
            Err(Error::from(ErrorKind::OpNotFound))
        }
    }

    pub fn get_name(&self, context: &Scope) -> Result<String> {
        let ops = &*context.ops.borrow();
        let reg = &*context.registry.borrow();
        if let Some(source_op) = ops.get(self) {
            Ok(source_op.name()?)
        } else {
            let tensor = reg.get(self)
                .ok_or(Error::from(Error::from(ErrorKind::TensorNotFound)))?;
            Ok(tensor.full_name.to_str().unwrap().to_owned())
        }
    }
}

impl GetOp for NodeIdent {
    fn get_op(&self) -> &NodeIdent {
        self
    }

    fn source_index(&self) -> Option<i32> {
        None
    }
}

impl<'a> GetOp for &'a NodeIdent {
    fn get_op(&self) -> &NodeIdent {
        *self
    }

    fn source_index(&self) -> Option<i32> {
        None
    }
}

/// Get the identity token of an tensor or an operation.
pub trait GetOp {
    fn get_op(&self) -> &NodeIdent;
    fn source_index(&self) -> Option<i32>;
}

#[derive(Debug, Clone)]
pub(crate) struct TensorData {
    /// fully qualified name, including the scope path
    pub full_name: PathBuf,
    pub dtype: DataType,
    pub idtype: IdType,
    /// operation & operation's output index
    pub data_origin: (OperationData, i32),
    pub data_origin_id: NodeIdent,
    pub shape: Shape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum IdType {
    Constant,
    Variable,
    Operation(&'static str),
    Placeholder,
}

impl IdType {
    fn is_initialized(&self) -> bool {
        match *self {
            IdType::Constant | IdType::Variable => true,
            _ => false,
        }
    }
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

macro_rules! impl_identity_traits {
    ($type:ty) => {
        impl PartialEq for $type {
            fn eq(&self, other: &Self) -> bool {
                self.ident == other.ident
            }
        }

        impl Eq for $type {}

        impl Hash for $type {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.ident.hash(state);
            }
        }

        impl Into<NodeIdent> for $type {
            fn into(self) -> NodeIdent {
                self.ident
            }
        }

        impl<'a> Into<NodeIdent> for &'a $type {
            fn into(self) -> NodeIdent {
                self.ident
            }
        }
    };
}

/// Tensor
#[derive(Debug, Clone, Copy)]
pub struct Tensor {
    pub(crate) ident: NodeIdent,
    pub(crate) idtype: IdType,
    pub(crate) dtype: DataType,
    /// Index of this tensor in the source operation output.
    pub(crate) idx: i32,
    pub(crate) initializer: Option<NodeIdent>,
    pub(crate) origin_op: Option<NodeIdent>,
}

impl Tensor {
    pub fn get_initializer(&self, context: &Scope) -> Result<Tensor> {
        if let Some(ref initializer) = self.initializer {
            let registry = &*context.registry.borrow();
            let init_data = &registry[&initializer];
            Ok(Tensor {
                ident: initializer.clone(),
                idtype: IdType::Constant,
                dtype: init_data.dtype,
                idx: init_data.data_origin.1,
                initializer: None,
                origin_op: Some(init_data.data_origin_id),
            })
        } else {
            Err(Error::from(ErrorKind::Stub))
        }
    }

    pub fn get_name(&self, context: &Scope) -> String {
        let registry = &*context.registry.borrow();
        registry[&self.ident].full_name.to_str().unwrap().to_owned()
    }

    pub fn get_shape(&self, context: &Scope) -> Shape {
        let registry = &*context.registry.borrow();
        registry[&self.ident].shape.clone()
    }

    pub fn set_shape<Sh>(self, context: &mut Scope, shape: Sh) -> Result<Tensor>
    where
        Sh: ShapeOps,
    {
        let current_shape = self.get_shape(context);
        let new_shape = current_shape
            .merge_with(&shape.to_shape())?
            .definition_i64()
            .ok_or(Error::from(ErrorKind::UndefinedTensorShape))?;
        ::ops::array_ops::reshape(context, self, new_shape.as_slice(), "")
    }

    pub fn set_shape_from_tensor<Sh>(self, context: &mut Scope, shape: Sh) -> Result<Tensor>
    where
        Sh: TensorOps,
    {
        let shape = shape.into_tensor(context);
        ::ops::array_ops::reshape(context, self, shape, "")
    }

    pub fn is_ref(&self) -> bool {
        if self.idtype == IdType::Variable {
            true
        } else {
            false
        }
    }

    pub fn op_type(&self, context: &Scope) -> &str {
        match self.idtype {
            IdType::Constant => "Constant",
            IdType::Variable => "Variable",
            IdType::Operation(op) => op,
            IdType::Placeholder => "Placeholder",
        }
    }

    pub fn consumers(&self, context: &Scope) -> Result<Vec<NodeIdent>> {
        let op = self.get_op();
        let ops = &*context.ops.borrow();
        let op = &ops[&op];
        let mut consumers = vec![];
        for (ref op_name, idx) in op.output_consumers(self.idx as usize)
            .into_iter()
            .map(|(op, idx)| (op.name().unwrap(), idx))
        {
            consumers.push(ops.iter()
                .find(|&(k, v)| &v.name().unwrap() == op_name)
                .map(|(k, v)| *k)
                .ok_or(Error::from(ErrorKind::Stub))?);
        }
        Ok(consumers)
    }
}

impl GetOp for Tensor {
    fn get_op(&self) -> &NodeIdent {
        if let Some(ref op) = self.initializer {
            op
        } else {
            self.origin_op.as_ref().unwrap()
        }
    }

    fn source_index(&self) -> Option<i32> {
        Some(self.idx)
    }
}

impl<'a> GetOp for &'a Tensor {
    fn get_op(&self) -> &NodeIdent {
        if let Some(ref op) = self.initializer {
            op
        } else {
            self.origin_op.as_ref().unwrap()
        }
    }

    fn source_index(&self) -> Option<i32> {
        Some(self.idx)
    }
}

impl_identity_traits!(Tensor);

/// Constant
#[derive(Debug, Clone, Copy)]
pub struct Constant {
    ident: NodeIdent,
    origin_op: NodeIdent,
    dtype: DataType,
}

impl Constant {
    pub fn new<Sh, T>(context: &mut Scope, value: &[T], shape: Sh) -> Constant
    where
        T: tf::TensorType,
        Sh: ShapeOps,
    {
        let name = context
            .resolve_tensor_name(None, IdType::Constant, false)
            .unwrap();
        context.constant(value, shape, name).unwrap()
    }

    pub fn get_name(&self, context: &Scope) -> String {
        let registry = &*context.registry.borrow();
        registry[&self.ident].full_name.to_str().unwrap().to_owned()
    }

    pub fn get_shape(&self, shape: &Scope) -> Shape {
        let registry = &*shape.registry.borrow();
        registry[&self.ident].shape.clone()
    }
}

/// Treat a constant as an initilized variable. Useful for operating with the context manager.
impl Into<Tensor> for Constant {
    fn into(self) -> Tensor {
        let Constant {
            ident,
            dtype,
            origin_op,
        } = self;
        Tensor {
            ident,
            dtype,
            idtype: IdType::Constant,
            idx: 0,
            initializer: None,
            origin_op: Some(origin_op),
        }
    }
}

impl GetOp for Constant {
    fn get_op(&self) -> &NodeIdent {
        &self.origin_op
    }

    fn source_index(&self) -> Option<i32> {
        Some(0)
    }
}

impl<'a> GetOp for &'a Constant {
    fn get_op(&self) -> &NodeIdent {
        &self.origin_op
    }

    fn source_index(&self) -> Option<i32> {
        Some(0)
    }
}

impl_identity_traits!(Constant);

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
    where
        T: tf::TensorType,
        TeS: ShapeSize,
    {
        let values = context.constant(initial_value, shape, "").unwrap();
        context
            .get_variable_with_initializer(values, false, "")
            .unwrap()
    }

    pub fn get_name(&self, context: &Scope) -> String {
        let registry = &*context.registry.borrow();
        registry[&self.ident].full_name.to_str().unwrap().to_owned()
    }

    pub fn get_shape(&self, scope: &Scope) -> Shape {
        let registry = &*scope.registry.borrow();
        registry[&self.ident].shape.clone()
    }

    /// Returns a Variable type if the Tensor is indeed a Variable, error otherwise.
    pub fn from_tensor(scope: &Scope, tensor: &Tensor) -> Result<Variable> {
        let registry = &*scope.registry.borrow();
        let tensor_data = &registry[&tensor.ident];
        if let Some(initializer) = tensor.initializer {
            Ok(Variable {
                ident: tensor.ident,
                dtype: tensor.dtype,
                initializer,
                idx: tensor.idx,
            })
        } else {
            Err(Error::from(ErrorKind::Stub))
        }
    }

    pub fn op_type(&self, context: &Scope) -> &str {
        "Variable"
    }
}

impl Into<Tensor> for Variable {
    fn into(self) -> Tensor {
        let Variable {
            ident,
            dtype,
            idx,
            initializer,
            ..
        } = self;
        Tensor {
            ident,
            dtype,
            idtype: IdType::Variable,
            idx,
            initializer: Some(initializer),
            origin_op: None,
        }
    }
}

impl GetOp for Variable {
    fn get_op(&self) -> &NodeIdent {
        &self.initializer
    }

    fn source_index(&self) -> Option<i32> {
        Some(self.idx)
    }
}

impl<'a> GetOp for &'a Variable {
    fn get_op(&self) -> &NodeIdent {
        &self.initializer
    }

    fn source_index(&self) -> Option<i32> {
        Some(self.idx)
    }
}

impl_identity_traits!(Variable);

#[doc(hidden)]
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

    pub fn gather(&self, scope: &Scope, indices: Tensor) -> Tensor {
        unimplemented!()
    }
}

/// An enumeration of the the different types of tensors.
#[derive(Debug)]
pub enum TensorContent {
    Bool(TypedTensor<bool>),
    Float(TypedTensor<f32>),
    Double(TypedTensor<f64>),
    UInt8(TypedTensor<u8>),
    Int8(TypedTensor<i8>),
    Int16(TypedTensor<i16>),
    Int32(TypedTensor<i32>),
    Int64(TypedTensor<i64>),
    String(TypedTensor<String>),
    QUInt8(TypedTensor<::QUInt8>),
    QUInt16(TypedTensor<::QUInt16>),
    QInt16(TypedTensor<::QInt16>),
    QInt32(TypedTensor<::QInt32>),
    BFloat16(TypedTensor<::BFloat16>),
    Complex64(TypedTensor<::Complex32>),
    Complex128(TypedTensor<::Complex64>),
}

impl Clone for TensorContent {
    fn clone(&self) -> TensorContent {
        match *self {
            TensorContent::Bool(ref val) => TensorContent::Bool(clone_tensor!(val)),
            TensorContent::Float(ref val) => TensorContent::Float(clone_tensor!(val)),
            TensorContent::Double(ref val) => TensorContent::Double(clone_tensor!(val)),
            TensorContent::UInt8(ref val) => TensorContent::UInt8(clone_tensor!(val)),
            TensorContent::Int8(ref val) => TensorContent::Int8(clone_tensor!(val)),
            TensorContent::Int16(ref val) => TensorContent::Int16(clone_tensor!(val)),
            TensorContent::Int32(ref val) => TensorContent::Int32(clone_tensor!(val)),
            TensorContent::Int64(ref val) => TensorContent::Int64(clone_tensor!(val)),
            TensorContent::String(ref val) => TensorContent::String(clone_tensor!(val)),
            TensorContent::QUInt8(ref val) => TensorContent::QUInt8(clone_tensor!(val)),
            TensorContent::QUInt16(ref val) => TensorContent::QUInt16(clone_tensor!(val)),
            TensorContent::QInt16(ref val) => TensorContent::QInt16(clone_tensor!(val)),
            TensorContent::QInt32(ref val) => TensorContent::QInt32(clone_tensor!(val)),
            TensorContent::BFloat16(ref val) => TensorContent::BFloat16(clone_tensor!(val)),
            TensorContent::Complex64(ref val) => TensorContent::Complex64(clone_tensor!(val)),
            TensorContent::Complex128(ref val) => TensorContent::Complex128(clone_tensor!(val)),
        }
    }
}

macro_rules! unwrap_tensor_content {
    ($variant:ident, $name:tt, $type:ty) => {
        pub fn $name(self) -> TypedTensor<$type> {
            match self {
                TensorContent::$variant(val) => val,
                _ => unreachable!()
            }
        }
    }
}

macro_rules! from_tensor_to_content {
    ($type:ty, $id:ident) => {
        impl From<TypedTensor<$type>> for TensorContent {
            fn from(tensor: TypedTensor<$type>) -> Self {
                TensorContent::$id(tensor)
            }
        }
    }
}

from_tensor_to_content!(f32, Float);
from_tensor_to_content!(f64, Double);
from_tensor_to_content!(i32, Int32);
from_tensor_to_content!(u8, UInt8);
from_tensor_to_content!(i16, Int16);
from_tensor_to_content!(i8, Int8);
from_tensor_to_content!(i64, Int64);
from_tensor_to_content!(bool, Bool);
from_tensor_to_content!(String, String);
from_tensor_to_content!(::QUInt8, QUInt8);
from_tensor_to_content!(::QUInt16, QUInt16);
from_tensor_to_content!(::QInt16, QInt16);
from_tensor_to_content!(::QInt32, QInt32);
from_tensor_to_content!(::BFloat16, BFloat16);
from_tensor_to_content!(::Complex32, Complex64);
from_tensor_to_content!(::Complex64, Complex128);

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
            TensorContent::QUInt8(_) => DataType::QUInt8,
            TensorContent::QUInt16(_) => DataType::QUInt16,
            TensorContent::QInt16(_) => DataType::QInt16,
            TensorContent::QInt32(_) => DataType::QInt32,
            TensorContent::BFloat16(_) => DataType::BFloat16,
            TensorContent::Complex64(_) => DataType::Complex64,
            TensorContent::Complex128(_) => DataType::Complex128,
        }
    }

    fn set_tensor_list_attr(
        new_op: &mut OperationDescription,
        name: &str,
        val: &[TensorContent],
    ) -> Result<()> {
        match val[0].get_datatype() {
            DataType::Bool => new_op.set_attr_tensor_list(name, collect_bool_tensor(val))?,
            DataType::Double => new_op.set_attr_tensor_list(name, collect_double_tensor(val))?,
            DataType::Float => new_op.set_attr_tensor_list(name, collect_float_tensor(val))?,
            DataType::Int32 => new_op.set_attr_tensor_list(name, collect_i32_tensor(val))?,
            DataType::UInt8 => new_op.set_attr_tensor_list(name, collect_u8_tensor(val))?,
            DataType::Int16 => new_op.set_attr_tensor_list(name, collect_i16_tensor(val))?,
            DataType::Int8 => new_op.set_attr_tensor_list(name, collect_i8_tensor(val))?,
            DataType::Int64 => new_op.set_attr_tensor_list(name, collect_i64_tensor(val))?,
            DataType::String => new_op.set_attr_tensor_list(name, collect_string_tensor(val))?,
            DataType::QUInt8 => new_op.set_attr_tensor_list(name, collect_quint8_tensor(val))?,
            DataType::QUInt16 => new_op.set_attr_tensor_list(name, collect_quint16_tensor(val))?,
            DataType::QInt16 => new_op.set_attr_tensor_list(name, collect_qint16_tensor(val))?,
            DataType::QInt32 => new_op.set_attr_tensor_list(name, collect_qint32_tensor(val))?,
            DataType::BFloat16 => new_op.set_attr_tensor_list(name, collect_bfloat16_tensor(val))?,
            DataType::Complex64 => {
                new_op.set_attr_tensor_list(name, collect_complex64_tensor(val))?
            }
            DataType::Complex128 => {
                new_op.set_attr_tensor_list(name, collect_complex128_tensor(val))?
            }
            _ => return Err(Error::from(ErrorKind::Stub)),
        }
        Ok(())
    }

    fn set_tensor_attr(
        new_op: &mut OperationDescription,
        name: &str,
        val: &[TensorContent],
    ) -> Result<()> {
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
            DataType::String => {
                new_op.set_attr_tensor(name, collect_string_tensor(val).pop().unwrap())?
            }
            DataType::QUInt8 => {
                new_op.set_attr_tensor(name, collect_quint8_tensor(val).pop().unwrap())?
            }
            DataType::QUInt16 => {
                new_op.set_attr_tensor(name, collect_quint16_tensor(val).pop().unwrap())?
            }
            DataType::QInt16 => {
                new_op.set_attr_tensor(name, collect_qint16_tensor(val).pop().unwrap())?
            }
            DataType::QInt32 => {
                new_op.set_attr_tensor(name, collect_qint32_tensor(val).pop().unwrap())?
            }
            DataType::BFloat16 => {
                new_op.set_attr_tensor(name, collect_bfloat16_tensor(val).pop().unwrap())?
            }
            DataType::Complex64 => {
                new_op.set_attr_tensor(name, collect_complex64_tensor(val).pop().unwrap())?
            }
            DataType::Complex128 => {
                new_op.set_attr_tensor(name, collect_complex128_tensor(val).pop().unwrap())?
            }
            _ => return Err(Error::from(ErrorKind::Stub)),
        }
        Ok(())
    }

    unwrap_tensor_content!(Float, unwrap_float, f32);
    unwrap_tensor_content!(Double, unwrap_double, f64);
    unwrap_tensor_content!(Int32, unwrap_i32, i32);
    unwrap_tensor_content!(UInt8, unwrap_u8, u8);
    unwrap_tensor_content!(Int16, unwrap_i16, i16);
    unwrap_tensor_content!(Int8, unwrap_i8, i8);
    unwrap_tensor_content!(Int64, unwrap_i64, i64);
    unwrap_tensor_content!(Bool, unwrap_bool, bool);
    unwrap_tensor_content!(String, unwrap_string, String);
    unwrap_tensor_content!(QUInt8, unwrap_quint8, ::QUInt8);
    unwrap_tensor_content!(QUInt16, unwrap_quint16, ::QUInt16);
    unwrap_tensor_content!(QInt16, unwrap_qint16, ::QInt16);
    unwrap_tensor_content!(QInt32, unwrap_qint32, ::QInt32);
    unwrap_tensor_content!(BFloat16, unwrap_bfloat16, ::BFloat16);
    unwrap_tensor_content!(Complex64, unwrap_complex64, ::Complex32);
    unwrap_tensor_content!(Complex128, unwrap_complex128, ::Complex64);
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
                    _ => unreachable!()
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
collect_tensors!(String, collect_string_tensor, String);
collect_tensors!(QUInt8, collect_quint8_tensor, ::QUInt8);
collect_tensors!(QUInt16, collect_quint16_tensor, ::QUInt16);
collect_tensors!(QInt16, collect_qint16_tensor, ::QInt16);
collect_tensors!(QInt32, collect_qint32_tensor, ::QInt32);
collect_tensors!(BFloat16, collect_bfloat16_tensor, ::BFloat16);
collect_tensors!(Complex64, collect_complex64_tensor, ::Complex32);
collect_tensors!(Complex128, collect_complex128_tensor, ::Complex64);

impl<'a, T> From<Vec<TypedTensor<T>>> for Attribute<'a>
where
    T: tf::TensorType,
    TensorContent: From<TypedTensor<T>>,
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

pub(crate) struct Function;

pub(crate) type GradFunc = Box<FnMut(&mut Scope, &[Option<Tensor>]) -> Result<Vec<Option<Tensor>>>>;

impl Function {
    pub fn get_gradient_func(&self) -> Option<GradFunc> {
        unimplemented!()
    }

    pub fn get_attr(&self, name: &str) -> Option<FuncAttr> {
        unimplemented!()
    }
}

#[allow(dead_code)]
pub(crate) enum FuncAttr {
    S(String),
    B(bool),
}

impl FuncAttr {
    pub fn unwrap_b(self) -> bool {
        match self {
            FuncAttr::B(b) => b,
            _ => unreachable!(),
        }
    }

    pub fn unwrap_s(self) -> String {
        match self {
            FuncAttr::S(s) => s,
            _ => unreachable!(),
        }
    }
}

#![allow(unused_variables)]

extern crate tensorflow as tf;
extern crate num_complex;
extern crate uuid;

#[macro_use]
mod macros;
#[macro_use]
pub mod ops;
pub mod train;
pub mod client;
pub(crate) mod framework;

use tf::{OperationDescription, Output, Shape};
pub type OperationData = tf::Operation;
pub type TypedTensor<T> = tf::Tensor<T>;
use tf::{DataType, Graph, Session, SessionOptions, Status, StepWithGraph};
use tf::{QUInt8, QInt16, QUInt16, QInt32, BFloat16};
use num_complex::{Complex32, Complex64};

#[derive(Debug)]
pub enum Error {
    /// TensorFlow API error
    TFError(Status),
    /// ffi::NulError
    NulError,
    Stub,
    Msg(String),
}

impl std::convert::From<Status> for Error {
    fn from(err: Status) -> Self {
        Error::TFError(err)
    }
}

impl std::convert::From<std::ffi::NulError> for Error {
    fn from(_err: std::ffi::NulError) -> Self {
        Error::NulError
    }
}

pub mod prelude {
    pub use super::framework::{Attribute, Constant, DefinedShape, NodeIdent, Operation, Scope,
                               Tensor, TensorArray, TensorContent, Variable};
    pub use super::client::ClientSession;
    pub use super::{OperationData, TypedTensor};
    pub use super::Error as TFError;
    pub use tf::{DataType, Status};

    pub use super::train;
    pub use super::ops;
}

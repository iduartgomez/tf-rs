extern crate tensorflow as tf;
extern crate uuid;

#[macro_use]
mod macros;
pub(crate) mod framework;
pub mod ops;
pub mod client;

use tf::{OperationDescription, Output, Shape};
pub type OperationData = tf::Operation;
pub type TypedTensor<T> = tf::Tensor<T>;
pub use tf::{Graph, DataType, Session, SessionOptions, StepWithGraph, Status, version};

#[derive(Debug)]
pub enum Error {
    /// TensorFlow API error
    TFError(Status),
    /// ffi::NulError
    NulError,
    Stub,
    StubMsg(String),
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
    pub use super::framework::{Constant, Ident, Scope, Operation, Tensor, TensorContent,
                               TensorArray, Variable};
    pub use super::{Error, OperationData, Status, TypedTensor};
}

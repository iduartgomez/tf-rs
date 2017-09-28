#![allow(unused_variables)]

extern crate tensorflow as tf;
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
    pub use super::framework::{Constant, NodeIdent, Operation, Scope, Tensor, TensorArray,
                               TensorContent, Variable, Attribute};
    pub use super::{OperationData, TypedTensor};
    pub use super::Error as TFError;
    pub use tf::{DataType, Status};

    pub use super::train;
    pub use super::ops;
}

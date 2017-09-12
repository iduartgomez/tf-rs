extern crate tensorflow as tf;
extern crate uuid;

#[macro_use]
mod macros;
pub(crate) mod framework;
pub mod ops;
pub mod client;

use tf::OperationDescription;
use tf::Output;
use tf::Shape;
pub type OperationData = tf::Operation;
pub type TypedTensor<T> = tf::Tensor<T>;
pub use tf::Graph;
pub use tf::DataType;
pub use tf::Session;
pub use tf::SessionOptions;
pub use tf::StepWithGraph;
pub use tf::Status;

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

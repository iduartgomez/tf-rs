//! A high level (Pythonic) API for the TensorFlow framework.

#![allow(unused_variables)]

extern crate tensorflow as tf;
extern crate num_complex;
extern crate uuid;
#[macro_use]
extern crate error_chain;

#[macro_use]
mod macros;
mod errors;
#[macro_use]
pub mod ops;
pub mod train;
pub mod client;
pub(crate) mod framework;

use tf::{OperationDescription, Output, Shape};
pub type OperationData = tf::Operation;
pub type TypedTensor<T> = tf::Tensor<T>;
use tf::{DataType, Graph, Session, SessionOptions, StepWithGraph};
use tf::{QUInt8, QInt16, QUInt16, QInt32, BFloat16};
use num_complex::{Complex32, Complex64};

pub mod prelude {
    pub use super::framework::{Attribute, Constant, DefinedShape, NodeIdent, Operation, Scope,
                               Tensor, TensorArray, TensorContent, Variable};
    pub use super::client::ClientSession;
    pub use super::{OperationData, TypedTensor};
    pub use super::errors::Error as TFError;
    pub use tf::{DataType, Status};

    pub use super::train;
    pub use super::ops;
}

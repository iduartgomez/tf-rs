//! Support for training models.

use super::*;
use super::framework::*;
use errors::*;

mod moving_averages;
mod slot_creator;
pub mod nn;
pub mod optimizers;

pub use self::moving_averages::*;
pub use self::slot_creator::*;

fn validate_convnet_data_dormat(data_format: &str) -> Result<&'static str> {
    match data_format {
        "NHWC" => Ok("NHWC"),
        "NCHW" => Ok("NCHW"),
        _ => Err(Error::from(ErrorKind::Stub)),
    }
}

fn _validate_convnet_3d_data_dormat(data_format: &str) -> Result<&'static str> {
    match data_format {
        "NDHWC" => Ok("NDHWC"),
        "NCDHW" => Ok("NCDHW"),
        _ => Err(Error::from(ErrorKind::Stub)),
    }
}

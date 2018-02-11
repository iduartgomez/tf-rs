//! Error module.

use tf::Status;

error_chain! {
    errors { 
        Stub 
        UnimplementedTraitMethod 
    }

    foreign_links {
        FFINulError(::std::ffi::NulError);
    }
}

impl From<Status> for Error {
    fn from(err: Status) -> Error {
        Error::from(format!("{}", err))
    }
}

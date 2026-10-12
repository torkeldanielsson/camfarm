//! Tracy (feature `tracy`) and NVTX (feature `nvtx`) zones; without the features the macros expand to nothing.

/// Profile the rest of the enclosing block as a named zone.
macro_rules! zone {
    ($name:literal) => {
        #[cfg(feature = "tracy")]
        let _tracy_zone = tracy_client::span!($name);
        #[cfg(feature = "nvtx")]
        let _nvtx_zone = nvtx::LocalRange::new({
            const NAME: &std::ffi::CStr = match std::ffi::CStr::from_bytes_with_nul(concat!($name, "\0").as_bytes()) {
                Ok(name) => name,
                Err(_) => panic!("zone name"),
            };
            NAME
        });
    };
}

macro_rules! plot {
    ($name:literal, $value:expr) => {
        #[cfg(feature = "tracy")]
        tracy_client::plot!($name, $value as f64);
        #[cfg(not(feature = "tracy"))]
        let _ = || $value;
    };
}

pub(crate) use {plot, zone};

pub fn start() {
    #[cfg(feature = "tracy")]
    tracy_client::Client::start();
}

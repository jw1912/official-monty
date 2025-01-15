pub mod policy;
pub mod value;

#[repr(C)]
struct Nets(value::ValueNetwork, policy::PolicyNetwork);

static NETS: Nets = unsafe { std::mem::transmute(*include_bytes!("../ataxx.network")) };

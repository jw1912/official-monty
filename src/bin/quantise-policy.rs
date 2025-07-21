use std::io::Write;

use monty::{
    networks::policy::{PolicyNetwork, UnqPolicyNetwork},
    read_into_struct_unchecked, MappedWeights,
};

fn main() {
    let unquantised: MappedWeights<UnqPolicyNetwork> =
        unsafe { read_into_struct_unchecked("sb30-attn.network") };

    let quantised = unquantised.data.quantise();

    let mut file = std::fs::File::create("attnq.network").unwrap();

    unsafe {
        let ptr: *const PolicyNetwork = quantised.as_ref();
        let slice_ptr: *const u8 = std::mem::transmute(ptr);
        let slice = std::slice::from_raw_parts(slice_ptr, std::mem::size_of::<PolicyNetwork>());
        file.write_all(slice).unwrap();
    }
}

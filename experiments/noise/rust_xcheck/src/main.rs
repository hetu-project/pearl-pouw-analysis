//! Cross-check: runs Pearl's own Rust reference (zk-pow @156ef9a8) on a fixed, deterministic
//! input and prints intermediate values, so pearl_ref.py can be compared byte-for-byte.
use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PublicProofParams, SeedDerivation};
use zk_pow::api::proof_utils::{CompiledPublicParams, compute_jackpot_hash};
use zk_pow::circuit::chip::compute_jackpot;
use zk_pow::circuit::pearl_noise::{compute_noise, generate_permutation_matrix, generate_uniform_random_matrix};

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }

fn main() {
    let header = IncompleteBlockHeader { version: 0, prev_block: [1; 32], merkle_root: [2; 32], timestamp: 0x66666666, nbits: 0x207FFFFF };
    let cfg = MiningConfiguration {
        common_dim: 2048, rank: 128, mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
        moe: None,
    };
    for (name, sd) in [("legacy", SeedDerivation::Legacy), ("salted", SeedDerivation::Salted)] {
        let p = PublicProofParams::new_dummy(header, sd, cfg, 256, 384, 16, 4);
        let c = CompiledPublicParams::from(&p);
        println!("[{name}] header_bytes={}", hex(&header.to_bytes()));
        println!("[{name}] config_bytes={}", hex(&cfg.to_bytes()));
        println!("[{name}] job_key={}", hex(&c.job_key));
        println!("[{name}] b_noise_seed={}", hex(&c.commitment_hash.0));
        println!("[{name}] a_noise_seed={}", hex(&c.commitment_hash.1));
        println!("[{name}] a_rows={:?} b_cols={:?}", c.a_rows_indices, c.b_cols_indices);
        let noise = compute_noise(&c);
        let sa: Vec<Vec<i8>> = (0..c.h).map(|u| (0..c.k).map(|l| (((u * 131 + l * 17) % 128) as i32 - 64) as i8).collect()).collect();
        let sb: Vec<Vec<i8>> = (0..c.w).map(|v| (0..c.k).map(|l| (((v * 37 + l * 29 + 5) % 128) as i32 - 64) as i8).collect()).collect();
        let jp = compute_jackpot(&c, &sa, &sb, &noise);
        let jh = compute_jackpot_hash(&jp, c.commitment_hash.1);
        println!("[{name}] noise_a_row0_first16={:?}", &noise.a[0][..16]);
        println!("[{name}] noise_b_row0_first16={:?}", &noise.b[0][..16]);
        let sum_a: i64 = noise.a.iter().flatten().map(|&x| x as i64).sum();
        let sum_b: i64 = noise.b.iter().flatten().map(|&x| x as i64).sum();
        println!("[{name}] noise_a_sum={sum_a} noise_b_sum={sum_b}");
        println!("[{name}] jackpot_msg={:?}", jp);
        println!("[{name}] jackpot_hash={}", hex(&jh));
        let eal = generate_uniform_random_matrix(&{let mut s=[0u8;32]; s[..8].copy_from_slice(b"A_tensor"); s}, &c.commitment_hash.1, &[0], 128);
        let perm = generate_permutation_matrix(&{let mut s=[0u8;32]; s[..8].copy_from_slice(b"A_tensor"); s}, &c.commitment_hash.1, 8, 128);
        println!("[{name}] e_al_row0_first8={:?} perm_first8={:?}", &eal[0][..8], perm);
    }
}

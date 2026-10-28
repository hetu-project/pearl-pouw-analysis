// Research experiment (added only in the scratch copy, not in the original repo).
// Usage: research_plain_vs_zk <node|gpu|gpu4096> [salted]
use std::time::Instant;

use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, SeedDerivation, ZKProof};
use zk_pow::api::proof_utils::CompiledPublicParams;
use zk_pow::api::{prove, verify};
use zk_pow::circuit::circuit_utils::CircuitCache;
use zk_pow::ffi::mine::mine;
use zk_pow::ffi::plain_proof::PlainProof;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).map(|s| s.as_str()).unwrap_or("node");
    let seed = if args.iter().any(|a| a == "salted") { SeedDerivation::Salted } else { SeedDerivation::Legacy };

    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1; 32],
        merkle_root: [2; 32],
        timestamp: 0x6666_6666,
        nbits: 0x1E01FFFF, // node DefaultNBits
    };
    let gpu_cols: Vec<u32> = (0..32u32).flat_map(|i| [8 * i, 8 * i + 1]).collect();
    let (m, n, k, config) = match which {
        // node/zkpow default: r=128, k=16r=2048, rows [0,8,64,72], cols [0,1,8,9,32,33,40,41], m=256, n=512
        "node" => (256usize, 512usize, 2048usize, MiningConfiguration {
            common_dim: 2048, rank: 128, mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
            moe: None,
        }),
        // GPU miner settings.py default patterns: rows [0,8] (h=2), cols 64 entries (w=64), r=128, k=4096
        "gpu4096" => (256usize, 256usize, 4096usize, MiningConfiguration {
            common_dim: 4096, rank: 128, mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&gpu_cols).unwrap(),
            moe: None,
        }),
        _ => panic!("unknown config"),
    };

    let t = Instant::now();
    let pp: PlainProof = mine(m, n, k, header, config, None, false, seed).unwrap();
    let mine_t = t.elapsed();
    let bytes = bincode::serialize(&pp).unwrap();
    println!("[{which}] m={m} n={n} k={k} r={} h={} w={}", config.rank, config.rows_pattern.size(), config.cols_pattern.size());
    println!("[{which}] mine (CPU reference miner, incl. random matrix gen): {:?}", mine_t);
    println!(
        "[{which}] PlainProof bincode bytes = {} | A: leaves={} siblings={} total_leaves={} | B: leaves={} siblings={} total_leaves={}",
        bytes.len(),
        pp.a.proof.leaf_data.len(), pp.a.proof.siblings.len(), pp.a.proof.total_leaves,
        pp.bt.proof.leaf_data.len(), pp.bt.proof.siblings.len(), pp.bt.proof.total_leaves
    );

    // plain verification (pool share path)
    let reps = 20;
    let t = Instant::now();
    for _ in 0..reps {
        verify::verify_plain_proof(&header, &pp, None, seed).unwrap();
    }
    println!("[{which}] verify_plain_proof avg over {reps}: {:?}", t.elapsed() / reps);

    // ZK prove
    let mut cache = CircuitCache::default();
    let t = Instant::now();
    let res = prove::zk_prove_plain_proof(header, &pp, &mut cache, true, seed).unwrap();
    println!("[{which}] zk_prove_plain_proof #1 (incl. circuit build): {:?}", t.elapsed());
    let t = Instant::now();
    let res2 = prove::zk_prove_plain_proof(header, &pp, &mut cache, true, seed).unwrap();
    println!("[{which}] zk_prove_plain_proof #2 (cached circuits): {:?}", t.elapsed());
    println!(
        "[{which}] public_data bytes = {}, proof_data bytes = {} (preamble 22 + plonky2 {}), 2nd proof_data bytes = {}",
        res.public_data.len(), res.proof_data.len(), res.proof_data.len() - 22, res2.proof_data.len()
    );
    println!("[{which}] proof_data identical across two proves: {}", res.proof_data == res2.proof_data);
    println!("[{which}] public_data identical across two proves: {}", res.public_data == res2.public_data);

    let (pub_params, zk) = ZKProof::deserialize(header, seed, &res.public_data, &res.proof_data).unwrap();
    let compiled = CompiledPublicParams::from(&pub_params);
    println!("[{which}] stark degree_bits = {} (expected_num_rows = {}), pow_bits={:?} rate_bits={:?}",
        compiled.degree_bits(), compiled.expected_num_rows(), zk.pow_bits, zk.rate_bits);

    // ZK verify with a fresh verifier-only cache (includes compiling verifier circuits)
    let mut vcache = CircuitCache::default();
    let t = Instant::now();
    verify::verify_block(&pub_params, &zk, &mut vcache).unwrap();
    println!("[{which}] verify_block fresh cache (compile verifier circuits + verify): {:?}", t.elapsed());
    let reps = 5;
    let t = Instant::now();
    for _ in 0..reps {
        verify::verify_block_cached_circuits_only(&pub_params, &zk, &vcache, None).unwrap();
    }
    println!("[{which}] verify_block_cached_circuits_only avg over {reps}: {:?}", t.elapsed() / reps);

    // Write artifacts
    let dir = std::path::Path::new("../../artifacts");
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join(format!("{which}_plain_proof.bincode")), &bytes).unwrap();
    std::fs::write(dir.join(format!("{which}_public_data.bin")), &res.public_data).unwrap();
    std::fs::write(dir.join(format!("{which}_proof_data.bin")), &res.proof_data).unwrap();
}

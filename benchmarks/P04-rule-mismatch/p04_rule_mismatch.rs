// P04 benchmark, part 1: miner, pool and consensus apply different rules to the same proof.
//
// Part A  The vLLM plugin mines whenever k >= 1024 (miner/vllm-miner/src/vllm_miner/config.yaml),
//         with r = 128 and the GPU hash-tile pattern (rows [0, 8], 64 fixed columns). The verifier
//         requires k >= 16r = 2048 (zk-pow/src/api/sanity_checks.rs). A proof mined with exactly
//         the plugin's parameters at k = 1024 (and k = 2048 as control) is pushed through every
//         stage the official pipeline runs: miner pre-check, pool share check, gateway ZK proving,
//         node consensus verification.
//
// Part B  Pool share verification (verify_plain_proof, the official validate_share example) uses
//         the unpenalized jackpot bound; the rank penalty (bound x 128/r) is applied only by the
//         Go node, after the fork height (node/blockchain/validate.go). Proofs mined at r = 256
//         are checked against both: every one is a valid share, about half fail the node's rule.
//
// Usage: cargo run --release --example p04_rule_mismatch [zk]
//        Without "zk" only the checks that do not need a ZK proof are run.
use std::time::Instant;

use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, SeedDerivation, ZKProof};
use zk_pow::api::proof_utils::nbits_to_difficulty;
use zk_pow::api::sanity_checks::{check_rank_penalty, extract_difficulty_bound, penalized_target_bound};
use zk_pow::api::{prove, verify};
use zk_pow::circuit::circuit_utils::CircuitCache;
use zk_pow::ffi::mine::mine;
use zk_pow::ffi::plain_proof::check_cert_version_eligible;

const RANK_ATTEMPTS: usize = 6;

fn gpu_config(k: u32, rank: u16) -> MiningConfiguration {
    // MinerSettings defaults (miner/miner-base/src/miner_base/settings.py): rows [0, 8], 64 columns
    let cols: Vec<u32> = (0..32u32).flat_map(|i| [8 * i, 8 * i + 1]).collect();
    MiningConfiguration {
        common_dim: k,
        rank,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 8]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&cols).unwrap(),
        moe: None,
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let zk = std::env::args().any(|a| a == "zk");
    let seed = SeedDerivation::Salted;
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1; 32],
        merkle_root: [2; 32],
        timestamp: 0x6666_6666,
        nbits: 0x1E01FFFF,
    };
    let (m, n) = (256usize, 256usize);
    let mut cache = CircuitCache::default();
    let mut vcache = CircuitCache::default();

    println!("Part A. Plugin threshold k >= 1024 vs verifier rule k >= 16r (r = 128)");
    for k in [1024u32, 2048] {
        let config = gpu_config(k, 128);
        println!("\n==== k={k} r=128, GPU tile pattern h=2 w=64, m={m} n={n}");
        let t = Instant::now();
        let pp = mine(m, n, k as usize, header, config, None, false, seed).unwrap();
        println!("[k={k}] mine(): winning tile found in {:?}", t.elapsed());
        match check_cert_version_eligible(3, &pp) {
            Ok(v) => println!("[k={k}] check_cert_version_eligible(3) [miner and gateway pre-check]: ok ({v:?})"),
            Err(e) => println!("[k={k}] check_cert_version_eligible(3) [miner and gateway pre-check]: REJECTED: {e}"),
        }
        match verify::verify_plain_proof(&header, &pp, None, seed) {
            Ok(()) => println!("[k={k}] verify_plain_proof [pool share check]: ACCEPTED"),
            Err(e) => println!("[k={k}] verify_plain_proof [pool share check]: REJECTED: {e}"),
        }
        if !zk {
            continue;
        }
        let t = Instant::now();
        match prove::zk_prove_plain_proof(header, &pp, &mut cache, true, seed) {
            Ok(res) => {
                println!("[k={k}] zk_prove_plain_proof [gateway, sanity_check=true]: proof produced in {:?}", t.elapsed());
                let (pub_params, proof) = ZKProof::deserialize(header, seed, &res.public_data, &res.proof_data).unwrap();
                match verify::verify_block(&pub_params, &proof, &mut vcache) {
                    Ok(()) => println!("[k={k}] verify_block [node consensus]: ACCEPTED"),
                    Err(e) => println!("[k={k}] verify_block [node consensus]: REJECTED: {e}"),
                }
            }
            Err(e) => println!("[k={k}] zk_prove_plain_proof [gateway]: FAILED after {:?}: {e}", t.elapsed()),
        }
    }

    println!("\nPart B. Pool share bound vs node rank-penalty bound (r = 256, k = 4096)");
    let config = gpu_config(4096, 256);
    let share_bound = extract_difficulty_bound(header.nbits, &config);
    let node_bound = penalized_target_bound(nbits_to_difficulty(header.nbits), &config).unwrap();
    println!("  pool share bound / node rank-penalized bound = {} (= r/128): a hit counted as a share is a valid block hit only half the time", share_bound / node_bound);
    if !zk {
        println!("  (run with \"zk\" to check mined proofs against the node's rank-penalty rule)");
        return;
    }
    let (mut shares, mut node_ok) = (0, 0);
    for i in 1..=RANK_ATTEMPTS {
        let pp = mine(m, n, 4096, header, config, None, false, seed).unwrap();
        let share = verify::verify_plain_proof(&header, &pp, None, seed).is_ok();
        shares += share as usize;
        let res = prove::zk_prove_plain_proof(header, &pp, &mut cache, true, seed).unwrap();
        let (pub_params, proof) = ZKProof::deserialize(header, seed, &res.public_data, &res.proof_data).unwrap();
        let block = verify::verify_block(&pub_params, &proof, &mut vcache).is_ok();
        let penalty = check_rank_penalty(&config, &pub_params.hash_jackpot, header.nbits);
        node_ok += penalty.is_ok() as usize;
        println!(
            "  proof {i}: verify_plain_proof [pool share] {} | verify_block [ZK] {} | check_rank_penalty [node, after fork 96251] {}",
            if share { "ACCEPTED" } else { "REJECTED" },
            if block { "ACCEPTED" } else { "REJECTED" },
            match penalty { Ok(()) => "ok".to_string(), Err(e) => format!("REJECTED ({e})") }
        );
    }
    println!("  summary: {shares}/{RANK_ATTEMPTS} counted as pool shares, {node_ok}/{RANK_ATTEMPTS} pass the node's rank-penalty rule");
}

// P01 benchmark: any matrix, including all-zero, mines blocks that pass full consensus verification.
// Mines with four kinds of committed matrices (uniformly random, all-zero, constant 7, constant -64)
// and runs each proof through every check a block must pass: plain-proof verification, certificate
// version, ZK proving + node-side ZK verification, and the rank penalty.
// Usage: p01_any_matrix [zk]   (without "zk" only the plain-proof checks run)
use std::time::Instant;

use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, SeedDerivation, ZKProof};
use zk_pow::api::sanity_checks::check_rank_penalty;
use zk_pow::api::{prove, verify};
use zk_pow::circuit::circuit_utils::CircuitCache;
use zk_pow::ffi::mine::mine;
use zk_pow::ffi::plain_proof::{check_cert_version_eligible, PlainProof};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let do_zk = std::env::args().any(|a| a == "zk");
    let seed = SeedDerivation::Salted; // current V3 chain

    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [1; 32],
        merkle_root: [2; 32],
        timestamp: 0x6666_6666,
        nbits: 0x1E01FFFF, // node DefaultNBits (low difficulty, as in node/zkpow tests)
    };
    // node/zkpow default config: r=128, k=16r=2048, h=4, w=8
    let (m, n, k) = (256usize, 512usize, 2048usize);
    let config = MiningConfiguration {
        common_dim: k as u32,
        rank: 128,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
        moe: None,
    };

    let cases: [(&str, Option<(i8, i8)>); 4] = [
        ("random[-64,64] (default)", None),
        ("all-zero (0,0)", Some((0, 0))),
        ("constant 7 (7,7)", Some((7, 7))),
        ("constant -64 (-64,-64)", Some((-64, -64))),
    ];
    let mut cache = CircuitCache::default();
    let mut vcache = CircuitCache::default();

    for (name, range) in cases.iter() {
        println!("==== case: {name} | m={m} n={n} k={k} r=128 h=4 w=8 nbits=0x{:08X} seed=Salted(V3)", header.nbits);
        let t = Instant::now();
        let pp: PlainProof = match mine(m, n, k, header, config, *range, false, seed) {
            Ok(p) => p,
            Err(e) => {
                println!("[{name}] mine() ERROR: {e:?}");
                continue;
            }
        };
        println!("[{name}] mine() ok in {:?}", t.elapsed());
        // show opened strip contents (first 16 bytes of first opened A row / B col)
        let a0: Vec<i8> = pp.a.proof.leaf_data[0][..16].iter().map(|&b| b as i8).collect();
        let b0: Vec<i8> = pp.bt.proof.leaf_data[0][..16].iter().map(|&b| b as i8).collect();
        println!("[{name}] opened A rows={:?} B cols={:?}", pp.a.row_indices, pp.bt.row_indices);
        println!("[{name}] first 16 committed A bytes of opened leaf: {:?}", a0);
        println!("[{name}] first 16 committed B bytes of opened leaf: {:?}", b0);
        println!("[{name}] A root={} B root={}", hex(&pp.a.proof.root), hex(&pp.bt.proof.root));

        match verify::verify_plain_proof(&header, &pp, None, seed) {
            Ok(()) => println!("[{name}] verify_plain_proof: ACCEPTED"),
            Err(e) => println!("[{name}] verify_plain_proof: REJECTED: {e:?}"),
        }
        match check_cert_version_eligible(3, &pp) {
            Ok(v) => println!("[{name}] check_cert_version_eligible(3): ok ({v:?})"),
            Err(e) => println!("[{name}] check_cert_version_eligible(3): REJECTED: {e:?}"),
        }

        if do_zk {
            let t = Instant::now();
            match prove::zk_prove_plain_proof(header, &pp, &mut cache, true, seed) {
                Ok(res) => {
                    println!("[{name}] zk_prove (sanity_check=true) ok in {:?}, proof_data={} B", t.elapsed(), res.proof_data.len());
                    let (pub_params, zk) = ZKProof::deserialize(header, seed, &res.public_data, &res.proof_data).unwrap();
                    match verify::verify_block(&pub_params, &zk, &mut vcache) {
                        Ok(()) => println!("[{name}] verify_block (ZK, consensus path): ACCEPTED"),
                        Err(e) => println!("[{name}] verify_block (ZK): REJECTED: {e:?}"),
                    }
                    match check_rank_penalty(&config, &pub_params.hash_jackpot, header.nbits) {
                        Ok(()) => println!("[{name}] check_rank_penalty: ok"),
                        Err(e) => println!("[{name}] check_rank_penalty: REJECTED: {e:?}"),
                    }
                }
                Err(e) => println!("[{name}] zk_prove: ERROR: {e:?}"),
            }
        }
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().take(8).map(|x| format!("{x:02x}")).collect::<String>() + ".."
}

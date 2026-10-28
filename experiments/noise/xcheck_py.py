"""Python side of the cross-check; must reproduce rust_xcheck.out exactly.
Same inputs as rust_xcheck/src/main.rs: PublicProofParams::new_dummy (hash_a=[1]*32, hash_b=[2]*32),
m=256, n=384, t_rows=16, t_cols=4, k=2048, r=128, rows=[0,8,64,72], cols=[0,1,8,9,32,33,40,41].
"""
from pearl_ref import (header_bytes, mining_config_bytes, job_key, commitment_hash, compute_noise,
                       compute_jackpot, jackpot_hash, uniform_rows, perm_pairs, SEED_LABEL_A)

hdr = header_bytes(0, bytes([1] * 32), bytes([2] * 32), 0x66666666, 0x207FFFFF)
cfg = mining_config_bytes(2048, 128, [0, 8, 64, 72], [0, 1, 8, 9, 32, 33, 40, 41])
k, r, m, n = 2048, 128, 256, 384
rows = [16 + x for x in [0, 8, 64, 72]]
cols = [4 + x for x in [0, 1, 8, 9, 32, 33, 40, 41]]
for name, salted in [("legacy", False), ("salted", True)]:
    jk = job_key(hdr, cfg)
    b_seed, a_seed = commitment_hash(jk, bytes([1] * 32), bytes([2] * 32), m, n, salted)
    print(f"[{name}] header_bytes={hdr.hex()}")
    print(f"[{name}] config_bytes={cfg.hex()}")
    print(f"[{name}] job_key={jk.hex()}")
    print(f"[{name}] b_noise_seed={b_seed.hex()}")
    print(f"[{name}] a_noise_seed={a_seed.hex()}")
    print(f"[{name}] a_rows={rows} b_cols={cols}")
    na, nb, _ = compute_noise(k, r, b_seed, a_seed, rows, cols)
    sa = [[((u * 131 + l * 17) % 128) - 64 for l in range(k)] for u in range(len(rows))]
    sb = [[((v * 37 + l * 29 + 5) % 128) - 64 for l in range(k)] for v in range(len(cols))]
    jp = compute_jackpot(sa, sb, na, nb, k, r)
    print(f"[{name}] noise_a_row0_first16={na[0][:16]}")
    print(f"[{name}] noise_b_row0_first16={nb[0][:16]}")
    print(f"[{name}] noise_a_sum={sum(map(sum, na))} noise_b_sum={sum(map(sum, nb))}")
    print(f"[{name}] jackpot_msg={jp}")
    print(f"[{name}] jackpot_hash={jackpot_hash(jp, a_seed).hex()}")
    eal = uniform_rows(SEED_LABEL_A, a_seed, [0], 128)
    perm = [[f, s] for f, s, _ in perm_pairs(SEED_LABEL_A, a_seed, 8, 128)]
    print(f"[{name}] e_al_row0_first8={eal[0][:8]} perm_first8={perm}")

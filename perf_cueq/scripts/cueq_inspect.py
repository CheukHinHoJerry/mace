import cuequivariance as cue

d = cue.descriptors.symmetric_contraction(
    16 * cue.Irreps("O3", "0e+1o+2e"), 16 * cue.Irreps("O3", "0e"), (2,)
)
p = d.polynomial
print("polynomial:\n", p)
for opn, stp in p.operations:
    print("\noperation:", opn)
    print("  subscripts        :", stp.subscripts)
    print("  num_paths         :", stp.num_paths)
    for i, o in enumerate(stp.operands):
        print(
            f"  operand[{i}] subscripts={o.subscripts} num_segments={len(o.segments)} "
            f"size={o.size} first_segments={o.segments[:3]}"
        )
    print("  first 3 paths:")
    for pa in stp.paths[:3]:
        print("   ", pa)

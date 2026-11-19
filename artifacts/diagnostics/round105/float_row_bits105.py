"""Exact 32-bit Float row replication via uint16 word copy/transpose/masked interleave.

No arithmetic conversion: every Float bit (incl -0, subnormals, NaNs) stays unchanged.
Original910 provides uint16 vor and 16-bit transpose, not Float transpose/broadcast.
"""


def broadcast_float_row_bits(t,dst_float_flat,src_float_row,n1,mb,scratch_half,transpose_half,broadcast_float):
    if mb%8 or mb not in(32,64):raise ValueError('supported MB32/64 in blocks of8Float rows')
    if not isinstance(n1,int)or n1<1:raise ValueError('positive NZ groups required')
    words=src_float_row.reinterpret_cast_to('uint16')
    matrix=scratch_half.reinterpret_cast_to('uint16')
    trans=transpose_half.reinterpret_cast_to('uint16')
    output=broadcast_float.reinterpret_cast_to('uint16')
    # The transpose yields16 planes of16 identical Half WORDS, alternating
    # low/high Float halves. Masked vor interleaves those planes without casts.
    for block in range(mb//8):
        t.vec_or(16,matrix,words[block*16],words[block*16],16,1,0,0)
        t.vec_trans(transpose_half,scratch_half,1,0,0)
        # One destination row is32 Half words=16Float. src blk stride0 replicates
        # one16-word plane in both destination blocks; repeat stride2 jumps
        # over the opposite plane for the next Float row. Both masks cover all
        # 32 destination words, exactly once, for8 rows.
        t.vor([0,0x55555555],output[block*256],trans[0],trans[0],8,1,0,0,2,2,2)
        t.vor([0,0xAAAAAAAA],output[block*256],trans[16],trans[16],8,1,0,0,2,2,2)
    for group in range(n1):
        t.data_move(dst_float_flat[group*mb*16],broadcast_float,0,1,mb*16*4//32,0,0)
    return mb//8

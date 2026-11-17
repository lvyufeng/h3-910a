"""Explicit output-row partitions for four actual L0 vision Linears; never split K."""
WIDTHS={'qkv':3456,'proj':1152,'fc1':4304,'fc2':1152}
INPUTS={'qkv':1152,'proj':1152,'fc1':1152,'fc2':4304}


def indices(name,rank,world=8):
    if name not in WIDTHS or world!=8 or not 0<=rank<world:raise ValueError('Unknown vision projection/rank/width')
    if name=='qkv':
        # Q/K/V are three outer groups,each has16x72 features. A local
        # attention partition needs TWO HEADS FROM EACH GROUP,not6Q-onlyheads.
        return [group*1152+index for group in range(3)for index in range(rank*144,(rank+1)*144)]
    width=WIDTHS[name]
    if width%world:raise ValueError('Expected exact output-column partition')
    local=width//world
    return list(range(rank*local,(rank+1)*local))


def padded_width(name,aligned=False):
    width=len(indices(name,0))
    return (width+15)//16*16 if aligned else width


def validate_plan():
    for name,width in WIDTHS.items():
        parts=[indices(name,rank)for rank in range(8)]
        flattened=[index for part in parts for index in part]
        if sorted(flattened)!=list(range(width))or len(set(flattened))!=width:raise ValueError('Output-row omission/overlap')
        if any(len(part)!=width//8 for part in parts):raise ValueError('Unequal width')
        if any(len(part)%2 for part in parts):raise ValueError('Raw BF16 pair packing requires even width')
    return True


validate_plan()

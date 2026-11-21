"""Pure scheduling contract: every VAE rank reaches the same tile-proof gate count."""

def aligned_local_tile_rounds(local_count, global_max_count):
    if type(local_count) is not int or type(global_max_count) is not int:
        raise TypeError('Tile counts must be plain integers')
    if not 0 <= local_count <= global_max_count <= 128:
        raise ValueError('Invalid or unbounded tile-proof counts')
    return [(i, i < local_count) for i in range(global_max_count)]

#!/usr/bin/env python3
"""Save only explicitly selected full-channel rows of diagnostic refiner captures.

Full tensors remain on the accelerator host. This bounded sample avoids multi-GB
activation transfers; error metrics must say sampled rows, not full-array precision.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    import torch
    root = Path(args.root)
    source = root / args.source
    target = root / (args.source + '_samples')
    assert not target.exists()
    target.mkdir()
    bundle = torch.load(root / 'same_initial_sampling123.pt', map_location='cpu', weights_only=True)
    tags = bundle['text_tags']
    # Every actual text row + 96 evenly spaced visual rows, complete channels.
    visual_rows = torch.nonzero(tags == 0).flatten()
    choices = torch.linspace(0, len(visual_rows) - 1, 96, dtype=torch.float64).round().long()
    selected = torch.cat((torch.nonzero(tags == 1).flatten(), visual_rows[choices])).unique(sorted=True)
    report = {'scope': __doc__, 'completed': False, 'source': str(source),
              'selected_rows': selected.tolist(), 'selected_token_tags': tags[selected].tolist(),
              'selected_rows_count': len(selected), 'full_rows_count': len(tags),
              'metric_scope': 'selected complete-channel rows, not full-array error', 'files': {}}
    for name in ['condition_proj', 'refiner_0_qkv', 'refiner_0_attention', 'refiner_0_MLP', 'refiner_0_residual',
                 'refiner_1_qkv', 'refiner_1_attention', 'refiner_1_MLP', 'refiner_1_residual', 'refiner_final']:
        for rank in [0, 1]:
            path = source / f'{name}_rank{rank}.pt'
            if not path.is_file():
                continue
            value = torch.load(path, map_location='cpu', weights_only=True)['value']
            assert value.shape[0] == len(tags)
            payload = {'value': value.index_select(0, selected).contiguous(), 'row_indices': selected,
                       'token_tags': tags[selected], 'full_shape': list(value.shape)}
            output = target / path.name
            with output.open('xb') as file:
                torch.save(payload, file)
            report['files'][path.name] = {'bytes': output.stat().st_size, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                                         'shape': list(payload['value'].shape), 'dtype': str(value.dtype)}
    report['completed'] = True
    (target / 'sample_manifest123.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'completed': True, 'sampled_rows': len(selected), 'files': len(report['files']),
                      'total_sample_bytes': sum(x['bytes'] for x in report['files'].values()), 'metric_scope': report['metric_scope']}, indent=2), flush=True)


if __name__ == '__main__':
    main()

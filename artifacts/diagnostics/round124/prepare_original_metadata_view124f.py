#!/usr/bin/env python3
"""Build an original-metadata NPU checkpoint view without copying any weights.

The resident NPU model_index adds a4-interval base_schedule that is absent from
our Git-blob-verified official GPU metadata at the pinned revision. Preserve the
resident file unchanged, use the exact official metadata in an experiment-only
view, and symlink all component entries to the already verified original NPU
checkpoint. This restores the legacy49-evaluation GPU comparison contract via
the existing uniform-schedule API, not a student/LoRA or hidden weight conversion.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path('/tmp/h3_align12401')
ORIGINAL=Path('/home/mseco/models/MiniMax-H3/FL2VA')
EXPECTED_GIT_BLOB='ed4dfb751d8417e5ab8cf58a6915214bda6ba676'


def git_blob_sha1(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def main():
    canonical=ROOT/'original_GPU_model_index124.json'
    data=canonical.read_bytes()
    assert len(data)==719 and git_blob_sha1(data)==EXPECTED_GIT_BLOB
    official=json.loads(data);resident_data=(ORIGINAL/'model_index.json').read_bytes();resident=json.loads(resident_data)
    assert 'base_schedule' not in official['_minimax_h3']
    changed=json.loads(resident_data)
    extra=changed['_minimax_h3'].pop('base_schedule')
    assert extra==[1.0,0.7,0.4,0.15,0.0] and changed==official
    parent=ROOT/'original_metadata124f';assert not parent.exists();parent.mkdir()
    view=parent/'FL2VA';view.mkdir()
    (view/'model_index.json').write_bytes(data)
    links={}
    for entry in ORIGINAL.iterdir():
        if entry.name=='model_index.json':continue
        destination=view/entry.name
        assert not destination.exists() and destination.parent.resolve()==view.resolve()
        destination.symlink_to(entry,target_is_directory=entry.is_dir())
        assert destination.resolve()==entry.resolve()
        links[entry.name]=str(entry)
    assert (ORIGINAL/'model_index.json').read_bytes()==resident_data
    report={'scope':__doc__,'completed':True,'view_checkpoint':str(view),'original_NPU_checkpoint':str(ORIGINAL),
            'model_weights_copied_or_downloaded':False,'original_NPU_metadata_modified':False,
            'canonical_model_index_bytes':len(data),'canonical_official_git_blob':EXPECTED_GIT_BLOB,
            'resident_model_index_git_blob':git_blob_sha1(resident_data),
            'only_semantic_metadata_difference':'Resident NPU metadata adds base_schedule [1,.7,.4,.15,0]; official pinned metadata does not.',
            'removed_only_in_private_view_not_original':{'base_schedule':extra},'symlinked_original_entries':links,
            'original13_transformer_shards_previously_verified_against_GPU':True,
            'same49_evaluation_uniform_sigma_contract_as_actual_GPU_reference':True}
    receipt=ROOT/'original_metadata_view124f.json';assert not receipt.exists();receipt.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

"""Small owned/atomic JSON writer for new diagnostic runs, not historical proofs.

A persistent .claim makes duplicate run IDs fail closed, even after a crash.
Use a new exact filename/run ID to retry. Readers see one complete JSON revision;
we do not make a half-written file look like a completed hardware experiment.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid


class EvidenceWriter:
    def __init__(self, path):
        self.path = Path(path).absolute()
        self.parent = self.path.parent.resolve(strict=True)
        if self.path.parent != self.parent or self.path.is_symlink():
            raise ValueError('Evidence path must have a real, non-symlink parent')
        if self.path.exists():
            raise FileExistsError('Refuse existing evidence: ' + str(self.path))
        self.claim = self.path.with_name(self.path.name + '.claim')
        self.token = uuid.uuid4().hex
        # Exclusive, persistent owner record; no stale-claim guessing/replay.
        with self.claim.open('x', encoding='utf-8') as f:
            f.write(json.dumps({'pid': os.getpid(), 'token': self.token}) + '\n')
        if self.path.exists():
            raise FileExistsError('Evidence appeared while claiming its run ID')
        self.previous_sha256 = None

    def write(self, payload):
        content = (json.dumps(payload, indent=2) + '\n').encode('utf-8')
        if self.path.parent.resolve(strict=True) != self.parent or self.path.is_symlink():
            raise ValueError('Evidence destination changed since claim')
        owner = json.loads(self.claim.read_text())
        if owner['token'] != self.token:
            raise RuntimeError('Diagnostic evidence ownership changed')
        if self.path.exists():
            observed = hashlib.sha256(self.path.read_bytes()).hexdigest()
            if observed != self.previous_sha256:
                raise RuntimeError('Refuse to overwrite an externally changed receipt')
        elif self.previous_sha256 is not None:
            raise RuntimeError('Previously published receipt disappeared')
        fd, name = tempfile.mkstemp(prefix='.evidence63_', dir=self.parent)
        temp = Path(name)
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            # Check both resolved absolute endpoints before atomic replacement.
            if temp.resolve(strict=True) != temp or temp.parent != self.parent:
                raise RuntimeError('Unexpected evidence temporary path')
            if self.path.resolve() != self.path or self.path.parent != self.parent:
                raise RuntimeError('Unexpected evidence destination')
            os.replace(temp, self.path)
            self.previous_sha256 = hashlib.sha256(content).hexdigest()
        finally:
            if temp.exists():
                if temp.resolve(strict=True) != temp or temp.parent != self.parent or not temp.name.startswith('.evidence63_'):
                    raise RuntimeError('Unexpected evidence temporary cleanup path')
                temp.unlink()

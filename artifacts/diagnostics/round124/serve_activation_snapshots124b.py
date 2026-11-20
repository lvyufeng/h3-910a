#!/usr/bin/env python3
"""Serve only completed diagnostic snapshot files between two owned tailnet machines.

Token authentication, private tailnet bind, exact file allowlist, no directory
listing and no model weight paths. Each data file is exposed only after its matching
JSON receipt exists, avoiding partial checkpoint/snapshot reads. This is a temporary
read-only activation transfer service, never a public model server.
"""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import socket
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--port',type=int,default=8127)
    args=parser.parse_args()
    root=Path(args.root).resolve();assert root.is_dir()
    token_path=root/'activation_transfer_token124.txt'
    assert not token_path.exists()
    token=secrets.token_hex(32);token_path.write_text(token+'\n');token_path.chmod(0o600)
    ip=subprocess.check_output(['tailscale','ip','-4'],text=True).strip()
    assert ip.startswith('100.') and '\n' not in ip
    source=(root/'GPU_trajectory124').resolve()
    allowed={f'snapshot_step{step:02d}.{suffix}' for step in [0,10,25,40,48] for suffix in ['pt','json']}
    allowed.update({'GPU_snapshot_context124.pt','GPU_snapshot_context124.json'})
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if not hmac.compare_digest(self.headers.get('X-H3-Activation-Token',''),token):
                self.send_error(403);return
            name=self.path.removeprefix('/')
            if name not in allowed:
                self.send_error(404);return
            path=(source/name).resolve()
            if path.parent!=source or not path.is_file() or not path.with_suffix('.json').is_file():
                self.send_error(404);return
            size=path.stat().st_size
            self.send_response(200);self.send_header('Content-Length',str(size));self.send_header('Content-Type','application/octet-stream');self.end_headers()
            with path.open('rb') as f:
                while data:=f.read(1024*1024):self.wfile.write(data)
        def log_message(self,format,*args):
            print(json.dumps({'activation_transfer':format%args}),flush=True)
    report={'scope':__doc__,'bind_ip':ip,'port':args.port,'allowed_file_names':sorted(allowed),'token_not_disclosed_in_receipt':True,'model_weights_served':False}
    (root/'activation_server_receipt124.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'read_only_activation_server_started':True,'tailnet_ip':ip,'port':args.port}),flush=True)
    ThreadingHTTPServer((ip,args.port),Handler).serve_forever()


if __name__=='__main__':main()

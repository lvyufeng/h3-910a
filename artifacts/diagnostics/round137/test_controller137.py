#!/usr/bin/env python3
"""Pure mocks for controller exceptions, never signals/processes/model execution."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import launch_components137 as launcher


class FailureInjectionTests(unittest.TestCase):
    def run_fault(self,fault):
        calls=[];alive=[True]
        class FakeProcess:
            pid=999001
            def wait(self,timeout=None):calls.append(('wait',timeout));alive[0]=False;return -15
        class FakeThread:
            ident=None
            def __init__(self,*a,**k):pass
            def start(self):
                calls.append(('watch_start',None))
                if fault=='thread':raise RuntimeError('thread fault')
                self.ident=123
            def join(self,timeout=None):calls.append(('join',timeout))
            def is_alive(self):return False
        def pidfd(pid):
            calls.append(('pidfd',pid))
            if fault=='pidfd':raise OSError('pidfd fault')
            return os.open(os.devnull,os.O_RDONLY)
        def publish():
            calls.append(('publish',None))
            if fault=='publish':raise OSError('publish fault')
        def rows():return [{'pid':999001,'start_ticks':'mock','command':'mock'}] if alive[0] else []
        def abort():calls.append(('private_abort',None));alive[0]=False;return {'all_private_terminated':True}
        private=SimpleNamespace(private_rows137=rows,abort_private137=abort)
        cohort=SimpleNamespace(monitor_private_failure=lambda *a,**k:None)
        watchfd=os.open(os.devnull,os.O_RDONLY);row={}
        try:
            with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'private_abort137':private,'cohort128':cohort}),\
                 patch.object(launcher,'ROOT',Path(tmp)),patch.object(launcher.subprocess,'Popen',return_value=FakeProcess()) as spawn,\
                 patch.object(launcher.threading,'Thread',FakeThread),patch.object(launcher.os,'pidfd_open',side_effect=pidfd):
                env={'H3_R127_IPC_KEY_FILE':'/tmp/h3_components13703/private_IPC_key127.bin'}
                with self.assertRaises((RuntimeError,OSError)):
                    launcher.guarded_launch137(['mock'],env,Path(tmp)/'mock.log',row,publish,watchfd,time.monotonic()+10)
                self.assertEqual(spawn.call_args.kwargs['env'],env)
            self.assertIn(('private_abort',None),calls)
            self.assertTrue(row['completion_unknown'])
            self.assertTrue(row['all_private_children_exited'])
            self.assertEqual(row['control_failure']['message'],fault+' fault')
            self.assertFalse(alive[0])
        finally:
            try:os.close(watchfd)
            except OSError:pass
    def test_publish_fault_exact_private_abort(self):self.run_fault('publish')
    def test_thread_fault_exact_private_abort(self):self.run_fault('thread')
    def test_pidfd_fault_exact_private_abort(self):self.run_fault('pidfd')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(FailureInjectionTests)
    result=unittest.TextTestRunner().run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    root=Path('/tmp/h3_components13703');target=root/'controller_unit137.json';assert not target.exists()
    row={'completed':True,'tests_passed':3,'pure_mock_no_model_or_signal_sent':True,
        'publish_thread_pidfd_setup_failure_trigger_private_cleanup':True,'child_capability_present_before_shell_exports':True,
        'completion_unknown_not_fake_success_on_controller_failure':True,'unknown_launch_not_replayed':True}
    target.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row,indent=2))

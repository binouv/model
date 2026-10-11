from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from durable_git import publish_commit


def git(cwd, *args):
    result = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


class DurableTests(unittest.TestCase):
    def test_concurrent_main_and_second_batch(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); remote=base/'remote'; worker=base/'worker'; other=base/'other'
            git(base,'init','--bare',str(remote))
            git(base,'clone',str(remote),str(worker))
            git(worker,'config','user.name','fixture');git(worker,'config','user.email','fixture@invalid')
            git(worker,'checkout','-b','main')
            (worker/'README.md').write_text('fixture')
            git(worker,'add','.');git(worker,'commit','-m','fixture');git(worker,'push','origin','main')
            git(base,'clone','--branch','main',str(remote),str(other))
            git(other,'config','user.name','other');git(other,'config','user.email','other@invalid')
            root=worker/'models/l100da_20261012_step0256/native';root.mkdir(parents=True)
            (root/'part').write_bytes(b'original bytes')
            conflicts=0
            def concurrent(*args):
                nonlocal conflicts
                if args==('push','origin','HEAD:refs/heads/main') and conflicts<4:
                    git(other,'pull','--ff-only','origin','main')
                    (other/'unrelated.txt').write_text(str(conflicts))
                    git(other,'add','.');git(other,'commit','-m',f'concurrent{conflicts}')
                    git(other,'push','origin','main');conflicts+=1
                return git(worker,*args)
            with patch('durable_git.time.sleep'):
                commit=publish_commit(concurrent,['models/l100da_20261012_step0256/native/part'],'publish','123',1)
            self.assertEqual(conflicts,4)
            self.assertEqual(git(remote,'rev-parse','main'),commit)
            self.assertEqual((worker/'unrelated.txt').read_text(),'3')
            self.assertEqual(git(remote,'show','flygraph/l100da-artifacts-123-001:models/l100da_20261012_step0256/native/part'),'original bytes')
            (root/'part2').write_bytes(b'second')
            publish_commit(concurrent,['models/l100da_20261012_step0256/native/part2'],'second','123',2)
            self.assertEqual(git(remote,'show','main:unrelated.txt'),'3')
            self.assertEqual(git(remote,'show','main:models/l100da_20261012_step0256/native/part2'),'second')
    def test_rejects_unrelated_path(self):
        with self.assertRaises(ValueError):publish_commit(lambda *a:'',['README.md'],'bad','123',1)
    def test_rejects_unsafe_identity(self):
        with self.assertRaises(ValueError):publish_commit(lambda *a:'',[],'bad','../../x',1)
    def test_rejects_conflicting_savepoint(self):
        def fake(*a):
            if a[0]=='ls-remote':return 'different refs/heads/x'
            return 'head' if a[0]=='rev-parse' else ''
        with self.assertRaises(RuntimeError):publish_commit(fake,[],'bad','123',1)

if __name__=='__main__':unittest.main()

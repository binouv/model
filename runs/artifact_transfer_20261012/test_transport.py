import copy
import json
from pathlib import Path
import tempfile
import unittest
import transport as t

class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.key = Path(cls.temp.name) / 'key.pem'
        cls.public = t.make_key(cls.key)
        cls.payload = {'transfer_id': cls.public['transfer_id'], 'checkpoint_sha256': t.MODEL_SHA,
                       'files': [{'name': 'test', 'url': 'https://sample.oaiusercontent.com/file?sig=PRIVATE-TEST-' + 'x' * 1200}]}
        cls.cipher = t.encrypt(cls.public, cls.payload)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def test_roundtrip(self): self.assertEqual(t.decrypt(self.key, self.public, self.cipher), self.payload)
    def test_no_plaintext(self): self.assertNotIn('PRIVATE-TEST', json.dumps(self.cipher))
    def test_mode(self): self.assertEqual(self.key.stat().st_mode & 0o777, 0o600)
    def test_order(self):
        c=copy.deepcopy(self.cipher); c['blocks'].reverse()
        with self.assertRaises(ValueError): t.decrypt(self.key,self.public,c)
    def test_nonce(self):
        c=copy.deepcopy(self.cipher); c['transfer_id']='0'*32
        with self.assertRaises(ValueError): t.decrypt(self.key,self.public,c)
    def test_fingerprint(self):
        c=copy.deepcopy(self.cipher); c['public_key_sha256']='0'*64
        with self.assertRaises(ValueError): t.decrypt(self.key,self.public,c)
    def test_truncation(self):
        c=copy.deepcopy(self.cipher); c['blocks'].pop()
        with self.assertRaises(ValueError): t.decrypt(self.key,self.public,c)
    def test_corruption(self):
        c=copy.deepcopy(self.cipher); c['blocks'][0]=('B' if c['blocks'][0][0]=='A' else 'A')+c['blocks'][0][1:]
        with self.assertRaises(ValueError): t.decrypt(self.key,self.public,c)
    def test_oversize(self):
        with self.assertRaises(ValueError): t.encrypt(self.public,{'transfer_id':self.public['transfer_id'],'blob':'x'*17000})
    def test_host_policy(self):
        self.assertTrue(t.allowed_url('https://sample.oaiusercontent.com/file?sig=example'))
        for url in ('http://sample.oaiusercontent.com/file','https://evil.example/file','https://sample.oaiusercontent.com.evil.example/file','https://user:secret@sample.oaiusercontent.com/file','https://sample.oaiusercontent.com:444/file'):
            with self.subTest(url=url): self.assertFalse(t.allowed_url(url))
    def test_invalid_download(self):
        with self.assertRaises(ValueError): t.download('http://127.0.0.1',Path(self.temp.name)/'out',{'size_bytes':1})
if __name__=='__main__': unittest.main(verbosity=2)

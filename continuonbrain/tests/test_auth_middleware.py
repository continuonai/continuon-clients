import unittest
from unittest.mock import patch, MagicMock
import json
import time
import os
from pathlib import Path
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cryptography.hazmat.primitives import hashes
import datetime

from continuonbrain.core.security import UserRole
from continuonbrain.api.middleware.auth import AuthProvider, get_auth_provider

class TestAuthMiddleware(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path("/tmp/continuon_test_auth")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.auth_provider = AuthProvider(self.tmp_dir)

        # Generate RSA Key Pair for testing
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
        # Generate Self-Signed Certificate containing the public key
        subject = issuer = x509.Name([
            x509.NameAttribute(x509.NameOID.COMMON_NAME, u"continuon.ai"),
        ])
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            self.public_key
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=10)
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True,
        ).sign(self.private_key, hashes.SHA256(), default_backend())

        self.pem_public = cert.public_bytes(serialization.Encoding.PEM).decode('utf-8')
        
        self.kid = "test_key_1"
        self.fake_keys = {self.kid: self.pem_public}

    def tearDown(self):
        import shutil
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch('requests.get')
    def test_fetch_public_keys(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.fake_keys
        mock_get.return_value = mock_response

        keys = self.auth_provider.get_public_keys()
        self.assertEqual(keys, self.fake_keys)
        self.assertTrue(self.auth_provider._keys_cache)
        self.assertGreater(self.auth_provider._keys_expire, time.time())

    def test_verify_valid_token(self):
        # Setup provider with known keys
        self.auth_provider._keys_cache = self.fake_keys
        self.auth_provider._keys_expire = time.time() + 3600

        # Create Token
        payload = {
            "iss": "https://securetoken.google.com/continuon-ai",
            "aud": "continuon-ai",
            "auth_time": int(time.time()),
            "user_id": "test_user_123",
            "sub": "test_user_123",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "email": "test@example.com",
            "email_verified": True,
            "role": "creator"
        }
        
        token = jwt.encode(
            payload, 
            self.private_key, 
            algorithm="RS256",
            headers={"kid": self.kid}
        )

        # Verify
        with patch.dict(os.environ, {"FIREBASE_PROJECT_ID": "continuon-ai"}):
            # Re-init to pick up env var? No, PROJECT_ID is module level constant.
            # We must patch the module constant or ensure env var is set before import.
            # For this test, assume default or patch where used.
            # The class uses global PROJECT_ID. 
            # Ideally we'd refactor to inject project_id, but for now let's mock the verify call's context if needed.
            # Actually, `continuon-ai` IS the default in the code, so it should match.
            
            decoded = self.auth_provider.verify_token(token)
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded["email"], "test@example.com")
            
            role = self.auth_provider.get_user_role(decoded)
            self.assertEqual(role, UserRole.CREATOR)

    def test_verify_expired_token(self):
        self.auth_provider._keys_cache = self.fake_keys
        self.auth_provider._keys_expire = time.time() + 3600

        payload = {
            "iss": "https://securetoken.google.com/continuon-ai",
            "aud": "continuon-ai",
            "exp": int(time.time()) - 10, # Expired
            "sub": "test",
            "role": "consumer"
        }
        token = jwt.encode(payload, self.private_key, algorithm="RS256", headers={"kid": self.kid})
        
        decoded = self.auth_provider.verify_token(token)
        self.assertIsNone(decoded)

    def test_verify_wrong_audience(self):
        self.auth_provider._keys_cache = self.fake_keys
        self.auth_provider._keys_expire = time.time() + 3600

        payload = {
            "iss": "https://securetoken.google.com/continuon-ai",
            "aud": "wrong-project",
            "exp": int(time.time()) + 3600,
            "sub": "test"
        }
        token = jwt.encode(payload, self.private_key, algorithm="RS256", headers={"kid": self.kid})
        
        decoded = self.auth_provider.verify_token(token)
        self.assertIsNone(decoded)

if __name__ == '__main__':
    unittest.main()

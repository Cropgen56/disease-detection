import subprocess
import time
import urllib.request
import urllib.error
import json
import unittest
import sys

class TestEndpointsRealServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start uvicorn server as a subprocess on port 8005 using the active Python executable
        cls.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8005"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Give the server a moment to spin up and load
        time.sleep(2.5)

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait()

    def test_root_endpoint(self):
        try:
            req = urllib.request.Request("http://127.0.0.1:8005/")
            with urllib.request.urlopen(req) as response:
                self.assertEqual(response.status, 200)
                data = json.loads(response.read().decode())
                self.assertEqual(data["status"], "online")
                self.assertFalse(data["phase1_active"])
                self.assertFalse(data["phase2_active"])
        except urllib.error.URLError as e:
            self.fail(f"Failed to connect to local server: {e}")

    def test_metadata_endpoint(self):
        try:
            req = urllib.request.Request("http://127.0.0.1:8005/api/v1/metadata")
            with urllib.request.urlopen(req) as response:
                self.assertEqual(response.status, 200)
                data = json.loads(response.read().decode())
                self.assertEqual(data["supported_crops_count"], 0)
                self.assertEqual(data["supported_crops"], [])
                self.assertFalse(data["phase1_active"])
                self.assertFalse(data["phase2_active"])
        except urllib.error.URLError as e:
            self.fail(f"Failed to connect to local server: {e}")

    def test_crop_classifier_returns_503(self):
        # Setup form data
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        data = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="test.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
            "dummy_bytes\r\n"
            f"--{boundary}--\r\n"
        ).encode('utf-8')

        req = urllib.request.Request(
            "http://127.0.0.1:8005/api/v1/classify-crop",
            data=data,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}"
            }
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                self.fail("Expected HTTP 503, but request succeeded with 200")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 503)
            resp_body = json.loads(e.read().decode())
            self.assertIn("Phase 1 Crop Classification model is not loaded", resp_body["detail"])

    def test_disease_detector_returns_503(self):
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        data = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="test.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
            "dummy_bytes\r\n"
            f"--{boundary}--\r\n"
        ).encode('utf-8')

        req = urllib.request.Request(
            "http://127.0.0.1:8005/api/v1/detect-disease?crop=tomato",
            data=data,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}"
            }
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                self.fail("Expected HTTP 503, but request succeeded with 200")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 503)
            resp_body = json.loads(e.read().decode())
            self.assertIn("Phase 2 Disease Detection model is not loaded", resp_body["detail"])

if __name__ == "__main__":
    unittest.main()

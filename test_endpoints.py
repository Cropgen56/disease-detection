import subprocess
import time
import urllib.request
import urllib.error
import json
import unittest
import sys
import os

class TestEndpointsRealServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start uvicorn server as a subprocess on port 8005 using the active Python executable
        cls.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8005"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Give the server a moment to spin up and load models (DINOv2 can take a bit longer)
        time.sleep(7.0)

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
                
                # Check dynamic active status
                has_phase1 = (
                    os.path.exists("weights/stage1_dinov2_30class.pt") or
                    os.path.exists("weights/efficientnet_b1_crop_mini.pt") or
                    os.path.exists("weights/yolo11n_crop_mini.pt")
                ) and os.path.exists("weights/class_to_idx_phase1.json")
                has_phase2 = os.path.exists("weights/efficientnet_b0_disease_mini.pt") and os.path.exists("weights/crop_disease_idx_map.json")

                self.assertEqual(data["phase1_active"], has_phase1)
                self.assertEqual(data["phase2_active"], has_phase2)
        except urllib.error.URLError as e:
            self.fail(f"Failed to connect to local server: {e}")

    def test_metadata_endpoint(self):
        try:
            req = urllib.request.Request("http://127.0.0.1:8005/api/v1/metadata")
            with urllib.request.urlopen(req) as response:
                self.assertEqual(response.status, 200)
                data = json.loads(response.read().decode())
                
                has_phase1 = (
                    os.path.exists("weights/stage1_dinov2_30class.pt") or
                    os.path.exists("weights/efficientnet_b1_crop_mini.pt") or
                    os.path.exists("weights/yolo11n_crop_mini.pt")
                ) and os.path.exists("weights/class_to_idx_phase1.json")
                has_phase2 = os.path.exists("weights/efficientnet_b0_disease_mini.pt") and os.path.exists("weights/crop_disease_idx_map.json")

                if has_phase2:
                    with open("weights/crop_disease_idx_map.json", "r") as f:
                        mapping = json.load(f)
                    expected_count = len(mapping)
                    expected_crops = sorted(list(mapping.keys()))
                else:
                    expected_count = 0
                    expected_crops = []

                self.assertEqual(data["supported_crops_count"], expected_count)
                self.assertEqual(data["supported_crops"], expected_crops)
                self.assertEqual(data["phase1_active"], has_phase1)
                self.assertEqual(data["phase2_active"], has_phase2)
        except urllib.error.URLError as e:
            self.fail(f"Failed to connect to local server: {e}")

    def test_crop_classifier_error_handling(self):
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
        
        has_phase1 = (
            os.path.exists("weights/stage1_dinov2_30class.pt") or
            os.path.exists("weights/efficientnet_b1_crop_mini.pt") or
            os.path.exists("weights/yolo11n_crop_mini.pt")
        ) and os.path.exists("weights/class_to_idx_phase1.json")

        try:
            with urllib.request.urlopen(req) as response:
                self.fail("Expected HTTP error, but request succeeded with 200")
        except urllib.error.HTTPError as e:
            if has_phase1:
                self.assertEqual(e.code, 400)
                resp_body = json.loads(e.read().decode())
                self.assertIn("Uploaded file is not a valid image", resp_body["detail"])
            else:
                self.assertEqual(e.code, 503)
                resp_body = json.loads(e.read().decode())
                self.assertIn("Phase 1 Crop Classification model is not loaded", resp_body["detail"])

    def test_disease_detector_error_handling(self):
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
        
        has_phase2 = os.path.exists("weights/efficientnet_b0_disease_mini.pt") and os.path.exists("weights/crop_disease_idx_map.json")

        try:
            with urllib.request.urlopen(req) as response:
                self.fail("Expected HTTP error, but request succeeded with 200")
        except urllib.error.HTTPError as e:
            if has_phase2:
                self.assertEqual(e.code, 400)
                resp_body = json.loads(e.read().decode())
                self.assertIn("Uploaded file is not a valid image", resp_body["detail"])
            else:
                self.assertEqual(e.code, 503)
                resp_body = json.loads(e.read().decode())
                self.assertIn("Phase 2 Disease Detection model is not loaded", resp_body["detail"])

if __name__ == "__main__":
    unittest.main()

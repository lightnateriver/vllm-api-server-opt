import time
import requests

from bundle_paths import HOST as VLLM_HOST, REQUEST_MODEL, TEST_IMAGES_DIR

IMG1 = f"file://{(TEST_IMAGES_DIR / 'test_img_1.png').resolve()}"
IMG2 = f"file://{(TEST_IMAGES_DIR / 'test_img_2.png').resolve()}"

payload_same = {
    "model": REQUEST_MODEL,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in one sentence."},
                {"type": "image_url", "image_url": {"url": IMG1}},
                {"type": "image_url", "image_url": {"url": IMG1}},
                {"type": "image_url", "image_url": {"url": IMG1}},
            ]
        }
    ],
    "max_tokens": 64,
    "temperature": 0.1,
    "chat_template_kwargs": {"enable_thinking": False}
}

print("=== Test 1: Same image repeated 3 times ===")
times = []
for i in range(3):
    start = time.perf_counter()
    resp = requests.post(f"{VLLM_HOST}/v1/chat/completions", json=payload_same, timeout=120)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    times.append(latency_ms)
    print(f"  Request {i+1}: {latency_ms:.1f}ms, tokens: {resp.json()['usage']['total_tokens']}")

print(f"  First: {times[0]:.1f}ms, Second: {times[1]:.1f}ms, Third: {times[2]:.1f}ms")
if times[1] < times[0] * 0.7:
    print("  Hash hit detected! (2nd request much faster)")
else:
    print("  No clear hash hit")

payload_diff = {
    "model": REQUEST_MODEL,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in one sentence."},
                {"type": "image_url", "image_url": {"url": IMG1}},
                {"type": "image_url", "image_url": {"url": IMG2}},
            ]
        }
    ],
    "max_tokens": 64,
    "temperature": 0.1,
    "chat_template_kwargs": {"enable_thinking": False}
}

print("\n=== Test 2: Different images (no hash hit expected) ===")
times_diff = []
for i in range(2):
    start = time.perf_counter()
    resp = requests.post(f"{VLLM_HOST}/v1/chat/completions", json=payload_diff, timeout=120)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    times_diff.append(latency_ms)
    print(f"  Request {i+1}: {latency_ms:.1f}ms")

print(f"  First: {times_diff[0]:.1f}ms, Second: {times_diff[1]:.1f}ms")
if times_diff[1] > times_diff[0] * 0.8:
    print("  Expected: similar times (no hash hit)")
else:
    print("  Unexpected: 2nd much faster (might be other caching)")

print("\n=== Summary ===")
print(f"Test 1 (same image): {times[0]:.1f}ms -> {times[1]:.1f}ms -> {times[2]:.1f}ms")
print(f"Test 2 (diff image): {times_diff[0]:.1f}ms -> {times_diff[1]:.1f}ms")

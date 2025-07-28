import requests
import time

url = "http://127.0.0.1:8000/generate"  # Replace with ngrok URL if needed
data = {
    "prompt": "how to manage time effectively?",
    "max_new_tokens": 20
}

# Warm-up
requests.post(url, json=data)

# Benchmark
n_requests = 20
times = []

for _ in range(n_requests):
    start = time.time()
    _ = requests.post(url, json=data)
    times.append(time.time() - start)

avg_latency = sum(times) / n_requests
throughput = n_requests / sum(times)

print(f"📊 Avg Latency: {avg_latency * 1000:.2f} ms")
print(f"⚡ Throughput: {throughput:.2f} requests/sec")

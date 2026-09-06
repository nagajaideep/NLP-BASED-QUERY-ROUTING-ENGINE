"""
RouteIQ Local Model Latency Test
Measures the model inference speed on localhost:8000
NOTE: Make sure the server is already running before executing this script!
"""

import requests
import time
import statistics

print("=" * 60)
print("RouteIQ Local Model Latency Test")
print("=" * 60)

# Test queries
test_queries = [
    "My payment failed but the amount was deducted.",
    "I want to return my order.",
    "How do I reset my password?",
    "The product I received is damaged.",
    "I need to speak to a supervisor immediately.",
]

url = "http://localhost:8000/predict"

print("\nChecking if server is running...")
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    if response.status_code == 200:
        print("✓ Server is running and healthy!\n")
    else:
        print(f"✗ Server responded with status {response.status_code}")
        exit(1)
except Exception as e:
    print(f"✗ Cannot connect to server: {e}")
    print("\nPlease start the server first with:")
    print("  python -m uvicorn backend.main:app --host localhost --port 8000")
    exit(1)

# Warm-up request
print("Running warm-up request...")
try:
    response = requests.post(url, json={"text": test_queries[0]}, timeout=10)
    print(f"✓ Warm-up complete (Status: {response.status_code})\n")
except Exception as e:
    print(f"✗ Warm-up failed: {e}")
    exit(1)

print(f"Measuring latency across {len(test_queries)} different queries, 20 iterations each...")
print(f"Total requests: {len(test_queries) * 20}\n")

all_latencies = []
failed = 0

for query_idx, query_text in enumerate(test_queries):
    query_latencies = []
    
    for i in range(20):
        try:
            start = time.perf_counter()
            response = requests.post(url, json={"text": query_text}, timeout=10)
            end = time.perf_counter()
            
            if response.status_code == 200:
                latency_ms = (end - start) * 1000
                query_latencies.append(latency_ms)
                all_latencies.append(latency_ms)
            else:
                failed += 1
        except Exception as e:
            failed += 1
    
    avg = statistics.mean(query_latencies) if query_latencies else 0
    print(f"Query {query_idx + 1}: {len(query_latencies)} successful, avg {round(avg, 2)} ms")

print("\n" + "=" * 60)
print("LOCAL MODEL LATENCY RESULTS")
print("=" * 60)

if all_latencies:
    all_latencies.sort()
    
    print(f"Total successful requests: {len(all_latencies)}")
    print(f"Failed requests: {failed}")
    print(f"\nLatency Statistics (local inference):")
    print(f"  Average:  {round(statistics.mean(all_latencies), 2)} ms")
    print(f"  Median:   {round(statistics.median(all_latencies), 2)} ms")
    print(f"  Minimum:  {round(min(all_latencies), 2)} ms")
    print(f"  Maximum:  {round(max(all_latencies), 2)} ms")
    
    # Calculate P95
    p95_index = int(0.95 * len(all_latencies))
    p95 = all_latencies[p95_index - 1] if p95_index > 0 else all_latencies[-1]
    print(f"  P95:      {round(p95, 2)} ms")
    
    # Check if under 200ms
    avg_latency = statistics.mean(all_latencies)
    if avg_latency < 200:
        print(f"\n✓ Average latency is under 200ms ({round(avg_latency, 2)} ms)")
    else:
        print(f"\n✗ Average latency exceeds 200ms ({round(avg_latency, 2)} ms)")
    
    print("\nMeasurement methodology:")
    print("  - FastAPI running locally on localhost:8000")
    print("  - DistilBERT model running on CPU")
    print("  - Measured end-to-end HTTP request time including:")
    print("    • Request parsing")
    print("    • Tokenization (DistilBertTokenizerFast)")
    print("    • Model inference (DistilBertForSequenceClassification)")
    print("    • Response formatting")
    print("  - 100 total requests across 5 different query types")
    print("  - Each query tested 20 times to account for variance")
else:
    print("✗ All requests failed. Unable to measure latency.")

print("=" * 60)

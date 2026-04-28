import time
import json
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor

from bundle_paths import DATA_DIR, HOST as VLLM_HOST, RESULTS_DIR, apply_request_model

class SimpleBenchmark:
    def __init__(self):
        self.results = {
            'warmup': [],
            'test': [],
            'e2e_latency_ms': [],
            'ttft_ms': [],
            'tpot_ms': []
        }
    
    def send_request(self, payload_file):
        with open(payload_file) as f:
            payload = json.load(f)
        payload = apply_request_model(payload)
        
        start = time.perf_counter()
        resp = requests.post(f"{VLLM_HOST}/v1/chat/completions", 
                           json=payload, timeout=300)
        end = time.perf_counter()
        
        e2e_ms = (end - start) * 1000
        data = resp.json()
        
        completion_tokens = data['usage']['completion_tokens']
        tpot = e2e_ms / completion_tokens if completion_tokens > 0 else 0
        
        return {
            'e2e_ms': e2e_ms,
            'ttft_ms': e2e_ms,
            'tpot_ms': tpot,
            'prompt_tokens': data['usage']['prompt_tokens'],
            'completion_tokens': completion_tokens
        }
    
    def run_benchmark(self):
        print("=== Simple Performance Benchmark ===\n")
        
        # Warmup rounds (0-2)
        print("Warmup phase (3 rounds)...")
        for i in range(3):
            print(f"  Round {i}...", end=' ', flush=True)
            result = self.send_request(DATA_DIR / f"round_{i}" / "payload.json")
            self.results['warmup'].append(result)
            print(f"E2E: {result['e2e_ms']:.1f}ms")
        
        # Test rounds (3-12)
        print("\nTest phase (10 rounds)...")
        for i in range(3, 13):
            print(f"  Round {i}...", end=' ', flush=True)
            result = self.send_request(DATA_DIR / f"round_{i}" / "payload.json")
            self.results['test'].append(result)
            self.results['e2e_latency_ms'].append(result['e2e_ms'])
            self.results['ttft_ms'].append(result['ttft_ms'])
            self.results['tpot_ms'].append(result['tpot_ms'])
            print(f"E2E: {result['e2e_ms']:.1f}ms")
        
        self._print_summary()
        self._save_results()
    
    def _print_summary(self):
        print("\n=== Results Summary ===")
        print(f"\nE2E Latency (ms):")
        print(f"  Mean: {statistics.mean(self.results['e2e_latency_ms']):.2f}")
        print(f"  P50: {statistics.median(self.results['e2e_latency_ms']):.2f}")
        print(f"  P95: {sorted(self.results['e2e_latency_ms'])[int(len(self.results['e2e_latency_ms'])*0.95)]:.2f}")
        print(f"  P99: {sorted(self.results['e2e_latency_ms'])[int(len(self.results['e2e_latency_ms'])*0.99)]:.2f}")
        
        print(f"\nTTFT (ms):")
        print(f"  Mean: {statistics.mean(self.results['ttft_ms']):.2f}")
        print(f"  P50: {statistics.median(self.results['ttft_ms']):.2f}")
        
        print(f"\nTPOT (ms/token):")
        print(f"  Mean: {statistics.mean(self.results['tpot_ms']):.4f}")
        print(f"  P50: {statistics.median(self.results['tpot_ms']):.4f}")
    
    def _save_results(self):
        output_file = f"{RESULTS_DIR}/benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    benchmark = SimpleBenchmark()
    benchmark.run_benchmark()

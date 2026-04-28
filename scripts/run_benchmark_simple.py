#!/usr/bin/env python3
"""Simplified performance benchmark for vLLM."""

import time
import json
import statistics
import requests
from datetime import datetime

from bundle_paths import DATA_DIR, HOST as VLLM_HOST, RESULTS_DIR, apply_request_model

class SimpleBenchmark:
    def __init__(self):
        self.results = {
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'vllm_host': VLLM_HOST,
                'data_dir': str(DATA_DIR),
            },
            'warmup': [],
            'test': []
        }
        
    def send_request(self, payload_file, round_num):
        """Send request and measure timing."""
        with open(payload_file) as f:
            payload = json.load(f)
        payload = apply_request_model(payload)
        
        # Measure E2E time
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{VLLM_HOST}/v1/chat/completions",
                json=payload,
                timeout=600
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"  ERROR: {e}")
            return None
        end = time.perf_counter()
        
        e2e_ms = (end - start) * 1000
        data = resp.json()
        
        usage = data.get('usage', {})
        completion_tokens = usage.get('completion_tokens', 0)
        prompt_tokens = usage.get('prompt_tokens', 0)
        
        # Calculate TTFT and TPOT
        # TTFT ≈ E2E for first token (simplified)
        ttft_ms = e2e_ms
        tpot_ms = e2e_ms / completion_tokens if completion_tokens > 0 else 0
        
        result = {
            'round': round_num,
            'e2e_ms': e2e_ms,
            'ttft_ms': ttft_ms,
            'tpot_ms': tpot_ms,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': usage.get('total_tokens', 0)
        }
        
        return result
    
    def run_benchmark(self):
        """Run full benchmark."""
        print("=" * 60)
        print("VLLM Performance Benchmark (Simplified)")
        print("=" * 60)
        print(f"\nVLLM Host: {VLLM_HOST}")
        print(f"Data Dir: {DATA_DIR}")
        print()
        
        # Warmup rounds (0-2)
        print("-" * 60)
        print("WARMUP PHASE (3 rounds)")
        print("-" * 60)
        for i in range(3):
            payload_file = DATA_DIR / f"round_{i}" / "payload.json"
            print(f"\nRound {i}: ", end='', flush=True)
            result = self.send_request(payload_file, i)
            if result:
                print(f"E2E={result['e2e_ms']:.1f}ms, "
                      f"TTFT={result['ttft_ms']:.1f}ms, "
                      f"TPOT={result['tpot_ms']:.3f}ms/token")
                self.results['warmup'].append(result)
            else:
                print("FAILED")
        
        # Test rounds (3-12)
        print("\n" + "-" * 60)
        print("TEST PHASE (10 rounds)")
        print("-" * 60)
        for i in range(3, 13):
            payload_file = DATA_DIR / f"round_{i}" / "payload.json"
            print(f"\nRound {i}: ", end='', flush=True)
            result = self.send_request(payload_file, i)
            if result:
                print(f"E2E={result['e2e_ms']:.1f}ms, "
                      f"TTFT={result['ttft_ms']:.1f}ms, "
                      f"TPOT={result['tpot_ms']:.3f}ms/token")
                self.results['test'].append(result)
            else:
                print("FAILED")
        
        self._print_summary()
        self._save_results()
    
    def _print_summary(self):
        """Print results summary."""
        if not self.results['test']:
            print("\nNo test results to summarize.")
            return
        
        e2e_times = [r['e2e_ms'] for r in self.results['test']]
        ttft_times = [r['ttft_ms'] for r in self.results['test']]
        tpot_times = [r['tpot_ms'] for r in self.results['test']]
        
        def calc_stats(times):
            if not times:
                return {'mean': 0, 'p50': 0, 'p95': 0, 'p99': 0, 'min': 0, 'max': 0}
            sorted_times = sorted(times)
            n = len(sorted_times)
            return {
                'mean': statistics.mean(sorted_times),
                'p50': sorted_times[int(n * 0.50)],
                'p95': sorted_times[int(n * 0.95)],
                'p99': sorted_times[int(n * 0.99)],
                'min': min(sorted_times),
                'max': max(sorted_times)
            }
        
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        print("\nE2E Latency (ms):")
        e2e_stats = calc_stats(e2e_times)
        print(f"  Mean: {e2e_stats['mean']:.2f}")
        print(f"  P50:  {e2e_stats['p50']:.2f}")
        print(f"  P95:  {e2e_stats['p95']:.2f}")
        print(f"  P99:  {e2e_stats['p99']:.2f}")
        print(f"  Min:  {e2e_stats['min']:.2f}")
        print(f"  Max:  {e2e_stats['max']:.2f}")
        
        print("\nTTFT (ms):")
        ttft_stats = calc_stats(ttft_times)
        print(f"  Mean: {ttft_stats['mean']:.2f}")
        print(f"  P50:  {ttft_stats['p50']:.2f}")
        print(f"  P95:  {ttft_stats['p95']:.2f}")
        print(f"  P99:  {ttft_stats['p99']:.2f}")
        
        print("\nTPOT (ms/token):")
        tpot_stats = calc_stats(tpot_times)
        print(f"  Mean: {tpot_stats['mean']:.4f}")
        print(f"  P50:  {tpot_stats['p50']:.4f}")
        print(f"  P95:  {tpot_stats['p95']:.4f}")
        print(f"  P99:  {tpot_stats['p99']:.4f}")
    
    def _save_results(self):
        """Save results to file."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        output_file = RESULTS_DIR / 'benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n\nResults saved to: {output_file}")

if __name__ == "__main__":
    benchmark = SimpleBenchmark()
    benchmark.run_benchmark()

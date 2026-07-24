# E1 Results — Generative Weights, Fractal Seed (Rust / candle)

- Task: **Fashion-MNIST** | target 784->512->10 = **407050 params** | epochs: 15 | seeds: 2
- Dense upper bound: **0.8908 ± 0.0029**

| Fractal comp | Seed params | Fractal acc | Plain MLP (same budget) | Fractal − Plain | % of dense | Stored-embed seed would be |
|---|---|---|---|---|---|---|
| 42.6× | 9554 | 0.8521 | 0.8529 (9550) | -0.0008 ➖ | 95.7% | 36039 (11.3×) |
| 43.9× | 9266 | 0.8530 | 0.8489 (8755) | +0.0041 ➖ | 95.8% | 16012 (25.4×) |
| 81.5× | 4994 | 0.8262 | 0.8308 (4780) | -0.0046 ➖ | 92.8% | 8628 (47.2×) |
| 151.3× | 2690 | 0.8202 | 0.7861 (2395) | +0.0342 ✅ | 92.1% | 10299 (39.5×) |
| 254.1× | 1602 | 0.7883 | 0.6956 (1600) | +0.0928 ✅ | 88.5% | 6065 (67.1×) |
| 423.1× | 962 | 0.7879 | 0.3281 (805) | +0.4598 ✅ | 88.5% | 10195 (39.9×) |

**Fractal seed** generates chunk addresses from a fixed sinusoidal index encoding, so its size is ~constant in target size — that is why it reaches high compression where the stored-embedding seed (last column) cannot.

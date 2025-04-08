# Federated Learning Strategies Comparison

| Alpha | Strategy        | Loss   | Accuracy | IoU    | Dice Coefficient | Dice Loss |
| ----- | --------------- | ------ | -------- | ------ | ---------------- | --------- |
| 0.1   | FedAvg          | 0.1296 | 0.9672   | 0.8273 | 0.9045           | 0.1464    |
| 0.1   | MOON            | 0.6959 | 0.3568   | 0.1995 | 0.3312           | 0.6486    |
| 0.1   | MOON Real-World | 0.6718 | 0.7626   | 0.0000 | 0.0000           | 0.6971    |
| 0.5   | FedAvg          | 0.1471 | 0.9407   | 0.8512 | 0.9191           | 0.1478    |
| 0.5   | MOON            | 0.2572 | 0.9089   | 0.7534 | 0.8528           | 0.3296    |
| 0.5   | MOON Real-World | 0.6817 | 0.7308   | 0.0000 | 0.0000           | 0.6639    |
| 1.0   | FedAvg          | 0.1406 | 0.9344   | 0.8310 | 0.9069           | 0.1684    |
| 1.0   | MOON            | 0.4530 | 0.8085   | 0.5870 | 0.7368           | 0.4113    |
| 1.0   | MOON Real-World | 0.6619 | 0.7195   | 0.0000 | 0.0000           | 0.6645    |
| 5.0   | FedAvg          | 0.1948 | 0.9264   | 0.7954 | 0.8848           | 0.2337    |
| 5.0   | MOON            | 0.2050 | 0.9151   | 0.7739 | 0.8715           | 0.2558    |
| 5.0   | MOON Real-World | 0.7000 | 0.2542   | 0.2677 | 0.4182           | 0.6519    |

## Observations and Analysis

### Real-World Implementation Details

**Important Note on MOON Real-World Data Handling:**
In the real-world implementation, data is **not redistributed** among clients as in the standard implementations. Instead, clients with fewer samples (below a threshold) are completely discarded and do not participate in the federated learning process. This more accurately reflects real-world scenarios where:

1. Some clients may not have enough data to meaningfully contribute
2. Data cannot be artificially redistributed between clients due to privacy constraints
3. Participation in federated learning is contingent on having sufficient local data

This client-filtering approach leads to:

- Fewer participating clients at lower alpha values (highly non-IID scenarios)
- Potentially less diverse training data reaching the global model
- More realistic but potentially less optimal performance metrics

## Visual Comparison

### Accuracy Comparison (Higher is better)

```
α=0.1 | FedAvg          | ██████████████████████████████████████████████ 0.9672
      | MOON            | █████████████████ 0.3568
      | MOON Real-World | ███████████████████████████████████ 0.7626

α=0.5 | FedAvg          | █████████████████████████████████████████████ 0.9407
      | MOON            | ████████████████████████████████████████████ 0.9089
      | MOON Real-World | ████████████████████████████████ 0.7308

α=1.0 | FedAvg          | ████████████████████████████████████████████ 0.9344
      | MOON            | ████████████████████████████████████ 0.8085
      | MOON Real-World | ███████████████████████████████ 0.7195

α=5.0 | FedAvg          | ████████████████████████████████████████████ 0.9264
      | MOON            | ████████████████████████████████████████████ 0.9151
      | MOON Real-World | ████████████ 0.2542
```

### IoU Comparison (Higher is better)

```
α=0.1 | FedAvg          | ████████████████████████████████████ 0.8273
      | MOON            | █████████ 0.1995
      | MOON Real-World | - 0.0000 (Clients with insufficient data discarded)

α=0.5 | FedAvg          | █████████████████████████████████████ 0.8512
      | MOON            | ██████████████████████████████████ 0.7534
      | MOON Real-World | - 0.0000 (Clients with insufficient data discarded)

α=1.0 | FedAvg          | ████████████████████████████████████ 0.8310
      | MOON            | ███████████████████████████ 0.5870
      | MOON Real-World | - 0.0000 (Clients with insufficient data discarded)

α=5.0 | FedAvg          | ███████████████████████████████████ 0.7954
      | MOON            | ██████████████████████████████████ 0.7739
      | MOON Real-World | ████████████ 0.2677
```

### Dice Coefficient Comparison (Higher is better)

```
α=0.1 | FedAvg          | █████████████████████████████████████████████ 0.9045
      | MOON            | ████████████████ 0.3312
      | MOON Real-World | - 0.0000 (Clients with insufficient data discarded)

α=0.5 | FedAvg          | █████████████████████████████████████████████ 0.9191
      | MOON            | ██████████████████████████████████████████ 0.8528
      | MOON Real-World | - 0.0000 (Clients with insufficient data discarded)

α=1.0 | FedAvg          | █████████████████████████████████████████████ 0.9069
      | MOON            | ███████████████████████████████████ 0.7368
      | MOON Real-World | - 0.0000 (Clients with insufficient data discarded)

α=5.0 | FedAvg          | ████████████████████████████████████████████ 0.8848
      | MOON            | ███████████████████████████████████████████ 0.8715
      | MOON Real-World | ████████████████████ 0.4182
```

### Loss Comparison (Lower is better)

```
α=0.1 | FedAvg          | ██████ 0.1296
      | MOON            | ████████████████████████████████ 0.6959
      | MOON Real-World | █████████████████████████████ 0.6718

α=0.5 | FedAvg          | ██████ 0.1471
      | MOON            | ████████████ 0.2572
      | MOON Real-World | █████████████████████████████████ 0.6817

α=1.0 | FedAvg          | ██████ 0.1406
      | MOON            | ██████████████████████ 0.4530
      | MOON Real-World | █████████████████████████████████ 0.6619

α=5.0 | FedAvg          | █████████ 0.1948
      | MOON            | █████████ 0.2050
      | MOON Real-World | ██████████████████████████████████ 0.7000
```

### Dice Loss Comparison (Lower is better)

```
α=0.1 | FedAvg          | ███████ 0.1464
      | MOON            | ██████████████████████████████ 0.6486
      | MOON Real-World | ████████████████████████████████ 0.6971

α=0.5 | FedAvg          | ███████ 0.1478
      | MOON            | ████████████████ 0.3296
      | MOON Real-World | ██████████████████████████████ 0.6639

α=1.0 | FedAvg          | ████████ 0.1684
      | MOON            | ████████████████████ 0.4113
      | MOON Real-World | ██████████████████████████████ 0.6645

α=5.0 | FedAvg          | ███████████ 0.2337
      | MOON            | ████████████ 0.2558
      | MOON Real-World | ██████████████████████████████ 0.6519
```

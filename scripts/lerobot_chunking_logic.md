# Action Chunking & Queuing Mechanism (LeRobot Async Inference)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   ROBOT CLIENT                          │
│                                                         │
│  ┌──────────────────┐       ┌────────────────────────┐  │
│  │  Control Loop     │       │  Action Receiver       │  │
│  │  (main thread)    │       │  (daemon thread)       │  │
│  │                   │       │                        │  │
│  │  1. Pop action    │       │  Polls server for      │  │
│  │     from queue    │       │  new action chunks     │  │
│  │  2. Send to robot │       │  and pushes into       │  │
│  │  3. Send obs if   │  ◄──  │  action_queue          │  │
│  │     queue low     │       │                        │  │
│  └──────────────────┘       └────────────────────────┘  │
│         │                              ▲                │
│         │ gRPC: SendObservations        │ gRPC: GetActions│
└─────────┼──────────────────────────────┼────────────────┘
          │                              │
          ▼                              │
┌─────────────────────────────────────────────────────────┐
│                   POLICY SERVER                         │
│                                                         │
│  obs_queue (size=1)  ──►  _predict_action_chunk()       │
│  (keeps only latest)      (model inference)             │
└─────────────────────────────────────────────────────────┘
```

## Timeline: Action Chunking with Overlap

Say `actions_per_chunk = 6`, `chunk_size_threshold = 0.5` (request new inference when queue <= 50% full).

```
Time step:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
            │   │   │   │   │   │   │   │   │   │   │   │   │   │   │

Chunk A:   [A0  A1  A2  A3  A4  A5]
                            ▲
                            │ queue at 50%, new obs sent
                            │
Chunk B:                   [B3  B4  B5  B6  B7  B8]
                                            ▲
                                            │ queue at 50% again
                                            │
Chunk C:                                   [C6  C7  C8  C9 C10 C11]

Executed:   A0  A1  A2  ?3  ?4  ?5  ?6  ?7  ?8  ?9  ...
```

The `?` steps are where **aggregation** happens — two chunks predict actions for the same timestep.

## Action Aggregation at Overlapping Timesteps

| Time step | Chunk A prediction | Chunk B prediction | Executed (with `weighted_average`) |
|-----------|-------------------|-------------------|-----------------------------------|
| 0         | A0                | —                 | A0                                |
| 1         | A1                | —                 | A1                                |
| 2         | A2                | —                 | A2                                |
| 3         | A3                | B3                | avg(A3, B3)                       |
| 4         | A4                | B4                | avg(A4, B4)                       |
| 5         | A5                | B5                | avg(A5, B5)                       |
| 6         | —                 | B6                | B6                                |
| 7         | —                 | B7                | B7                                |
| 8         | —                 | B8                | B8                                |

This blending is what makes motion smooth — without it, you'd get a hard jump at step 3 when switching from chunk A to chunk B.

## Queue State Over Time

```
Queue size
    6 │ ██
    5 │ ██ ██
    4 │ ██ ██ ██
    3 │ ██ ██ ██ ██ ◄── threshold (50%), trigger new inference
    2 │ ██ ██ ██ ██ ██
    1 │ ██ ██ ██ ██ ██ ██
    0 │ ██ ██ ██ ██ ██ ██ ··· (waiting for chunk B to arrive)
      └──────────────────────
        t0 t1 t2 t3 t4 t5 t6

      │◄── executing chunk A ──►│◄── chunk B arrives, queue refills ──►│
```

## Key Config Parameters

| Parameter | Where | Purpose |
|-----------|-------|---------|
| `actions_per_chunk` | client config | How many actions from model output to use |
| `chunk_size_threshold` | client (`robot_client.py:403`) | Queue fraction that triggers new obs send (0.5 = send when half consumed) |
| `environment_dt` | client (`robot_client.py:476`) | Time between each action execution tick |
| `inference_latency` | server (`policy_server.py:236`) | Minimum time before returning actions |
| `aggregate_fn` | client (`robot_client.py:331`) | How overlapping actions merge (default: take latest) |
| `must_go` | client (`robot_client.py:426`) | Forces observation through when queue is empty (prevents stall) |

## Compare: Isaac-GR00T vs LeRobot

| | Isaac-GR00T (SO-100 example) | LeRobot async server |
|---|---|---|
| Threading | Single thread | 2 threads (control + action receiver) |
| Action queue | None | Yes, with aggregation |
| Temporal blending | No | Yes (`weighted_average`) |
| Overlap between chunks | No, sequential | Yes, controlled by `chunk_size_threshold` |
| Timing | `time.sleep(0.02)` hardcoded | `environment_dt` configurable |
| Obs sent | Every chunk boundary | When queue drains below threshold |

## Comparison (THE important table)

| Model | Eval set | P % | R % | mAP50 % |
| ------- | ------- | ------- | ------- | ------- |
| Generic | Own val (construction) | 58.1 | 64.4 | 59.1 |
| Generic | Industrial proxy | 1.5 | 2.2 | 0.4 |
| Generic | OCP site | 29.3 | 26.2 | 18.5 |
| Ours | Fused val | 75.6 | 64.6 | 48.6 |
| Ours | Industrial proxy | 73.5 | 69.2 | 49.6 |
| Ours | OCP site | 84.9 | 87.4 | 65.5 |

Shared classes scored for both models: helmet, vest, gloves, no_helmet, no_gloves (conf=0.25, IoU=0.5; P/R at this fixed confidence, same harness for both models).

## Per-class (Ours, on val)

| Class | P % | R % | mAP50 % | # images |
| ------- | ------- | ------- | ------- | ------- |
| Person | 86.8 | 86.3 | 90.2 | 379 |
| helmet | 75.6 | 53.6 | 59.8 | 137 |
| vest | 87.6 | 81.0 | 85.4 | 306 |
| gloves | 60.3 | 30.6 | 33.8 | 63 |
| boots | 81.8 | 64.2 | 73.6 | 289 |
| goggles | 86.6 | 84.0 | 86.2 | 218 |
| no_helmet | 68.8 | 54.8 | 57.0 | 12 |
| no_gloves | 66.0 | 38.3 | 45.1 | 33 |
| no_goggle | 0.0 | 0.0 | 0.0 | 0 |
| gas mask | 88.5 | 79.1 | 86.1 | 65 |
| ALL | 78.0 | 63.5 | 68.6 | 452 |

## Runtime (same GPU as training)

- preprocess: 2.12 ms
- inference: 4.99 ms
- postprocess: 2.24 ms
- total: 9.35 ms / frame
- ~21 cameras at 5 FPS each
- formula: cameras = floor((1000 / total_ms_per_frame) / fps)

## Training

- epochs: 50
- ultralytics version: 8.4.95
- GPU: Tesla T4
- split used for headline numbers: val

# Handoff README

Answers to the 5 required questions:

1. **Gas-mask dataset:** confirmed — Roboflow
   https://universe.roboflow.com/daniil-yarmov/gas-masks/dataset/1 (v1),
   361 images, downloaded and verified by the pipeline.
2. **Headline numbers from val or test?** val
   (comparison-table rows state their eval set explicitly).
3. **Epochs + GPU:** 50 epochs on Tesla T4.
4. **Classes with almost no images:** no_goggle
   (threshold: <200 train boxes; exact counts in split_summary.json).
5. **Alert system:** designed only — not built.

Notes:

- Model B (generic) is the hosted Roboflow model `ppe_detection-dnfen/1`;
  it was never trained by us, so there is **no generic.pt** in this handoff.
  Its rows in metrics.md come from the same shared evaluation harness as
  Model A; its published own-val numbers are copied as-is.
- Step 7 (LLM judge) was skipped in this run — out of scope for this handoff.

"""The released DVS Gesture classifier and its evaluator, for `downstream_gesture_frozen`.

`downstream_gesture_frozen` scores filtered DVS Gesture clips with the SEW 7B-Net of Fang et
al. (NeurIPS 2021) at the authors' released checkpoint, held frozen. The evaluator runs in an
environment with torch and SpikingJelly (`SNN_PYTHON` in that module), so the rest of the
package imports nothing from here except `frames`, which needs numpy alone.

* `snn7b` -- the 7B-Net in current SpikingJelly, with the map from the released state dict.
* `train_dvsgesture` -- the training loop the network was checked with; the evaluator uses
  its `evaluate` and `seed_everything`.
* `evaluate_dvsgesture_variants` -- scores every condition of a frame manifest with the one
  frozen network and bootstraps paired accuracy differences over held-out subjects.
* `frames` -- the released count-bin integrator, `events_to_number_frames`.

These produced `results/downstream_gesture_frozen.json` and
`results/downstream_gesture_frozen_evaluation.json`. The first three are copied unchanged, and
`tests/test_sew7b.py` pins their bytes; `frames` holds `events_to_number_frames`, unchanged,
taken from a larger module. The checkpoint is third-party and is not redistributed; DATA.md
says where to put it.
"""

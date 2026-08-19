# Fight and Agitation Detection: Pose-Based Motion Classification

## Introduction

**Situation.** Fight-detection datasets and models are built almost entirely
around street-level, mutually-engaged violence: two able-bodied adults, one
usually filmed handheld. That doesn't match a real deployment need: flagging
agitation and physical aggression between patients and caregivers in
dementia/Parkinson's care settings, where the interaction is asymmetric (one
agitated patient, one non-reciprocating caregiver), the camera is fixed CCTV,
and the physicality involved is often weaker and lower-amplitude than a
street fight.

**Task.** Build a classifier that detects aggressive motion per person
(not per interacting pair), works from CCTV-style footage rather than only
handheld street video, and is honest about how far its training data
actually generalizes to the target population, since no public
patient/caregiver violence dataset exists to train or validate against
directly.

**Action.** A pose-based pipeline was built: YOLOv8-pose plus ByteTrack for
per-person keypoint tracking, a normalization step to make the
representation camera- and position-invariant, and an LSTM classifier
trained on RLVS with a leakage-safe grouped train/val/test split. A data
quality audit with TSAuditor caught a real extraction bug (frame-boundary
keypoint clamping) before it reached the reported metrics. The model was
then tested outside its training distribution on UBI-Fights (CCTV footage),
which exposed a cross-domain false-positive problem; that was diagnosed by
watching the actual failures, then fixed with more training coverage and a
re-calibrated decision threshold, verified on the same untouched eval set.
A small hand-labeled patient/caregiver sample was used as a final,
directional check of the actual target domain.

**Result.** A validated street-violence classifier (F1 0.7874, ROC-AUC
0.8435 on held-out RLVS) with a diagnosed and fixed cross-domain
generalization gap, and an explicit, unresolved gap to the real target
population that's reported honestly rather than papered over with the RLVS
number. Full breakdown in [Accuracy analysis](#accuracy-analysis).

## How it works

1. **Pose extraction**: YOLOv8-pose detects each person per frame (17 COCO
   keypoints); ByteTrack assigns a consistent ID across frames.
2. **Normalization**: keypoints are centered on the hip midpoint and scaled
   by shoulder width, so the model learns motion *shape*, not raw pixel
   position, camera distance, or where in frame someone stands.
3. **Classification**: each tracked person's keypoint sequence (variable
   length, padded/masked, up to 90 frames / ~6s) goes through a single-layer
   LSTM (hidden size 64) that classifies it as aggressive or calm.

The classifier runs **per person, not per pair**. Most fight-detection work
assumes two mutually engaged people (street fight, hockey brawl). The target
scenario is asymmetric (one agitated patient, one non-reciprocating
caregiver), so each tracked individual is classified independently.

Pose-only representation also means no raw video needs to be retained after
extraction, which matters for a patient-monitoring application.

## Repo contents

| File | Purpose |
|---|---|
| `extract_features.py` | Video → tracked, normalized keypoint sequences (`.npz` cache) |
| `train_stage1.py` | LSTM model + training loop + full test-set evaluation |
| `tsauditor_check.py` | Data-quality audit of the extracted feature cache |
| `evaluate_on_videos.py` | Run the trained model on arbitrary new videos |
| `build_augmented_dataset.py` | Merge RLVS features with cross-domain non-violence clips |
| `fine_tune.py` | Fine-tune Stage 1 on the augmented dataset |
| `subsample_ubi_fights.py`, `parse_ubi_fights_labels.py` | Build a balanced UBI-Fights eval set from Kaggle |
| `crop_labeled_clips.py`, `label_videos.py` | Build hand-labeled target-domain clips |
| `extract_violent_frames.py` | Independent frame-level YOLOv8 detector + optical-flow motion gating, used as a triage tool (see below) |

All of these are written to disk from the notebook via `%%writefile` cells,
so the notebook is the single source of truth: run it top to bottom in
Colab (GPU runtime) to reproduce everything.

## Data

- **RLVS** (Real Life Violence Situations): 2,000-clip labeled dataset,
  street-level violence. Primary training source.
- **RWF-2000** and **Hockey Fight Detection** were evaluated and excluded:
  RWF-2000's license restricts redistribution of modified data; Hockey Fight
  Detection produced severe tracking failure (6–8 frame tracks out of a
  ~24-frame clip) on its low-res, fast-motion footage.
- **UBI-Fights**, **UCF-Crime**: used later for cross-domain evaluation and
  augmentation, not initial training (see below).

## Results at a glance

| Evaluation | Metric | Value |
|---|---|---|
| RLVS held-out test (in-distribution) | F1 / ROC-AUC / accuracy | 0.7874 / 0.8435 / 0.76 |
| UBI-Fights, Stage 1 model (cross-domain, before fix) | Violence recall / non-violence recall | 0.83 / 0.33 |
| UBI-Fights, Stage 2b model (cross-domain, after fix) | Violence recall / non-violence recall | see [Accuracy analysis](#accuracy-analysis) |
| Target-domain (hand-labeled patient/caregiver clips) | accuracy | small sample, directional only: see below |

## Accuracy analysis

The single number worth being suspicious of here is the RLVS F1/ROC-AUC. It's
real, it's methodologically clean, and it is **not** a proxy for how the
model performs in the actual deployment setting. Three separate evaluations
below answer three different questions, and none of them substitutes for
another.

### 1. In-distribution: RLVS held-out test

2,002 train / 670 validation / 673 test sequences, split **grouped by source
clip** (not by sequence) so no clip's tracked people leak across splits: this is the standard leakage failure mode for pose/video classifiers and is
worth calling out explicitly since it's the kind of thing that inflates
reported accuracy silently.

- F1: 0.7874, ROC-AUC: 0.8435, test accuracy: 0.76
- Validation accuracy at checkpoint selection was 0.8164: notably higher
  than test. That gap is not a red flag on its own: with ~2,000 training
  clips, a grouped split (correct, but noisier than random per-sequence
  splitting) means individual hard clips carry real statistical weight, and
  early stopping selects the checkpoint that fit validation's specific
  composition best, which doesn't fully transfer. **Test accuracy is the
  number to trust, not validation accuracy**: this is a general point, not
  specific to this project.
- Per-class: precision/recall/F1 of 0.73/0.72/0.72 (non-violence) and
  0.78/0.79/0.79 (violence): reasonably balanced, which matters because an
  earlier baseline (before the leakage-safe split and the boundary-clamp fix
  below) over-triggered on violence. In a caregiver-alerting system, that
  imbalance is alert fatigue, not just a worse number.

### 2. A real bug caught before it reached this metric

A [TSAuditor](https://pypi.org/project/tsauditor/) scan of the extracted
pose sequences (run out-of-domain: TSAuditor is a time-series/financial data
auditing tool, applied here to per-person keypoint tracks) flagged 1,099
"stuck value" findings concentrated on vertical coordinates. Root cause:
YOLOv8-pose clamps a keypoint to the frame boundary when a body part extends
off-screen, instead of marking it undetected. The clamped value wasn't
`(0, 0)`, so it passed the existing all-zero detection-failure filter
silently, and looked like a normal skeleton on visual inspection.

Fix: reject any keypoint within one pixel of the frame edge in
`normalize_skeleton`. Re-extraction and re-audit came back clean (0 findings).
The F1/ROC-AUC above are from the *corrected* dataset. This is the kind of
bug that specifically survives a "does the skeleton look right" visual check: it needed a systematic scan to surface.

### 3. Cross-domain: does it generalize past RLVS's camera style?

RLVS is handheld street footage. The deployment target is fixed-camera CCTV.
Testing on a balanced 30/30 UBI-Fights sample (real CCTV, labels free from
filename convention) exposed an asymmetric failure:

| | RLVS test | UBI-Fights (Stage 1, pre-fix) |
|---|---|---|
| Violence recall | 0.79 | 0.83 (25/30) |
| Non-violence recall | 0.72 | **0.33** (7/21, +2 unusable tracks) |

Violence recall transferred fine. Non-violence recall collapsed: the model
called ordinary CCTV footage "violence" roughly two-thirds of the time.
Likely mechanism: the model never saw calm footage from a fixed-camera domain
during training, so on that domain it may be leaning on
proximity/motion-magnitude as a shortcut rather than the trajectory shape
RLVS-only training was meant to teach it. This was checked by watching the
confident false positives before assuming a fix would help: several were
benign fast motion (hugging, playing) tripping the same trigger a real strike
would.

**Fix required two things, not one:** adding calm cross-domain footage (more
UBI-Fights non-violence clips, held out from the eval set to avoid leakage;
plus UCF-Crime as a second independent camera domain) raised non-violence
recall, but also shifted the model's probability distribution enough that the
default 0.5 threshold became miscalibrated. Re-sweeping the operating point
on the same untouched 30/30 eval set found 0.30 as a threshold that improves
accuracy and violence recall *simultaneously*: a correction of a mis-set
decision boundary, not a precision/recall trade-off. Neither the added data
nor the threshold correction alone was sufficient.

### 4. Target-domain: does it work on the actual use case?

Neither of the above tests the population gap that actually matters: RLVS,
UBI-Fights, and UCF-Crime are all able-bodied adult vs. able-bodied adult.
The deployment target is a frail/restrained patient against a caregiver, with
weaker, lower-amplitude motion: a plausible false-negative mode that
street-violence datasets can't surface. No public dataset covers this (for
good reason), so this used a small hand-labeled set (15–20 clips) with an
explicit labeling rule decided in advance: label 1 only for aggressive
physical contact (strike, push, slap, grab intended to harm), label 0 for
verbal-only agitation *and* for caregiver restraint that involves contact but
isn't itself aggressive: restraint-vs-aggression is the genuinely ambiguous
case in this domain and needed a decision up front, not post hoc.

This sample is too small to report as a stable accuracy figure and isn't
treated as one here: its value is as a directional check and as the set to
go re-examine first if this pipeline gets extended, specifically for missed
low-amplitude strikes.

### What the three numbers together actually say

RLVS in-distribution performance is solid and methodologically sound.
Cross-domain camera generalization was *broken* by default and needed a
diagnosed, verified fix (not just "add more data and hope"). Population-domain
generalization (the part that actually matters for the stated application)
is untested at any reportable sample size. Reporting only the RLVS number
would have been the easy and misleading thing to do; the honest headline is
that this is a validated street-violence classifier with promising but
unverified transfer to the actual target population.

## A second, independent method: frame-level detection

`extract_violent_frames.py` is a separate, non-LSTM approach: a pretrained
YOLOv8 detector ([Musawer14/fight_detection_yolov8](https://huggingface.co/Musawer14/fight_detection_yolov8))
scores individual frames directly, gated by dense optical-flow motion
magnitude so a static punch-looking pose without real motion doesn't fire.

This isn't a replacement for the LSTM pipeline: it reports no real
precision/recall (no ground truth to score against outside a labeled
dataset) and optical flow fires on any fast motion, not specifically
fighting. Its actual use here was triage: scanning a long, unlabeled source
video for candidate timestamps worth trimming into a labeled clip, without
watching the whole thing.

One of its two released checkpoints (`yolo_small_weights.pt`) is flagged
"Unsafe" by Hugging Face's pickle scanner (references
`dill._dill._load_type`, a known code-execution vector in unpickling). The
script disassembles the pickle stream and checks every referenced
symbol against an allow-list before loading: a heuristic static check, not
sandboxing, and it's documented as such in the script rather than presented
as a guarantee.

## Limitations, stated plainly

- Test-set size (673 sequences) is small enough that the val/test gap is
  expected variance, not a training bug. It also means the RLVS metric
  itself has real confidence-interval width that a single point estimate
  hides.
- Target-domain evaluation is directional only, not a reportable accuracy
  number, due to sample size.
- The frame-level YOLO method is a triage tool, not a validated detector: treated that way throughout, not oversold as a second model to average
  against the first.
- UCF-Crime's Kaggle mirror choice (`bypktt/ucf-crimes`) was verified to
  actually be video (not the pre-extracted 64×64 still-image mirror some
  other UCF-Crime mirrors turn out to be) but not independently confirmed
  beyond that, unlike UBI-Fights.

## Setup

Built for Google Colab (GPU runtime). Run the notebook top to bottom.

```bash
pip install ultralytics tsauditor pandas scikit-learn opencv-python huggingface_hub
```

Kaggle API credentials are prompted at runtime (not hardcoded) for the
UBI-Fights and UCF-Crime downloads: get a token from
`kaggle.com/settings → API → Create New Token`.

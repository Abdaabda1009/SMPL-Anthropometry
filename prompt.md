# Image edit/enhance prompts — pace suit, 4-view reference set

Purpose: given a photo a user uploads, edit it (not generate from scratch)
into a 4-view reference set — front, back, left profile, right profile —
of that same person wearing the pace suit, for use as input to the
image→3D→`fit_mesh.py` measurement pipeline. See `REPORT.md` / `ACCURACY_REPORT.md`
for why clean, consistent reference images matter to fit accuracy.

Written for an image-*editing* model (img2img / "edit this image", e.g.
Gemini image edit, GPT-image edit, Flux Kontext) that takes an uploaded
photo plus a text instruction — not a text-to-image generator starting
from nothing. The difference matters: an edit model already has the
person's real face, body and proportions in the pixels; the job here is to
change clothing/pose/background/angle *without* touching any of that.

---

## Critical constraint — read before using any prompt below

**Do not alter the uploaded person's body shape, proportions, height, build,
or identity in any way.** Only change: clothing (→ pace suit), pose, camera
angle, background, and lighting. Whatever body the uploaded photo shows is
what gets measured downstream — if the edit model "helpfully" slims,
bulks, or reproportions the body while adding the suit, every measurement
from this pipeline becomes wrong at the source, silently. Every prompt
below repeats this constraint explicitly for that reason — don't trim it
out when adapting them.

If the tool you're using has a strength/denoise parameter for image edits,
keep it low enough that the underlying body geometry is preserved and only
surface details (clothing, background) change.

---

## Shared garment description

Reuse this block verbatim in all four prompts below.

```
Edit the clothing in this photo to a skintight black long-sleeve
full-body compression suit (motion-capture/biometric sensor-suit style),
second-skin fit with no loose or baggy fabric anywhere. Dense pattern of
small light-gray/white dots clustered over the chest, shoulders, upper
back, and glutes, thinning to sparse dots elsewhere; matte black base
fabric with sheer dark-gray mesh side panels along the ribs and hips;
visible contour-following seam lines. Suit covers torso, arms, and legs to
the ankles; hands and feet bare.
```

## Shared pose / scene description

Reuse this block verbatim in all four prompts below. Arms-down with a
small gap from the torso (not pressed against the body) matches what
`fit_mesh.py` calls "arms-down (A-pose)" — it auto-detects this from body
proportions and initializes the fit for it, but an arm pressed flat against
the torso risks the "arm merges with torso" failure mode its own code
comments call out, so keep that gap visible.

```
Adjust the pose to: standing relaxed and upright, arms straight down at
the sides with a small natural gap between the upper arms and the torso,
feet close together, neutral expression, hair pulled back off the
shoulders if applicable. Replace the background with a neutral flat gray
studio backdrop and even soft diffuse lighting with no harsh shadows.
Frame the entire body head to toe with margin at top and bottom. Keep the
result photorealistic — no stylization, no distortion, no change to the
person's actual proportions.
```

---

## 1 — FRONT VIEW (edit this one first; use its output as the reference for 2–4)

```
Using the uploaded photo as the base, do not alter the person's body
shape, proportions, height, build, or identity in any way.

[shared garment description]

[shared pose / scene description]

Camera / view: front view — subject facing directly toward the camera,
camera at chest height, straight-on, no angle.
```

## 2 — BACK VIEW

```
Using the uploaded photo as the base (or the front-view edit's output, if
generating these in sequence for consistency), do not alter the person's
body shape, proportions, height, build, or identity in any way.

[shared garment description]

[shared pose / scene description]

Camera / view: back view — subject facing directly away from the camera
(180° turn from the front view), same standing pose, arms at sides with
the same small gap from the torso, same camera height and distance, so the
back, shoulders, spine line, and glutes are fully visible.
```

## 3 — LEFT PROFILE VIEW

```
Using the uploaded photo as the base (or the front-view edit's output, if
generating these in sequence for consistency), do not alter the person's
body shape, proportions, height, build, or identity in any way.

[shared garment description]

[shared pose / scene description]

Camera / view: left profile view — camera rotated 90° to the subject's
left side, subject's left side facing the camera, strict side profile (not
a 3/4 angle), same standing pose, same camera height and distance.
```

## 4 — RIGHT PROFILE VIEW

```
Using the uploaded photo as the base (or the front-view edit's output, if
generating these in sequence for consistency), do not alter the person's
body shape, proportions, height, build, or identity in any way.

[shared garment description]

[shared pose / scene description]

Camera / view: right profile view — camera rotated 90° to the subject's
right side, subject's right side facing the camera, strict side profile
(not a 3/4 angle), same standing pose, same camera height and distance.
```

---

## Notes

- **Sequencing for consistency:** edit the front view first, confirm it kept
  the body unchanged, then feed *that* output (not the raw upload) into the
  back/left/right edits if your tool supports chaining — this keeps the
  suit's exact seam/dot layout consistent across views, the same way
  chaining from a single reference did for the from-scratch generation
  version of these prompts.
- **If no photo is uploaded** (synthetic test case instead of a real
  person): fall back to describing a target body in words instead of
  "using the uploaded photo as the base." `subject_1.txt` in this repo is
  a worked example — see the conversation this file came from, or ask for
  it to be regenerated, for the from-scratch (non-edit) version of these
  same four prompts built from that file's anthropometric fields.
- **Next step once you have all four images:** run them through your
  image→3D tool (Meshy AI / Tripo3D), then measure the resulting `.glb`:
  ```
  python fit_mesh.py --input glb/<name>.glb --model_type smplx \
      --gender <MALE|FEMALE> --height <cm> --save_measurements out.csv
  ```
  Fill `--gender`/`--height` from what the uploaded photo's subject
  actually is, not from `subject_1.txt`, unless you're deliberately
  re-running the synthetic test case.

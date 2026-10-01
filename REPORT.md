
claude --resume ba7a737c-778f-458f-b32d-7aa276d7492d


# Optimization & Size-Reduction Report

Static review only. No scripts were run and no code was changed. Line numbers refer to the current working tree. Time estimates are reasoned from the code, not measured.

---

## 1. Why the "model cache" doesn't help repeated requests on the same GLB

The body-model cache works, but it only avoids reloading the `.pkl` files:

- `measure.py:58` — `@lru_cache` on `_create_model_cached` saves loading the ~500 MB SMPLX pkl.
- `fit_mesh.py:100` — `build_model` shares that cache.

It does **not** cache the expensive part, which is the fit. Every `POST /measure/mesh` (`api.py:186-236`) does all of this again:

1. Parse the GLB and run `fix_normals` (`fit_mesh.py:62-67`).
2. Run about 1,110 Adam iterations (80×2 init, 200 rigid, 300 pose, 500 shape, 250 polish). Each iteration does a full `cdist` and a backward pass.
3. Run the measurement pass.

The fit is seeded (`torch.Generator().manu al_seed(0)` plus a seeded `np.random`), so the same GLB, gender and model type gives identical betas every time. That is why repeated tests take the same time: the work is deterministic and is thrown away after each job.

Three smaller issues make the model cache look like it isn't working:

| Issue | Where | Effect |
|---|---|---|
| `lru_cache` doesn't block concurrent misses | `measure.py:58` | Two simultaneous first requests for the same gender each load the pkl, so you get two copies (~2× RAM) and both are slow. |
| `PRELOAD_GENDERS` is empty by default | `api.py:56` | The first request for each of MALE/FEMALE/NEUTRAL pays the pkl load. Testing different genders looks like a cache miss every time. |
| `_load_topology` builds a throwaway `smplx.SMPLX(...)` just to read `.faces` | `measure.py:79` | It loads another ~500 MB pkl that is never reused, because `create_model` loads its own copy. |

### Recommended fix: a result cache (biggest win)

Key it on `sha256(file bytes) + model_type + gender + n_points + FIT_VERSION`.

- **Store:** betas (10 floats), `final_chamfer`, `tgt_h`, and native `measurements` and `labeled`. This is a few KB per entry.
- **Height:** don't include `height_cm` in the key. It is a pure rescale (`measure.py:257-266`), so compute `height_normalized` on cache hit.
- **Hashing:** compute the hash while streaming the upload in `measure_mesh` (`api.py:258`), at no extra I/O cost.
- **Concurrent duplicates:** keep an in-flight `dict[key, Future]` so two identical simultaneous uploads share one fit.
- **Bounds:** use an LRU of about 256 entries. Optionally persist as JSON under a cache dir so restarts keep hits.
- **Overlays:** `overlay=true` needs `posed_verts` and `target_pts`, so store the overlay path in the entry. On a hit with no overlay file, either re-fit or skip the overlay.
- **Result:** a repeat of the same GLB should return in about 1 s or less instead of a multi-minute fit. `/measure/shape` costs about 1 s.

### Also fix
- Set `PRELOAD_GENDERS=MALE,FEMALE,NEUTRAL` in `run_api.sh` if RAM allows, or wrap `_create_model_cached` in a per-key lock.
- Make `_load_topology` use `create_model(...).faces` so no extra model is loaded. This drops ~500 MB of transient RAM and 1–3 s of startup.
- `run_api.sh` doesn't set `OMP_NUM_THREADS`, but the Dockerfile sets 2. With `MAX_WORKERS=2` fits and no thread cap, torch oversubscribes the cores. Export `OMP_NUM_THREADS`/`MKL_NUM_THREADS` there too.

---

## 2. Runtime optimizations (code stays, gets faster or lighter)

| # | Where | Change | Expected gain |
|---|---|---|---|
| 1 | `api.py:_run_fit` | Result cache plus in-flight dedupe (section 1). | Repeats go from minutes to about 1 s. |
| 2 | `fit_mesh.py:255` | `chamfer_distance(posed_t, target)` builds a full 10475×12000 `cdist`, about 500 MB. Compute it in chunks of about 2000 rows, or under `torch.no_grad()` on a 6000-point subsample. | Removes a ~500 MB peak per job. With 2 workers this can be the OOM trigger. |
| 3 | `fit_mesh.py:239` | The polish stage uses `ti = rand_idx(n_tgt, n_tgt)`, all 12,000 points, so the distance matrix is 8000×12000 for 250 iterations with autograd. Cap the target at about 6000–8000. | About 30–40% faster polish and a smaller autograd footprint. Check chamfer stays within tolerance on your test GLBs. |
| 4 | `fit_mesh.py:62` | `trimesh.load` also loads textures and materials. Pass `skip_materials=True` (check that trimesh 3.15.1 accepts it for GLB) or load with `force="mesh"`. | Faster and lighter load on large GLBs. |
| 5 | `measure.py:29` `set_shape` | The forward pass runs with autograd on. Wrap it in `torch.no_grad()`. | Small saving of memory and time per `/measure/*` call. |
| 6 | `measure.py:368-372, 441-444` | `SMPLXMeasurementDefinitions()` is instantiated 4× per `MeasureBody()`. Instantiate it once and cache it, or make the definitions module-level constants. | Minor per-request saving. |
| 7 | `measure.py:121-127` | The `pass` statements in `measure()` are no-ops where `continue` was intended. An unknown name raises `KeyError`, and already-measured names are re-measured. Use `continue`. | Correctness fix, and it stops repeat work in `label_measurements`. |
| 8 | `fit_mesh.py:load_target_points` | It returns the full trimesh, but the API discards it. Return it only for the CLI overlay path. | Lets the GLB mesh be freed sooner. |

---

## 3. DELETABLE: dead or broken code (safe to remove)

| File | Lines | Why | Approx. lines saved |
|---|---|---|---|
| `evaluate.py` | `if __name__` block, 27–55 | Calls `MeasureSMPL(smpl_path=...)` and `from_smpl`. Neither exists, so it is broken. It also uses a hard-coded `/SMPL-Anthropometry/data/SMPL` path. | ~29 |
| `measure.py` | `__main__` block, 505–536 | Duplicates `test.py`, and `pprint`/`argparse` are only imported for it. | ~32 |
| `measure.py` | `Measurer.from_verts` / `from_body_model` stubs, 106–110 | Empty `pass` bodies, overridden in both subclasses. | ~5 |
| `measure.py` | Commented-out code, 306–310 | Dead comments. | ~5 |
| `measure.py` | `num_thetas` parameter (`create_model`, `from_body_model`), plus `num_thetas=self.num_joints` at 411 and 485 | The docstring at line 46 says it does nothing. | ~4 |
| `utils.py` | `point_segmentation_to_face_segmentation` and `__main__`, 111–172 | One-off tooling that generated the JSON files already in `data/`. It pulls in `tqdm` and `Counter`. Move it to `tools/` or delete it. | ~62 |
| `visualize.py` | `viz_*` functions and the trailing scripts, 379–897 | Nothing imports them. The only external use is `Visualizer` (`measure.py:320`). They are debug plots for joints, segmentation and landmarks. | ~520 |
| `visualize.py` | Unused imports: `plotly` (partly), `make_subplots`, `argparse`, `smplx`, `json`, `torch`. Remove the ones left unused after the `viz_*` functions go. | Only `go`, `px`, `trimesh`, `np` and the definitions are used by `Visualizer`. | ~6 |
| `docker/` (Dockerfile, build.sh, run.sh, requirements.txt) | all | Legacy dev image. It is superseded by the root `Dockerfile` and `requirements.txt`, and it pins `scikit-learn`/`tqdm` that nothing in the API uses. | 4 files, ~24 lines |
| `env_depoly.yml` | all | The name is misspelled. It is a raw `conda env export` with macOS arm64 build strings, so it can't be recreated elsewhere. Replace it with `requirements.txt` or a 10-line `environment.yml`. | 76 |
| `mesh_measurements.csv` | all | A generated output of `--save_measurements`, not source. Untrack it and gitignore it. | 18 |
| `.DS_Store` | tracked in git | Untrack it with `git rm --cached .DS_Store`. |  |

## 4. REMOVABLE: duplication that can be merged

| File | Lines | Change | Approx. lines saved |
|---|---|---|---|
| `measure.py` | `MeasureSMPL` and `MeasureSMPLX`, 345–491 | The classes are ~95% identical. `from_verts` and `from_body_model` are copy-pasted, and only the constants differ (model type, landmarks, definitions, joints, vertex count). Make one `Measurer` class configured by a small per-model config dict or dataclass, with `MeasureBody` as a factory. Keep the public names as thin aliases if callers depend on them. | ~60 |
| `test.py` | 1–146 | A demo, not a test. Trim the boilerplate, or move it to `examples/`. It imports `pandas`, which is not in `requirements.txt`. | ~40 (optional) |
| `fit_mesh.py` | `main()` overlay block, 297–310, and `api.py:_write_overlay`, 163–183 | Two near-identical plotly overlay writers. Share one function in a small module (for example `overlay.py`) with an option for the target trace. | ~15 |
| Docs (`README.md`, `system.md`, `Project_Overview.md`, `measurement_engine.md`, `PERFORMANCE_REPORT.md`, `ACCURACY_REPORT.md`, `PRODUCTION_LXC.md`) | ~94 KB, 7 files | I did not audit these line by line. With three architecture/overview docs (`system.md`, `Project_Overview.md`, `measurement_engine.md`), I'd expect overlap. Consolidate into README plus one reference doc. They are excluded from the image by `.dockerignore`, so this is repo hygiene, not runtime. | — |

## 5. Non-code size (working tree and repo)

| Item | Size | Action |
|---|---|---|
| `fit_overlay_mesh.html` | 32 MB | Delete locally. It is generated and already gitignored. |
| `fit_overlay.html` | 11 MB | Same. |
| `glb/` | 351 MB | Test inputs, gitignored. Keep only the ones you need. |
| `.git` | 260 MB | Very large for ~1 MB of tracked files, so large blobs were probably committed in the past. Check with `git rev-list --objects --all` piped through `git cat-file --batch-check`, then rewrite history if needed. |
| `.gitignore` | line 7 | `...ipynb.DS_Store` is one line with no newline, so neither the notebook nor `.DS_Store` is ignored. Split it into two lines. |
| `__pycache__/` | 104 KB | Already ignored. |

## 6. Estimated impact

- **Python source:** about 3,040 lines now. Removing the deletable code (~660) and merging the duplicates (~75) takes out roughly **24%**, with `visualize.py` about 58% smaller (897 → ~380).
- **Other files:** `docker/` (4 files), `env_depoly.yml` and `mesh_measurements.csv` go away, and roughly 43 MB of generated HTML comes off disk.
- **Runtime:**
  - The result cache turns repeat GLBs from a full fit into a lookup.
  - The chamfer and polish caps cut peak RAM by about 0.5 GB per job.
  - Fixing `_load_topology` drops one more ~500 MB pkl load.

## 7. Suggested order

1. Result cache plus in-flight dedupe, which fixes the problem you reported.
2. `_load_topology` via `create_model`, and `PRELOAD_GENDERS` / thread caps in `run_api.sh`.
3. Chunked final chamfer and the capped polish target. Re-run your GLBs and compare chamfer % and measurements before and after.
4. Delete the dead code (section 3) and merge the classes (section 4).
5. Repo hygiene: `.gitignore`, `git rm --cached`, `.git` history size.

Nothing above has been applied. Tell me which sections to implement.

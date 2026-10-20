# Checks run for this package

Source: GitHub commit `bf05db8c2ccd4fd95cc59ad899155e0ce889f1d5` plus the detector and interface updates from this conversation. No changes were pushed to GitHub.

## Passed

- Ten Python tests using Flask 3.1.2, scikit-learn 1.7.2, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3 and joblib 1.5.2.
- Real TF-IDF/logistic-regression fitting on temporary synthetic fixtures, grouped splitting, evaluation, saving, reloading and sentence scoring.
- Public CSV preparation, label mapping, balanced sampling, duplicate handling and overwrite refusal.
- Flask page/assets, rewrite response with attached detection, detection routes, malformed input, cross-origin rejection and request-size errors.
- Prompt assembly, actual six-row training split, matching-input exclusion, validation and busy locking.
- Qwen loading/generation control flow with mocked Torch/Transformers, including full weights, 4-bit selection, missing CUDA, low VRAM, reuse and lock cleanup.
- Browser logic with jsdom: rewrite display, word count, inline highlight toggle, edit invalidation, copy, JSON export with current edited detection, light/dark toggle, clearing, late-result rejection and request failures.
- Python compilation, JavaScript syntax, existing notebook JSON structure and personal CSV/split checks.

Synthetic fixtures were stored outside the package. No fake trained detector is shipped, and fixture scores are not performance claims.

## Not verified here

This environment has no NVIDIA CUDA device, no Windows command shell and no installed browser executable. Real Qwen inference, GPU speed/memory, Windows launchers and pixel-level browser rendering must be checked on the PC. A browser download was attempted but failed. Browser logic was tested in jsdom instead.

The original trained detector was not available here. Its archived report is included as a historical result only. A newly trained detector may produce different scores.

## Folder cleanup verification

After moving modules into src and training tools into scripts, the Python and UI checks were rerun. Shared paths still point to the root data, models and web folders. Migration checks cover backups, keeping user assets, missing update files and a failed apply.

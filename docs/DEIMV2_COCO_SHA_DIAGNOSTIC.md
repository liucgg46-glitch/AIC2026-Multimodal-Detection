# DEIMv2 COCO annotation SHA diagnostic

GPU-B at AIC commit `60eb3e4f029cd95e7f6e769e9a0fd7de3e245227` produced the expected 1600/400 images, 12153/3041 annotations and exact `labels_clean` aggregate SHA, but its two annotation JSON SHA values differed from the local reference. Do not waive the gate or run DEIMv2 smoke until the cause is identified.

The local exporter was rerun at that same commit with Python 3.14.2 and OpenCV 4.13.0. It produced the exact prior local SHA values again: train `ecb518c6bc4b78d86f7e7d33edceefea186e313cb129b43344ac995ebe0d5c85`, val `159b1b92b86276202b1f77a1c947514a682f9a1c2a62ee8356a55bc4ad8669a4`. The prior reference is therefore reproducible locally.

The local identity audit uses fixed split order and hashes the literal UTF-8 lines `file_name\0width\0height\n` for dimensions, and `stem\0file_name\0byte_size\0file_sha256\n` for original image identity:

| Subset | Dimension fingerprint | Original visible aggregate SHA256 |
| --- | --- | --- |
| train | `9cf51f5950f2842ced67bf8ec8964a3a98fb3a8ae34c5e47295fd46953646089` | `a30d51404d21017766434fbfc03b07e89101d4121a61ec3239b1b70de51e3626` |
| val | `189c51a02d3c2adfd414cfe1a0df80fcf74db1e47d5d76f6ace244f47063415e` | `92b505f0391ba3f084b9c677ba1907f1a2ed26af9cec6c1e85f0189023718a0a` |

After pushing the diagnostic commit containing this script, run the identity audit in GPU-B's project checkout, using its existing exported view. This is read-only; do not start formal training from a changed experiment SHA without freezing it again:

```bash
cd /root/data1/AIC2026/AIC2026-Multimodal-Detection-B
python scripts/data/audit_deimv2_coco_identity.py \
  --dataset-root data/processed/deimv2_rgb_coco \
  > /root/data1/AIC2026/deimv2_coco_identity_B.json
```

The report gives Python/OpenCV versions, a precisely specified dimension fingerprint, original visible image aggregate SHA, first three source image SHA values, annotation SHA, and counts. Compare the fingerprints to the local report in the response to the user. The previously supplied server fingerprints cannot be compared until their serialization formula is known: hashing the same dimensions with different separators or field order gives unrelated SHA values.

For the first **actual JSON semantic diff**, copy GPU-B's `instances_train.json` and `instances_val.json` into `artifacts/DEIMV2_COCO_B_AUDIT/annotations/` locally, then run:

```powershell
python scripts/data/audit_deimv2_coco_identity.py `
  --dataset-root tmp/deimv2_coco_recheck_60eb3e4 `
  --compare-root artifacts/DEIMV2_COCO_B_AUDIT
```

Interpretation:

- If the original visible aggregate SHA differs, compare the first mismatched per-file SHA before changing the exporter.
- If the original visible aggregate and dimension fingerprints match but COCO numeric fields differ, inspect the first reported `$.annotations[...]` path for a cross-version arithmetic or serialization issue.
- If semantics are identical but JSON bytes differ, inspect encoding/serialization and then rework the hash contract around canonical JSON.
- If dimensions differ, inspect EXIF orientation and OpenCV decoding; image count and portrait count alone do not establish dimension identity.

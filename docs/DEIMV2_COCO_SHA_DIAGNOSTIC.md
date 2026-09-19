# DEIMv2 COCO annotation SHA diagnostic and resolution

GPU-B at AIC commit `60eb3e4f029cd95e7f6e769e9a0fd7de3e245227` produced the expected 1600/400 images, 12153/3041 annotations and exact `labels_clean` aggregate SHA, but its two annotation JSON SHA values differed from the local reference. The cause is now established: `Path.write_text` converted the terminal `\n` to `\r\n` on Windows. The original exporter wrote text in platform-native newline mode. It did **not** produce a JSON semantic difference.

The local exporter was rerun at that same commit with Python 3.14.2 and OpenCV 4.13.0. It produced the exact prior local SHA values again: train `ecb518c6bc4b78d86f7e7d33edceefea186e313cb129b43344ac995ebe0d5c85`, val `159b1b92b86276202b1f77a1c947514a682f9a1c2a62ee8356a55bc4ad8669a4`. The GPU-B Python 3.8.10/OpenCV 5.0.0 files each contained one byte less: train `7a0c369ee0c4ca0a15858436904fadcbf6fb105741c43611498ed9c2d5f88536`, val `d33c458bc9d09cd40f609c12e8548a96757cedcb88609f53d503831a32436a6a`.

Direct recursive JSON comparison found `first_semantic_diff=None` for both splits. The first and only byte difference was the penultimate byte of each local file: local `\r\n`, GPU-B `\n`. Using GPU-B's exact NUL-separated dimension-fingerprint script on local images gave train `a14595067001ff54b1cd71f38441da9e0eb21e5557b5645e66fd28cd6d39d588` and val `674bbe62b92c622f3ae9bfdfaac37122eeecc42bf59e53c3f914b274fdf22e31`, matching GPU-B exactly. The earlier different local dimension hashes used a different serialization formula and did not indicate image differences.

The exporter now writes explicit UTF-8 bytes with LF. A full local re-export produced JSON files byte-for-byte identical to the synchronized GPU-B JSON, including the manifest before adding the new visible-identity fields. A subsequent full re-export with the visible-identity gate preserved those exact train/val annotation SHA values. The DEIMv2 runtime preparer now verifies annotation bytes and staged image bytes against pinned hashes, so neither platform newline conversion nor different source images can silently pass the gate.

The local identity audit uses fixed split order and hashes the literal UTF-8 lines `file_name\0width\0height\n` for dimensions, and `stem\0file_name\0byte_size\0file_sha256\n` for original image identity:

| Subset | Dimension fingerprint | Original visible aggregate SHA256 |
| --- | --- | --- |
| train | `9cf51f5950f2842ced67bf8ec8964a3a98fb3a8ae34c5e47295fd46953646089` | `a30d51404d21017766434fbfc03b07e89101d4121a61ec3239b1b70de51e3626` |
| val | `189c51a02d3c2adfd414cfe1a0df80fcf74db1e47d5d76f6ace244f47063415e` | `92b505f0391ba3f084b9c677ba1907f1a2ed26af9cec6c1e85f0189023718a0a` |

After pushing the fixed-exporter commit, re-run the exporter and verify the identity in GPU-B's project checkout. This audit is read-only:

```bash
cd /root/data1/AIC2026/AIC2026-Multimodal-Detection-B
python scripts/data/audit_deimv2_coco_identity.py \
  --dataset-root data/processed/deimv2_rgb_coco \
  > /root/data1/AIC2026/deimv2_coco_identity_B.json
```

The report gives Python/OpenCV versions, a precisely specified dimension fingerprint, original visible image aggregate SHA, first three source image SHA values, annotation SHA, and counts. The runtime preparer also independently verifies the staged images. Do not reuse an old manifest that lacks `visible_identity`.

For future JSON comparisons, copy GPU-B's `instances_train.json` and `instances_val.json` into `artifacts/DEIMV2_COCO_B_AUDIT/` locally, then run:

```powershell
python scripts/data/audit_deimv2_coco_identity.py `
  --dataset-root data/processed/deimv2_rgb_coco `
  --compare-root artifacts/DEIMV2_COCO_B_AUDIT
```

Interpretation for any future mismatch:

- If the original visible aggregate SHA differs, compare the first mismatched per-file SHA before changing the exporter.
- If the original visible aggregate and dimension fingerprints match but COCO numeric fields differ, inspect the first reported `$.annotations[...]` path for a cross-version arithmetic or serialization issue.
- If semantics are identical but JSON bytes differ, inspect encoding/serialization and then rework the hash contract around canonical JSON.
- If dimensions differ, inspect EXIF orientation and OpenCV decoding; image count and portrait count alone do not establish dimension identity.

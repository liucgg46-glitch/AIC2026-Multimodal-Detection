"""Repository-local fusion for two source-audited Ultralytics releases."""

def require_ultralytics():
    import ultralytics
    if ultralytics.__version__ not in ("8.3.253", "8.4.144"):
        raise RuntimeError("Fusion requires source-audited Ultralytics 8.3.253 or 8.4.144; run compatibility probe")

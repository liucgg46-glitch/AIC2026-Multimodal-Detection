import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def new_python_files():
    return (list((ROOT / "src/fusion").glob("*.py")) +
            list((ROOT / "tests/fusion").glob("*.py")) +
            [ROOT / "scripts/train/train_fusion.py",
             ROOT / "scripts/train/probe_fusion_compatibility.py",
             ROOT / "scripts/train/smoke_fusion_server.py"])


def test_all_new_python_parses_as_python38_without_modern_typing_syntax():
    for path in new_python_files():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path), feature_version=8)
        assert not any(type(node).__name__ == "Match" for node in ast.walk(tree)), path
        for node in ast.walk(tree):
            annotation = None
            if isinstance(node, ast.arg):
                annotation = node.annotation
            elif isinstance(node, (ast.AnnAssign, ast.FunctionDef, ast.AsyncFunctionDef)):
                annotation = node.annotation if isinstance(node, ast.AnnAssign) else node.returns
            if annotation is not None:
                assert not any(isinstance(part, ast.BinOp) and isinstance(part.op, ast.BitOr)
                               for part in ast.walk(annotation)), path
                assert not any(isinstance(part, ast.Subscript) and isinstance(part.value, ast.Name)
                               and part.value.id in ("list", "dict", "tuple", "set")
                               for part in ast.walk(annotation)), path


def test_local_compatibility_probe():
    result = subprocess.run([sys.executable, "scripts/train/probe_fusion_compatibility.py"], cwd=str(ROOT),
                            check=True, capture_output=True, text=True)
    assert '"status": "PASS"' in result.stdout
    assert '"private_overrides": []' in result.stdout


def test_formal_cli_requires_expected_sha():
    result = subprocess.run([sys.executable, "scripts/train/train_fusion.py", "--train"], cwd=str(ROOT),
                            check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "--expected-sha is required with --train" in result.stderr


def test_formal_cli_refuses_dirty_tree_before_trainer_creation():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    result = subprocess.run(
        [sys.executable, "scripts/train/train_fusion.py", "--train", "--expected-sha", head],
        cwd=str(ROOT), check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Reviewed execution requires a clean working tree" in result.stderr

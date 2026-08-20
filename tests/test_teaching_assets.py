import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_notebook_is_valid_and_code_cells_compile() -> None:
    notebook_path = ROOT / "notebooks" / "po2qat_results_lab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 10
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")


def test_teaching_docs_and_accessible_charts_exist() -> None:
    assignment = ROOT / "docs" / "assignments" / "PO2QAT_ASSIGNMENT.md"
    assert "Grading rubric (40 points)" in assignment.read_text(encoding="utf-8")

    for name in ("strong-vision-accuracy.svg", "strong-llm-results.svg"):
        svg = (ROOT / "docs" / "assets" / name).read_text(encoding="utf-8")
        assert "<title" in svg
        assert "<desc" in svg

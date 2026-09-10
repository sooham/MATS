import json
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "22_noisy_channel_bayesian_agreement_all_token_probes.ipynb"
)


def test_agreement_probe_notebook_is_valid_and_complete() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    code_source = "\n\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    )
    compile(code_source, str(NOTEBOOK_PATH), "exec")

    required_targets = {
        "gain",
        "negative_gain",
        "agreement_first_minus_second",
        "agreement_second_minus_first",
        "z_star",
        "agreement_first_vector",
        "agreement_second_vector",
        "agreement_concatenated_vector",
    }
    for target in required_targets:
        assert f'"{target}"' in code_source

    assert 'MODEL_NAME_OR_PATH = str(REPO_ROOT / "models" / "Qwen--Qwen3.5-4B")' in code_source
    assert 'ACTIVATION_LAYERS = "all"' in code_source
    assert 'ACTIVATION_TOKENS = "all"' in code_source
    assert "AgreementTranscriptDatasetGenerator(" in code_source
    assert "train_groups.isdisjoint(test_groups)" in code_source
    assert "train_transcripts.isdisjoint(test_transcripts)" in code_source
    assert "train_row_ids.isdisjoint(test_row_ids)" in code_source

    assert 'PROBE_TRAIN_DEVICE = "cuda:0"' in code_source
    assert "exact_batched_cuda_dual_cholesky" in code_source
    assert "torch.linalg.cholesky_ex" in code_source
    assert "torch.bmm" in code_source
    assert "no CPU fallback is allowed" in code_source
    assert "sklearn" not in code_source
    assert "upload_large_folder" in code_source
    assert "RUN_UPLOAD_PROBES_TO_HF = False" in code_source

    cell_ids = [cell.get("id") for cell in cells]
    main_results_index = cell_ids.index("main-probe-results")
    symmetry_index = cell_ids.index("run-and-display-symmetry-analysis")
    upload_index = cell_ids.index("publish-probes-to-huggingface")
    assert cell_ids.index("symmetric-reliability-rationale") > main_results_index
    assert symmetry_index > main_results_index
    assert upload_index > symmetry_index

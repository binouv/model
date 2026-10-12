"""RTX 3090 engineering-only R04 smoke test: pinned head -> hidden -> bridge -> frozen writer.

This is NOT a quality evaluation, and the default training label ('тест') is
unrelated to the teacher answer on purpose. Checks GPU loading, prompt-only
hidden extraction, gradient connectivity, model freezing and causal shift.
Both models run *sequentially*, avoiding 4B+0.8B simultaneous VRAM pressure.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from latent_bridge import LatentBridge, assemble_writer_embeddings, answer_cross_entropy


def pinned_model(name: str, spec: dict):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"],
        revision=spec["revision"],
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        trust_remote_code=False,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    ).eval()
    model.requires_grad_(False)
    model_type = type(model).__name__
    if "Qwen3_5ForCausalLM" not in model_type:
        raise RuntimeError(f"{name}: expected text-only Qwen3_5ForCausalLM, got {model_type}")
    if int(model.config.hidden_size) != spec["text_hidden_size"]:
        raise RuntimeError(f"{name}: unexpected hidden size {model.config.hidden_size}")
    if int(model.config.num_hidden_layers) != spec["text_num_hidden_layers"]:
        raise RuntimeError(f"{name}: unexpected number of text layers")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError(f"{name}: frozen weights unexpectedly trainable")
    return model


def safe_chat_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def calc_peak():
    torch.cuda.synchronize()
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / (1024 ** 3), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().parents[1] / "MODEL_LOCK.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "smoke.json")
    parser.add_argument("--max-peak-gib", type=float, default=22)
    args = parser.parse_args()
    specs = json.loads(args.lock.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: install official CUDA-enabled torch in the Python 3.13 venv, then rerun.")
    if torch.cuda.get_device_capability(0)[0] < 8 or not torch.cuda.is_bf16_supported():
        raise SystemExit("A CUDA GPU with working BF16 support is required.")
    try:
        import transformers
        from transformers import AutoTokenizer
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install requirements.txt after installing torch cu128") from exc

    report = {
        "run_status": "incomplete",
        "scientific_status": "engineering_smoke_only_not_an_accuracy_or_speed_benchmark",
        "head_decides_content_length": "protocol_goal_not_proven_by_smoke",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "vram_total_gib": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda_runtime": torch.version.cuda, "transformers": transformers.__version__},
        "pinned_models": {},
    }
    torch.manual_seed(20261012)
    torch.cuda.manual_seed_all(20261012)
    api = HfApi()
    for role in ("head", "writer"):
        spec = specs[role]
        resolved = api.model_info(spec["hf_id"], revision=spec["revision"]).sha
        if resolved != spec["revision"]:
            raise RuntimeError(f"Revision mismatch for {role}: {resolved}")
        report["pinned_models"][role] = {"repo": spec["hf_id"], "sha": resolved}

    torch.cuda.reset_peak_memory_stats()
    print("[1/3] Loading frozen 4B HEAD (BF16, text-only)", flush=True)
    begin = time.perf_counter()
    head = pinned_model("head", specs["head"])
    head_tokenizer = AutoTokenizer.from_pretrained(specs["head"]["hf_id"],
                           revision=specs["head"]["revision"], trust_remote_code=False)
    question = specs["first_gpu_smoke"]["prompt"]
    prompt = safe_chat_prompt(head_tokenizer, question)
    ids = head_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
    with torch.no_grad():
        outputs = head(input_ids=ids, use_cache=False,
                       output_hidden_states=True, return_dict=True)
        teacher_last = outputs.hidden_states[-1][:, -1, :].detach().float().cpu().clone()
    if tuple(teacher_last.shape) != (1, specs["head"]["text_hidden_size"]):
        raise AssertionError(f"Wrong head feature shape: {teacher_last.shape}")
    if not torch.isfinite(teacher_last).all():
        raise AssertionError("Nonfinite head state")
    report["head"] = {"question": question, "input_tokens": int(ids.size(1)),
                       "latent_shape": list(teacher_last.shape),
                       "latent_std": float(teacher_last.std()),
                       "wall_seconds": round(time.perf_counter() - begin, 3),
                       "memory": calc_peak()}
    del outputs, head, head_tokenizer, ids
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.memory_allocated() > 256 * 1024 * 1024:
        print("Warning: some allocations remain after HEAD unload", flush=True)

    print("[2/3] Loading frozen 0.8B WRITER (BF16, text-only)", flush=True)
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    writer = pinned_model("writer", specs["writer"])
    writer_tok = AutoTokenizer.from_pretrained(specs["writer"]["hf_id"],
                         revision=specs["writer"]["revision"], trust_remote_code=False)
    cfg = specs["bridge"]
    bridge = LatentBridge(specs["head"]["text_hidden_size"],
                          specs["writer"]["text_hidden_size"],
                          cfg["plan_tokens"], cfg["hidden_size"]).to("cuda:0")
    optimizer = AdamW(bridge.parameters(), lr=1e-4)
    # Fixed start marker. It conveys NO user question or response style.
    fixed_header = "<|im_start|>assistant\n"
    start_ids = writer_tok(fixed_header, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")
    # Engineering-only target label. NOT the answer selected by 4B.
    engineering_label = specs["first_gpu_smoke"]["engineering_label"] + "<|im_end|>"
    target_ids = writer_tok(engineering_label, add_special_tokens=False,
                            return_tensors="pt").input_ids.to("cuda:0")
    teacher_states = teacher_last.to("cuda:0")
    soft_plan = bridge(teacher_states)
    combined = assemble_writer_embeddings(soft_plan, start_ids, target_ids,
                                           writer.get_input_embeddings())
    predictions = writer(inputs_embeds=combined, use_cache=False, return_dict=True)
    loss = answer_cross_entropy(predictions.logits, target_ids,
                                soft_plan.size(1), start_ids.size(1))
    if not torch.isfinite(loss):
        raise AssertionError("Nonfinite training loss")
    loss.backward()
    grad_norm = sum(float(p.grad.float().norm()) for p in bridge.parameters() if p.grad is not None)
    if grad_norm <= 0 or not torch.isfinite(torch.tensor(grad_norm)):
        raise AssertionError("No valid gradient through frozen WRITER to bridge")
    if any(p.grad is not None for p in writer.parameters()):
        raise AssertionError("WRITER must remain fully frozen")
    optimizer.step()
    report["writer_and_bridge"] = {
        "start_marker_only": fixed_header,
        "writer_saw_original_question": False,
        "writer_received_length_or_style_flag": False,
        "training_label": "engineering_only_not_from_teacher_and_not_an_accuracy_result",
        "plan_shape": list(soft_plan.shape),
        "bridge_trainable_parameters": sum(p.numel() for p in bridge.parameters()),
        "writer_trainable_parameters": sum(int(p.requires_grad) * p.numel() for p in writer.parameters()),
        "loss_before_one_step": round(float(loss.detach()), 5),
        "bridge_grad_norm": round(grad_norm, 6),
        "wall_seconds": round(time.perf_counter() - begin, 3),
        "memory": calc_peak(),
    }
    max_peak = max(report["head"]["memory"]["peak_reserved_gib"],
                   report["writer_and_bridge"]["memory"]["peak_reserved_gib"])
    if max_peak >= args.max_peak_gib:
        raise RuntimeError(f"Peak reserved VRAM {max_peak:.2f} GiB exceeds safe ceiling {args.max_peak_gib}")
    report["run_status"] = "passed_smoke_no_scientific_hypothesis_tested"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"ENGINEERING SMOKE PASS; report: {args.output}")


if __name__ == "__main__":
    main()
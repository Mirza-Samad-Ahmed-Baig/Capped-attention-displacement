"""Deployment-cost endpoints for E-S3 (PROTOCOL A 3.4): 12-layer BiomedBERT vs the 6-layer student.

Measures total and non-embedding parameters, fp16 weight size, GPU batch-1 latency and batch-32
throughput at 384 tokens (eval, bf16, SDPA), and CPU 4-thread batch-1 latency. Run only when the GPU
is otherwise idle (no training job); results/cost.json (write-once). Timing is descriptive (one laptop).
"""
import os
import statistics
import time

import env

torch = env.setup_model_process(threads=4)
from torch import nn  # noqa: E402

import runlib  # noqa: E402

MODEL = os.path.join(env.WORKSPACE, "domain_distill", "models", "biomedbert")
LAYERS = [0, 2, 4, 6, 8, 10]


def build(keep=None):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(MODEL, local_files_only=True, use_safetensors=False, weights_only=True,
                                  add_pooling_layer=False, attn_implementation="sdpa")
    if keep:
        m.encoder.layer = nn.ModuleList([m.encoder.layer[i] for i in keep])
        m.config.num_hidden_layers = len(keep)
    return m.eval()


def timeit(fn, reps, warm=3):
    for _ in range(warm):
        fn()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return statistics.median(out)


def measure(model, name):
    total = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for p in model.embeddings.parameters())
    rec = {"parameters": total, "non_embedding_parameters": total - emb, "fp16_megabytes": total * 2 / 1e6}
    ids1 = torch.randint(1000, 20000, (1, 384))
    ids32 = torch.randint(1000, 20000, (32, 384))
    with torch.no_grad():
        m = model.cuda()

        def gpu(x):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m(input_ids=x)
            torch.cuda.synchronize()
        x1, x32 = ids1.cuda(), ids32.cuda()
        rec["gpu_batch1_latency_ms"] = 1000 * timeit(lambda: gpu(x1), 50)
        rec["gpu_batch32_seq_per_s"] = 32 / timeit(lambda: gpu(x32), 10)
        m = model.cpu().float()
        rec["cpu4_batch1_latency_ms"] = 1000 * timeit(lambda: m(input_ids=ids1), 10, warm=2)
    print(name, rec, flush=True)
    return rec


def main():
    env.cap_vram()
    out = {"utc": runlib.utc(), "device": torch.cuda.get_device_name(0), "tokens": 384,
           "teacher_size_12_layers": measure(build(), "12-layer"), "student_6_layers": measure(build(LAYERS), "6-layer")}
    print(runlib.write_json_once(os.path.join(env.ROOT, "results", "cost.json"), out))


if __name__ == "__main__":
    main()

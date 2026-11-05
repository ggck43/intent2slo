"""Energy probe v2 (notebook 18, plan plans/20261025.energyAnalysis.md). One vLLM server per model, an asynchronous client, GPU energy from the NVML counter.
Configurations: zs3, fs3, zs3_cot, fs3_cot (native thinking off). Per configuration:
  - concurrency 1 on 60 valid intents (12 per level), per-request energy = counter before/after, fresh prefix cache; two extra passes over the first 20 (noise check);
  - concurrency 1 on 40 invalid intents (10 per category);
  - closed loops at concurrency 16 (96 distinct intents) and 64 (256 distinct intents), window energy from the counter, per-request start/end times for steady-state trimming.
A foreign-process check (number of compute processes on the GPU, baseline taken after the server is ready) is recorded in every window; contaminated windows are flagged.
Logs under <out-dir>/<model-tag>/: server.log, requests.jsonl, phases.jsonl, power_trace.csv (time, power, phase, n_procs), env.json.
Usage (vllm venv): python scripts/energy_probe2.py --model Qwen/Qwen3.5-4B --gpu 0 --out-dir outputs/energy/run_<id> [--quick]
"""
import argparse
import asyncio
import copy
import csv
import datetime as dt
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pynvml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_io import load_records, load_snapshot, snapshot_compact  # noqa: E402
from src.energy import invalid_sample, loop_sample, n1_sample  # noqa: E402
from src.parse import parse_response  # noqa: E402
from src.prompts import build_messages, load_fs_prompt  # noqa: E402

PROMPTS = {"zs3": "zs3_prompt.txt", "fs3": "fs3_prompt.txt", "zs3_cot": "zs3_cot_prompt.txt", "fs3_cot": "fs3_cot_prompt.txt"}
MAX_TOKENS = {"zs3": 2048, "fs3": 2048, "zs3_cot": 4096, "fs3_cot": 4096}
VLLM = ROOT / ".venv-vllm" / "bin" / "vllm"


class Gpu:
    def __init__(self, index):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.index, self.name, self.phase, self.trace = index, pynvml.nvmlDeviceGetName(self.h), "", []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True); self._thread.start()

    def energy_j(self): return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1000.0
    def power_w(self): return pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
    def mem_used_mib(self): return pynvml.nvmlDeviceGetMemoryInfo(self.h).used / 2 ** 20
    def n_procs(self): return len(pynvml.nvmlDeviceGetComputeRunningProcesses(self.h))

    def _sample(self, step=0.25):
        k = 0
        while not self._stop.is_set():
            self.trace.append((time.time(), self.power_w(), self.phase, self.n_procs() if k % 8 == 0 else ""))
            k += 1
            self._stop.wait(step)

    def idle_power(self, seconds=10.0):
        t0 = time.time(); e0 = self.energy_j(); time.sleep(seconds)
        return (self.energy_j() - e0) / (time.time() - t0)

    def save_trace(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t_epoch", "power_W", "phase", "n_procs"]); w.writerows(self.trace)

    def stop(self): self._stop.set()


class Client:
    def __init__(self, base_url, model, family):
        self.http = httpx.AsyncClient(base_url=base_url, timeout=None, limits=httpx.Limits(max_connections=512))
        self.model, self.family = model, family

    async def chat(self, messages, max_tokens):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0, "seed": 0}
        if self.family in ("qwen", "gemma"):
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = await self.http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        j = r.json(); ch = j["choices"][0]; u = j["usage"]
        return {"text": ch["message"]["content"] or "", "finish_reason": ch["finish_reason"], "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
                "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0}


class Setup:
    def __init__(self, snap_str):
        self.snap = snap_str
        self.parts = {k: load_fs_prompt(v) for k, v in PROMPTS.items()}

    def messages(self, rec, cfg):
        system, template = self.parts[cfg]
        return build_messages(rec, self.snap, system, template)


def record_of(res, rec, phase, cfg, t0, t1, energy=None, extra=None):
    p = parse_response(res["text"])
    out = {"phase": phase, "config": cfg, "index": rec["index"], "status": rec["meta"]["status"], "level": rec["meta"]["level"], "t_start": t0, "t_end": t1, "latency_s": round(t1 - t0, 4),
           "energy_J": energy, "prompt_tokens": res["prompt_tokens"], "cached_tokens": res["cached_tokens"], "uncached_prompt_tokens": res["prompt_tokens"] - res["cached_tokens"],
           "completion_tokens": res["completion_tokens"], "finish_reason": res["finish_reason"], "parse_ok": p["parse_ok"], "answer_text": res["text"]}
    if extra:
        out.update(extra)
    return out


def start_server(model, gpu_index, port, log_path, max_len, family, hf_cache):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index), VLLM_USE_FLASHINFER_SAMPLER="0", VLLM_SERVER_DEV_MODE="1")
    if hf_cache:
        env["HF_HUB_CACHE"] = hf_cache
    cmd = [str(VLLM), "serve", model, "--port", str(port), "--dtype", "bfloat16", "--max-model-len", str(max_len), "--gpu-memory-utilization", "0.7",
           "--max-num-seqs", "128", "--seed", "0", "--generation-config", "vllm", "--enable-prefix-caching", "--enable-prompt-tokens-details"]
    if family in ("gemma", "ministral"):
        cmd += ["--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0, "video": 0})]
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
    return proc, cmd


def wait_ready(proc, port, timeout=1500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            if httpx.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("server did not become ready")


def stop_server(proc, gpu):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT); proc.wait(timeout=120)
    except Exception:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    t0 = time.time()
    while gpu.mem_used_mib() > 1500 and time.time() - t0 < 120:
        time.sleep(2)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))] if xs else None


async def fresh_cache(client, setup, dummy, cfg):
    """empty the prefix cache and send one intent that is not in any sample: afterwards only the shared prompt prefix is cached, as in deployment where every intent is new"""
    (await client.http.post("/reset_prefix_cache")).raise_for_status()
    await client.chat(setup.messages(dummy, cfg), MAX_TOKENS[cfg])
    await asyncio.sleep(1.0)


async def sequential(gpu, client, setup, sample, cfg, phase, req_f, dummy, base_procs):
    await fresh_cache(client, setup, dummy, cfg)
    gpu.phase = phase
    p0, e_all0, t_all0 = gpu.n_procs(), gpu.energy_j(), time.time()
    for rec in sample:
        msgs = setup.messages(rec, cfg)
        t0 = time.time(); e0 = gpu.energy_j()
        res = await client.chat(msgs, MAX_TOKENS[cfg])
        e1 = gpu.energy_j(); t1 = time.time()
        req_f.write(json.dumps(record_of(res, rec, phase, cfg, t0, t1, energy=e1 - e0)) + "\n"); req_f.flush()
    return {"phase": phase, "config": cfg, "concurrency": 1, "requests": len(sample), "window_energy_J": gpu.energy_j() - e_all0, "window_s": time.time() - t_all0,
            "procs_start": p0, "procs_end": gpu.n_procs(), "contaminated": max(p0, gpu.n_procs()) > base_procs}


async def closed_loop(gpu, client, setup, sample, cfg, N, phase, req_f, dummy, base_procs):
    await fresh_cache(client, setup, dummy, cfg)
    gpu.phase = phase
    results, it = [], iter(range(len(sample)))

    async def worker():
        while True:
            try:
                i = next(it)
            except StopIteration:
                return
            rec = sample[i]
            msgs = setup.messages(rec, cfg)
            t0 = time.time()
            res = await client.chat(msgs, MAX_TOKENS[cfg])
            results.append(record_of(res, rec, phase, cfg, t0, time.time(), extra={"concurrency": N}))

    p0 = gpu.n_procs(); e0, t0 = gpu.energy_j(), time.time()
    await asyncio.gather(*[worker() for _ in range(N)])
    e1, t1 = gpu.energy_j(), time.time()
    for r in results:
        req_f.write(json.dumps(r) + "\n")
    req_f.flush()
    K = len(sample); lat = [r["latency_s"] for r in results]; gen = sum(r["completion_tokens"] for r in results)
    return {"phase": phase, "config": cfg, "concurrency": N, "requests": K, "t_window_start": t0, "t_window_end": t1, "window_s": t1 - t0, "window_energy_J": e1 - e0,
            "energy_per_intent_J": (e1 - e0) / K, "mean_power_W": (e1 - e0) / (t1 - t0), "throughput_req_per_s": K / (t1 - t0), "throughput_tok_per_s": gen / (t1 - t0),
            "latency_mean_s": sum(lat) / len(lat), "latency_p50_s": pct(lat, 50), "latency_p95_s": pct(lat, 95), "mean_completion_tokens": gen / K,
            "mean_cached_tokens": sum(r["cached_tokens"] for r in results) / K, "procs_start": p0, "procs_end": gpu.n_procs(), "contaminated": max(p0, gpu.n_procs()) > base_procs}


async def main_async(a):
    model_tag = a.model.split("/")[-1].lower().replace(".", "")
    out = Path(a.out_dir) / model_tag; out.mkdir(parents=True, exist_ok=True)
    low = a.model.lower(); family = "gemma" if "gemma" in low else "ministral" if "ministral" in low else "qwen"
    records = load_records(); snap_str = snapshot_compact(load_snapshot())
    if a.quick:
        s1 = n1_sample(records, 1); noise_n = 2; s_inv = invalid_sample(records, 1); k16, k64 = 16, 32
    else:
        s1 = n1_sample(records, 12); noise_n = 20; s_inv = invalid_sample(records, 10); k16, k64 = 96, 256
    used = {r["index"] for r in s1}
    s16 = loop_sample(records, k16, used); s64 = loop_sample(records, k64, used)
    dummy = next(r for r in records if r["meta"]["status"] == "valid" and r["meta"]["level"] == 2 and r["index"] not in used | {x["index"] for x in s16 + s64})
    Path(a.out_dir, "samples.json").write_text(json.dumps({"n1": [r["index"] for r in s1], "invalid": [r["index"] for r in s_inv], "c16": [r["index"] for r in s16], "c64": [r["index"] for r in s64], "dummy": dummy["index"]}, indent=1))
    gpu = Gpu(a.gpu)
    t0 = time.time()
    while gpu.mem_used_mib() > 1500 and time.time() - t0 < 600:
        time.sleep(5)
    if gpu.mem_used_mib() > 1500:
        raise RuntimeError(f"GPU {a.gpu} is not free ({gpu.mem_used_mib():.0f} MiB used)")
    phases = open(out / "phases.jsonl", "a"); req_f = open(out / "requests.jsonl", "a")
    def log_phase(d):
        d = {"t_end": time.time(), "model": a.model, **d}; phases.write(json.dumps(d) + "\n"); phases.flush()
        print("PHASE", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != "cmd"}), flush=True)
    env = {"date": dt.datetime.now().isoformat(timespec="seconds"), "model": a.model, "family": family, "gpu_index": a.gpu, "gpu_name": gpu.name, "hostname": platform.node(),
           "vllm": subprocess.run([str(ROOT / ".venv-vllm/bin/python"), "-c", "import importlib.metadata as m;print(m.version('vllm'))"], capture_output=True, text=True).stdout.strip(),
           "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit", "--format=csv,noheader", "-i", str(a.gpu)], capture_output=True, text=True).stdout.strip(),
           "args": vars(a), "prompt_files": PROMPTS, "max_tokens": MAX_TOKENS}
    (out / "env.json").write_text(json.dumps(env, indent=1))
    gpu.phase = "idle_empty_before"; log_phase({"phase": "idle_empty_before", "power_W": gpu.idle_power(10)})
    gpu.phase = "server_start"; e0, ts = gpu.energy_j(), time.time()
    proc, cmd = start_server(a.model, a.gpu, a.port, out / "server.log", a.max_len, family, a.hf_cache)
    try:
        ready = wait_ready(proc, a.port)
        log_phase({"phase": "server_start", "ready_s": ready, "energy_J": gpu.energy_j() - e0, "cmd": cmd})
        gpu.phase = "idle_loaded"; time.sleep(3)
        base_procs = gpu.n_procs()
        log_phase({"phase": "idle_loaded", "power_W": gpu.idle_power(10), "gpu_mem_MiB": gpu.mem_used_mib(), "baseline_procs": base_procs})
        client = Client(f"http://localhost:{a.port}", a.model, family); setup = Setup(snap_str)
        for cfg in a.configs.split(","):
            print(f"===== {a.model} | {cfg} =====", flush=True)
            log_phase(await sequential(gpu, client, setup, s1, cfg, f"{cfg}_n1_valid", req_f, dummy, base_procs))
            for rep in (1, 2):
                log_phase(await sequential(gpu, client, setup, s1[:noise_n], cfg, f"{cfg}_n1_rep{rep}", req_f, dummy, base_procs))
            log_phase(await sequential(gpu, client, setup, s_inv, cfg, f"{cfg}_n1_invalid", req_f, dummy, base_procs))
            log_phase(await closed_loop(gpu, client, setup, s16, cfg, 16, f"{cfg}_c16", req_f, dummy, base_procs))
            log_phase(await closed_loop(gpu, client, setup, s64, cfg, 64, f"{cfg}_c64", req_f, dummy, base_procs))
            gpu.phase = f"{cfg}_idle_after"; log_phase({"phase": "idle_loaded_after", "config": cfg, "power_W": gpu.idle_power(5)})
    finally:
        if "client" in locals():
            await client.http.aclose()
        stop_server(proc, gpu)
        gpu.phase = "idle_empty_after"; log_phase({"phase": "idle_empty_after", "power_W": gpu.idle_power(10)})
        gpu.stop(); gpu.save_trace(out / "power_trace.csv"); phases.close(); req_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--gpu", type=int, required=True); ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--configs", default="zs3,fs3,zs3_cot,fs3_cot")
    ap.add_argument("--max-len", type=int, default=16384); ap.add_argument("--hf-cache", default=os.environ.get("HF_HUB_CACHE"))
    ap.add_argument("--quick", action="store_true", help="pipeline test with tiny samples")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

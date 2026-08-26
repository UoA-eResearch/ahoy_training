#!/usr/bin/env python3
"""Run one model across TWO GB10s -- the support code behind menu option 7.

Some of these GB10s are installed as cabled pairs: two 200 GbE ConnectX links
between the boxes carrying RoCEv2, with fixed addressing on the private rail --

    head    192.168.100.1   (lower hostname of the pair; menus and rank 0 run here)
    worker  192.168.100.2

A 72B model in bf16 (~150 GB) beats one box's ~110 GB usable, but split across
a pair it fits with room to spare. Two mechanisms, one per direction of use:

  * training   -- run_torchrun(): torchrun on both nodes, DeepSpeed ZeRO-3
                  sharding the frozen weights (~75 GB a side) over the link;
                  the LoRA recipe itself is unchanged
  * inference  -- serve(): vLLM with tensor parallel 2 over Ray, using the
                  /opt/vllm/venv these boxes are provisioned with, spoken to
                  through its OpenAI-compatible API (chat() below)

The NCCL/Ray environment here is the hardware-validated set from the pair
bring-up (gb10-puppet, documentation/07-distributed-vllm.md). The traps it
encodes, so nobody has to rediscover them: the RoCE devices are named rocep*
(matching ^mlx5 finds nothing and NCCL silently falls back to slow TCP), GID
index 3 is the RoCEv2 IPv4 GID, Ray's OOM monitor must be off because unified
memory makes the KV cache look like system RAM, the campus proxy must never
see rail traffic, and vLLM's Ray channel must be forced to nccl or every
decode step round-trips Ray's object store (a ~10000x slowdown that looks
like a hang).
"""
import argparse, json, os, re, shlex, signal, subprocess, sys, time, urllib.request

HEAD_IP, WORKER_IP = "192.168.100.1", "192.168.100.2"
RAY_PORT, API_PORT = 6380, 8001      # the lais-agent owns 6379/8000; stay clear of it
MASTER_PORT = 29500                  # torchrun rendezvous, on the rail
VLLM_VENV = "/opt/vllm/venv"
REMOTE_DIR = "~/ahoy_training"       # repo location on the worker (same as tune_remote.sh uses)
OUR_URL = f"http://127.0.0.1:{API_PORT}/v1"
MANAGED_URL = "http://127.0.0.1:8000/v1"   # a portal-managed vLLM; reused read-only, never stopped
PID_FILE, LOG_FILE = "out/pair-vllm.pid", "out/pair-vllm.log"
LORA_NAME = "tuned"                  # serving name for an adapter mounted into vLLM

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
       "-o", "StrictHostKeyChecking=accept-new", WORKER_IP]

# never let the campus proxy near requests to our own endpoints
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# --------------------------------------------------------------------------- environment
def cluster_env(own_ip, rdma=True):
    """The validated NCCL/Gloo/Ray environment for the pair rail."""
    no_proxy = ",".join(x for x in (os.environ.get("no_proxy", ""),
                                    "localhost,127.0.0.1,::1", HEAD_IP, WORKER_IP) if x)
    return {
        "NCCL_SOCKET_IFNAME": "enp1s0f0np0,enp1s0f1np1",
        "NCCL_IB_HCA": "rocep1s0f0,rocep1s0f1",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_IB_DISABLE": "0" if rdma else "1",
        "GLOO_SOCKET_IFNAME": "enp1s0f0np0",
        "NCCL_DEBUG": "WARN",   # surface NCCL failures in the run log
        # unified memory rides the OOM edge during the sharded 72B load; keep
        # the caching allocator from overshooting beyond what it's using
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.7",
        "VLLM_HOST_IP": own_ip,
        "RAY_grpc_enable_http_proxy": "0",
        "RAY_memory_monitor_refresh_ms": "0",
        "RAY_memory_usage_threshold": "1.0",
        "no_proxy": no_proxy,
        "NO_PROXY": no_proxy,
    }


def _exports(own_ip, rdma=True):
    """The same environment as a shell prefix, for commands run on the worker."""
    pairs = " ".join(f"{k}={shlex.quote(v)}" for k, v in cluster_env(own_ip, rdma).items()
                     if k not in ("no_proxy", "NO_PROXY"))
    return (f"export {pairs}; "
            'export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1,'
            f'{HEAD_IP},{WORKER_IP}\"; export NO_PROXY=\"$no_proxy\"; '
            "ulimit -l unlimited 2>/dev/null || true; ")


def _ssh_run(cmd, capture=True, check=False, fail_msg=None):
    r = subprocess.run(SSH + [cmd], text=True, capture_output=capture)
    if check and r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip() if capture else ""
        sys.exit(f"{fail_msg or 'command on the worker failed'}: {cmd}\n{detail}")
    return r


# --------------------------------------------------------------------------- preflight
def _local_ips():
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr", "show"],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return set()
    return set(re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", out))


def check():
    """Problems keeping this box from driving its pair (empty list = good to go)."""
    ips = _local_ips()
    if WORKER_IP in ips:
        return [f"this box is the WORKER of its pair ({WORKER_IP}) -- run from the "
                f"head (the lower hostname of the pair, {HEAD_IP} on the rail)"]
    if HEAD_IP not in ips:
        return [f"no pair rail on this box (no {HEAD_IP} on any interface) -- this "
                "GB10 is not the head of a cabled pair"]
    probs = []
    if subprocess.run(["ping", "-c1", "-W2", WORKER_IP], capture_output=True).returncode:
        probs.append(f"the worker does not answer on the rail (ping {WORKER_IP} failed) "
                     "-- check the cabling/switch before trusting the pair")
    elif _ssh_run("true").returncode:
        probs.append(f"passwordless SSH to the worker failed -- fix with: "
                     f"ssh-copy-id {WORKER_IP}  (from this box; make a key with "
                     "ssh-keygen first if you have none)")
    return probs


def require_pair():
    probs = check()
    if probs:
        sys.exit("cannot run across the pair from this box:\n  - " + "\n  - ".join(probs))


def _rdma_ok():
    """RDMA pins large memory regions; a low memlock cap kills NCCL at init
    with 'unhandled system error'. If either box caps it, warn once and fall
    back to NCCL over TCP on the same 200 GbE links -- works, just slower."""
    import resource
    ok_local = ok_remote = False
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if hard == resource.RLIM_INFINITY or hard >= 8 << 30:
        try:
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
        except (ValueError, OSError):
            pass
        ok_local = True
    r = _ssh_run("ulimit -H -l")
    out = (r.stdout or "").strip()
    ok_remote = out == "unlimited" or (out.isdigit() and int(out) >= (8 << 30) // 1024)
    if ok_local and ok_remote:
        return True
    print("!! memlock is capped on "
          + ("both boxes" if not (ok_local or ok_remote) else
             "this box" if not ok_local else "the worker")
          + " -- using NCCL over TCP on the 200 GbE links instead of RDMA (slower).")
    print("   fix: '* - memlock unlimited' in /etc/security/limits.conf on both boxes")
    return False


# --------------------------------------------------------------------------- worker sync
def _rsync(src, dst):
    subprocess.run(["rsync", "-a", "-e", " ".join(SSH[:-1]), src, dst], check=True)


def sync_to_worker(dirs=("scripts", "data")):
    """Mirror the given repo directories to the worker over the rail."""
    _ssh_run(f"mkdir -p {REMOTE_DIR}", check=True, fail_msg="mkdir on the worker failed")
    for d in dirs:
        if os.path.isdir(d):
            _rsync(f"{d}/", f"{WORKER_IP}:{REMOTE_DIR}/{d}/")


def _ensure_deepspeed_local():
    if subprocess.run([sys.executable, "-c", "import deepspeed"],
                      capture_output=True).returncode == 0:
        return
    print("installing deepspeed into this venv (one-off)...")
    uv = os.path.expanduser("~/.local/bin/uv")
    uv = uv if os.path.isfile(uv) else "uv"
    r = subprocess.run([uv, "pip", "install", "--python", sys.executable, "deepspeed"])
    if r.returncode:
        sys.exit("deepspeed install failed -- try by hand:\n"
                 f"  DS_BUILD_OPS=0 {uv} pip install --no-build-isolation "
                 f"--python {sys.executable} deepspeed")


def prepare_worker():
    """Repo + venv + deepspeed on the worker. Idempotent; slow only on first use."""
    _ssh_run("bash -lc 'command -v uv'", check=True,
             fail_msg="uv is not installed on the worker -- on it, run: "
                      "curl -LsSf https://astral.sh/uv/install.sh | sh\nerror")
    sync_to_worker()
    print("preparing the worker's venv (a few minutes on first use)...")
    _ssh_run(f"bash -lc 'cd {REMOTE_DIR} && bash scripts/setup.sh'", capture=False,
             check=True, fail_msg="setup.sh failed on the worker")
    _ssh_run(f"bash -lc 'cd {REMOTE_DIR} && .venv/bin/python -c \"import deepspeed\" "
             "2>/dev/null || uv pip install --python .venv/bin/python deepspeed'",
             capture=False, check=True, fail_msg="deepspeed install failed on the worker")


# --------------------------------------------------------------------------- training
def run_torchrun(argv):
    """Run `python argv...` as a 2-node torchrun: rank 0 here on the head,
    rank 1 on the worker over SSH. Returns the head rank's exit code."""
    require_pair()
    stop_serving(quiet=True)                  # both GPUs must be free for training
    rdma = _rdma_ok()
    _ensure_deepspeed_local()
    prepare_worker()

    common = (f"--nnodes=2 --nproc-per-node=1 --master-addr={HEAD_IP} "
              f"--master-port={MASTER_PORT}")
    worker_cmd = (_exports(WORKER_IP, rdma) +
                  f"cd {REMOTE_DIR} && source .venv/bin/activate && "
                  f"torchrun {common} --node-rank=1 " + shlex.join(argv) +
                  " 2>&1 | sed -u 's/^/[worker] /'")
    print("launching rank 1 on the worker...")
    worker = subprocess.Popen(SSH + [worker_cmd])

    env = dict(os.environ, **cluster_env(HEAD_IP, rdma))
    torchrun = os.path.join(os.path.dirname(sys.executable), "torchrun")
    try:
        rc = subprocess.call([torchrun, *common.split(), "--node-rank=0", *argv], env=env)
    finally:
        try:
            worker.wait(timeout=60)
        except subprocess.TimeoutExpired:
            worker.terminate()
            _ssh_run("pkill -f 'torchrun --nnodes=2' || true")
    return rc


# --------------------------------------------------------------------------- serving (vLLM over the pair)
def _served(base_url):
    """Set of model names an endpoint serves, or None if nothing is listening."""
    try:
        with _OPENER.open(base_url + "/models", timeout=3) as r:
            return {m["id"] for m in json.load(r)["data"]}
    except Exception:
        return None


def serve(model_id, lora=None):
    """An OpenAI-compatible endpoint for `model_id` served across the pair
    (vLLM, tensor parallel 2 over Ray). Returns (base_url, name) where `name`
    is what requests pass as "model": the model id, or the adapter's serving
    name when `lora` (an adapter directory) is given.

    A live endpoint that already serves what we need is reused -- including a
    portal-managed one on :8000, which is never started or stopped from here.
    """
    want = {model_id, LORA_NAME} if lora else {model_id}
    name = LORA_NAME if lora else model_id
    ours = _served(OUR_URL)
    if ours is not None and want <= ours:
        print(f"reusing the running pair server on :{API_PORT}")
        return OUR_URL, name
    if not lora:
        managed = _served(MANAGED_URL)
        if managed is not None and model_id in managed:
            print(f"reusing the portal-managed vLLM on :8000 (already serves {model_id})")
            return MANAGED_URL, model_id
    if ours is not None:                       # ours, but serving the wrong thing
        stop_serving(quiet=True)
    _start_cluster(model_id, lora)
    return OUR_URL, name


def _start_cluster(model_id, lora):
    require_pair()
    if not os.path.isdir(VLLM_VENV):
        sys.exit(f"{VLLM_VENV} not found -- paired GB10s are provisioned with it "
                 "(gb10-puppet profile::gb10::vllm); is this box puppet-managed?")
    # a leftover root-owned /tmp/ray makes ray abort while opening its logs
    if os.path.isdir("/tmp/ray") and not os.access("/tmp/ray", os.W_OK):
        sys.exit("/tmp/ray exists but is not writable by you (left by a root run?) "
                 "-- remove it first: sudo rm -rf /tmp/ray")
    os.makedirs("out", exist_ok=True)
    rdma = _rdma_ok()
    ray = f"{VLLM_VENV}/bin/ray"
    env = dict(os.environ, **cluster_env(HEAD_IP, rdma))

    print("starting ray on both boxes...")
    subprocess.run([ray, "stop", "--force"], capture_output=True)
    _ssh_run(f"{ray} stop --force")
    # NOTE: RAY_ADDRESS must NOT be set for `ray start --head` (it would try to
    # attach to itself); it is set only on the vLLM server below.
    r = subprocess.run([ray, "start", "--head", f"--port={RAY_PORT}",
                        f"--node-ip-address={HEAD_IP}", "--num-gpus=1"],
                       env=env, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"ray head failed to start:\n{r.stderr or r.stdout}")
    _ssh_run(_exports(WORKER_IP, rdma) +
             f"{ray} start --address={HEAD_IP}:{RAY_PORT} "
             f"--node-ip-address={WORKER_IP} --num-gpus=1",
             check=True, fail_msg="ray failed to start on the worker")
    for _ in range(24):
        out = subprocess.run([ray, "status", f"--address={HEAD_IP}:{RAY_PORT}"],
                             capture_output=True, text=True).stdout
        if out.count("node_") >= 2:
            break
        time.sleep(5)
    else:
        stop_serving(quiet=True)
        sys.exit("the worker never joined the ray cluster -- check the rail "
                 f"(ping {WORKER_IP}) and try again")

    if lora:
        sync_to_worker(dirs=(os.path.relpath(lora).split(os.sep)[0],))

    cmd = [f"{VLLM_VENV}/bin/python3", "-m", "vllm.entrypoints.openai.api_server",
           "--model", model_id, "--host", "127.0.0.1", "--port", str(API_PORT),
           "--tensor-parallel-size", "2", "--distributed-executor-backend", "ray",
           "--enforce-eager", "--max-model-len", "4096",
           "--gpu-memory-utilization", "0.80"]
    if lora:
        cmd += ["--enable-lora", "--max-lora-rank", "64",
                "--lora-modules", f"{LORA_NAME}={os.path.abspath(lora)}"]
    env["RAY_ADDRESS"] = f"{HEAD_IP}:{RAY_PORT}"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE"] = "nccl"  # the ~10000x trap
    log = open(LOG_FILE, "ab")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log,
                            start_new_session=True)
    open(PID_FILE, "w").write(str(proc.pid))
    print(f"loading {model_id} into vLLM across the pair -- first use downloads "
          f"~150 GB on EACH box. Progress: tail -f {LOG_FILE}")

    deadline = time.time() + 3 * 3600
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = subprocess.run(["tail", "-15", LOG_FILE],
                                  capture_output=True, text=True).stdout
            stop_serving(quiet=True)
            sys.exit(f"the vLLM server died while starting; last log lines:\n{tail}")
        try:
            _OPENER.open(f"http://127.0.0.1:{API_PORT}/health", timeout=3)
            break
        except Exception:
            time.sleep(10)
    else:
        stop_serving(quiet=True)
        sys.exit(f"the vLLM server did not come up within 3 h -- see {LOG_FILE}")
    print(f"pair server ready on :{API_PORT}  (stop later: python scripts/pair.py --stop)")


def stop_serving(quiet=False):
    """Stop our vLLM server and our ray processes on both boxes. Never touches
    a portal-managed vLLM on :8000 (different user; `ray stop` can't see it)."""
    had = False
    if os.path.isfile(PID_FILE):
        try:
            pid = int(open(PID_FILE).read().strip())
            os.killpg(pid, signal.SIGTERM)
            had = True
            for _ in range(20):
                try:
                    os.killpg(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.5)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        os.remove(PID_FILE)
    ray = f"{VLLM_VENV}/bin/ray"
    if os.path.isfile(ray):
        subprocess.run([ray, "stop", "--force"], capture_output=True)
        if HEAD_IP in _local_ips():
            _ssh_run(f"{ray} stop --force")
    if had and not quiet:
        print("stopped the pair vLLM server")


# --------------------------------------------------------------------------- client
def chat(base_url, model, messages, max_tokens=200, greedy=False):
    """One chat completion against a pair endpoint; returns the reply text."""
    body = dict(model=model, messages=messages, max_tokens=max_tokens)
    body.update(dict(temperature=0.0) if greedy else dict(temperature=0.7, top_p=0.9))
    req = urllib.request.Request(base_url + "/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="two-GB10 pair utilities")
    ap.add_argument("--check", action="store_true", help="preflight the pair and exit")
    ap.add_argument("--stop", action="store_true", help="stop the pair vLLM/ray")
    ap.add_argument("--serve", metavar="MODEL", help="serve MODEL across the pair")
    ap.add_argument("--lora", metavar="DIR", help="adapter dir to mount into --serve")
    a = ap.parse_args()
    if a.check:
        probs = check()
        if probs:
            sys.exit("\n".join("- " + p for p in probs))
        print("pair OK: this is the head, and the worker answers on the rail")
    elif a.stop:
        stop_serving()
    elif a.serve:
        url, name = serve(a.serve, lora=a.lora)
        print(f"endpoint: {url}  model name for requests: {name}")
    else:
        ap.print_help()

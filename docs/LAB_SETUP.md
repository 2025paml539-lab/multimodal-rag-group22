# Lab setup, step by step

**This project can only be run in the BITS lab.** Use your laptop for editing code and running offline tests.
Everything that produces a reported number runs on the lab's NVIDIA A100.

Every member does this once (about 90 minutes in your first session). Every command below was run and checked on 21 Nov 2026
unless it is marked **(not yet tested)**.

What we found in the lab:

| Item | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB, 80 GB |
| Driver | 535, CUDA 12.2 |
| Python in the notebook | 3.11 (conda) |
| Your limits | 30 GB disk, 14 GB memory, 6 CPUs, 1 GPU, 10 sessions of 2 hours (check your own account) |
| What survives a restart | your home folder `/home/jovyan` |
| What is wiped on restart | `/tmp` (we keep downloaded models there) |

---

## Step 1. Book a session

1. Open the Prayogshala portal (`prayogshala.bits-pilani.ac.in`) and log in.
2. Under **My Laboratories** find **PCAMZG516 Applied Generative AI** (marked *Booking Required*, *Remote*).
3. Click **Book a Slot**, choose a date and a time slot (IST), click **Confirm Booking**.
4. When your slot starts, the **Start Session** button under *Upcoming Lab Sessions* becomes active. Click it.

You see "Total Sessions 10" in the booking window. Do not waste sessions: do the preparation below before you book the second one.

## Step 2. Create your notebook (once only)

Start Session opens the **Kubeflow** dashboard. In the left menu click **Notebooks**, then **+ New Notebook**.

| Box | Value |
|---|---|
| Name | `g22-genai-a2` |
| Type | JupyterLab (already selected) |
| Image | leave the default (`jupyter-scipy`) |
| Minimum CPU | `6` |
| Minimum Memory Gi | `14` |
| Maximum Memory Gi | `15` (open *Advanced Options* under CPU/RAM) |
| Number of GPUs | `1` |
| GPU Vendor | `NVIDIA` |
| Workspace Volume | new volume, size **`30`** Gi, mount path `/home/jovyan` |
| everything else | leave as it is |

Click **LAUNCH**. Wait for the green tick (1–5 minutes), then click **CONNECT**.

Errors we met, and the fix:

| Message | Cause | Fix |
|---|---|---|
| Launch button is grey | a required box is empty | check Name and that the volume size is filled |
| `exceeded quota … requests.storage=50Gi … limited 32Gi` | disk quota is 32 GB per student | use `30` Gi, not 50 |
| `exceeded quota … requests.memory=16512Mi … limited 16Gi` | memory quota is 16 GB per student | use `14` Gi minimum and `15` Gi maximum |
| After a failed launch the volume name already exists | the failed attempt created the volume | remove the "New volume" row, click **+ Attach existing volume** and pick `g22-genai-a2-workspace` |

Do not click the trash icon on a notebook or on its volume. That deletes your files.

## Step 3. Open a terminal and check the machine

In JupyterLab click the **Terminal** tile (under *Other*) and paste:

```bash
nvidia-smi
free -h; nproc; df -h ~
python --version
echo "keep test $(date)" > ~/keep_test.txt; ls -la ~
```

Expected: an A100 80 GB, Python 3.11, `/home/jovyan` showing 30 GB.
Next session run `cat ~/keep_test.txt` to confirm your files are still there.

## Step 4. Install PyTorch for the GPU

The lab driver is old, so use exactly this build and **never upgrade it**:

```bash
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It downloads about 3 GB (10–15 minutes). Expected last line: `2.8.0+cu128 True NVIDIA A100-SXM4-80GB`.

## Step 5. Install Ollama in your home folder

The newest Ollama refuses the lab's driver (`NVIDIA driver too old … required 550 or newer`) and falls back to the CPU.
Version **0.12.6** works. It installs without admin rights:

```bash
mkdir -p ~/ollama012 && cd ~/ollama012
( curl -fL -o o.tgz https://github.com/ollama/ollama/releases/download/v0.12.6/ollama-linux-amd64.tgz ) && tar -xzf o.tgz && rm o.tgz
du -sh ~/ollama012
```

Start it, pull a model, and check it uses the GPU:

```bash
export OLLAMA_MODELS=/tmp/ollama_models
nohup ~/ollama012/bin/ollama serve > ~/ollama012/serve.log 2>&1 &
sleep 8; grep -i "inference compute" ~/ollama012/serve.log
~/ollama012/bin/ollama pull qwen3:8b
~/ollama012/bin/ollama run qwen3:8b "Say hello in one sentence." --verbose
~/ollama012/bin/ollama ps
```

Expected: `library=CUDA`, an answer in about one second, `eval rate` above 100 tokens/s, and **100% GPU** in the last table.
If it says `library=cpu`, you installed a newer Ollama by mistake.

Models for Part B (one family each; they are stored in `/tmp`, so they are downloaded again after a restart).
Session S2 (23 Nov): `qwen3:8b` and `deepseek-r1:14b` pulled successfully (5.2 GB and 9.0 GB). **`gpt-oss:20b` could not be pulled** (see the next section).

```bash
~/ollama012/bin/ollama pull qwen3:8b          # Qwen (tested)
# ~/ollama012/bin/ollama pull gpt-oss:20b   # FAILS on Ollama 0.12.6, see "gpt-oss-20b" below
~/ollama012/bin/ollama pull deepseek-r1:14b   # DeepSeek-R1-Distill-Qwen-14B (pulled OK in S2)
```

### gpt-oss-20b (found in S2)

`ollama pull gpt-oss:20b` with Ollama 0.12.6 fails with `pull model manifest: 412: The model you are attempting to pull requires a newer version of Ollama`.
The registry now asks for a newer client, and newer Ollama builds are the ones that reject the lab driver (535, CUDA 12.2).
The assignment requires one model from each family (Qwen, DeepSeek, gpt-oss-20b), so do not replace it with another family.
Allowed by the assignment (Part B rule 15): a hosted open-weight endpoint (for example Groq, Together, OpenRouter, Fireworks or Hugging Face Inference).
Record the provider, quantisation and hardware in the report. Keep the API key in an environment variable, never in code, a notebook or the repository.
Ask the instructor to confirm before relying on it. A newer Ollama on this driver was **not tested**.

## Step 6. Get the code

```bash
cd ~
git clone https://github.com/2025paml539-lab/multimodal-rag-group22.git
```

Update later with `cd ~/multimodal-rag-group22 && git pull`. The repository is public, so no login is needed to pull.
Pushing from the lab needs a GitHub token (create it on GitHub, never paste it into a file or a chat). Until you set one up,
download results with the JupyterLab file browser (right-click, Download).

## Step 7. Project packages

A virtual environment in your home folder, so a restart cannot remove it.

**Install PyTorch into the venv first.** In S2 the venv could not see the PyTorch from step 4, so `pip install -r requirements-lab.txt` downloaded
`torch 2.15.0` built for CUDA 13 plus about 3 GB of CUDA 13 libraries. That build cannot use this lab GPU (driver 535, CUDA 12.2).

```bash
python -m venv --system-site-packages ~/venv_rag
source ~/venv_rag/bin/activate
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
pip install -r ~/multimodal-rag-group22/requirements-lab.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected last line: `2.8.0+cu128 True NVIDIA A100-SXM4-80GB`. If it shows `2.15.0`, run:
`pip uninstall -y torch triton` and the `nvidia-*` and `cuda-*` packages that were installed, then repeat the torch line above.
Pip prints dependency "conflicts" for `kfp`, `mlflow-skinny` and `httpcore`. They belong to other lab tools and can be ignored.
Confirmed in S2: `2.8.0+cu128 True NVIDIA A100-SXM4-80GB`. The venv is about 7 GB.

`requirements-lab.txt` installed without errors in S2 (23 Nov). PyMuPDF, python-pptx and RapidOCR were then used on 4 sample files
(3 PDFs, 1 PPTX): text, table, image and OCR elements were all produced. Never add or upgrade `torch` there.

## Start of every session

```bash
cd ~/multimodal-rag-group22 && git pull
bash scripts/lab_start.sh qwen3:8b            # add more model names to pull them too
```

`scripts/lab_start.sh` starts Ollama on the GPU, pulls the models you name, and prints the GPU state.
It was used in S2 (23 Nov): Ollama started on the GPU (`library=CUDA`) and `qwen3:8b` was pulled.

## End of every session (do not skip)

1. Download or push anything you produced (results, index, notebooks). The lab is not a backup.
2. Copy big files (index, extracted images) to Google Drive.
3. Run `df -h ~` and make sure you are under 25 GB.
4. Close the browser tab. **Do not** use File → Shut Down or Log Out, and do not delete the notebook.
5. If you ran Ollama, nothing else to do; it stops with the notebook.

## Limits and rules

- Do not upgrade PyTorch or Ollama. They match this driver.
- Keep large model files in `/tmp` and nothing important there.
- Never put passwords, tokens or API keys in code, notebooks or the repository.
- Public documents only. No confidential or personal data.
- Each student works in their own notebook. Do not share logins.

## Not yet verified

- Whether a newer Ollama can run on the lab driver (needed for a local gpt-oss-20b).
- Whether the hosted endpoint for gpt-oss-20b is accepted by the instructor.
- Whether the notebook and GPU still work after your booked slot has ended.
- Streamlit: opening `.../proxy/8501/` on the notebook address returned `404 Not Found` in S2 (no `jupyter-server-proxy`). A Streamlit demo cannot be opened through the lab. Use a screen recording or run it on a laptop.
- Whether two sessions can be booked on the same day.
- Whether teammates can be added as contributors to one notebook (the Kubeflow menu has *Manage Contributors*).

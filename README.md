# DPO Preference Alignment — Base vs DPO (Coding Explanations)

A complete pipeline: dataset prep → DPO + LoRA fine-tuning → LLM-judge evaluation → FastAPI + Streamlit live demo.

Base model: `Qwen/Qwen2.5-1.5B-Instruct`
Domain: coding-explanation preference alignment

---

## STEP-BY-STEP GUIDE (Start to Finish)

### Step 0 — Prerequisites
- A GitHub account
- A Google account (for Colab, free GPU)
- A Hugging Face account (to permanently store your trained model)
- An OpenAI API key (for the LLM-judge evaluation step — needs a few dollars of billing credit)

### Step 1 — Push this project to your own GitHub repo
1. Create a new empty repo on GitHub, e.g. `dpo-alignment-project`
2. On your laptop, unzip this project folder
3. Open the folder in VS Code, open a terminal, and run:
```
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/dpo-alignment-project.git
git push -u origin main
```

### Step 2 — Local dev environment (optional but recommended)
Install Python 3.11 (not 3.13/3.14 — some ML packages don't have prebuilt wheels for very new Python versions yet).
```
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt
```
This lets you edit/test code locally. Actual model training still needs a GPU (Step 4).

### Step 3 — Prepare the dataset
```
python src/prepare_data.py
```
This downloads `HuggingFaceH4/ultrafeedback_binarized`, filters it to coding-related prompts, and saves:
- `data/train_prefs.jsonl` (450 examples)
- `data/eval_prefs.jsonl` (50 held-out examples)

Commit these to GitHub once generated.

### Step 4 — Train the model (in Google Colab — needs a free GPU)
1. Go to https://colab.research.google.com → New notebook
2. Runtime menu → Change runtime type → **T4 GPU**
3. In a cell:
```python
!git clone https://github.com/YOUR_USERNAME/dpo-alignment-project.git
%cd dpo-alignment-project
!pip install -r requirements.txt --quiet
!python src/train.py
```
4. Training takes ~35-45 minutes (108 steps). **Do not close the tab, switch tabs, or let your laptop sleep** while this runs.
5. When you see `Training complete.`, immediately (same cell block, same tab) save the model — see Step 5.

### Step 5 — IMPORTANT: Save the trained model immediately after training finishes
Colab wipes all files when the runtime disconnects. Upload the model to Hugging Face **right after training completes, in the same session**:
```python
from huggingface_hub import login
login()   # paste a token from https://huggingface.co/settings/tokens (Write access)
```
```
!hf upload YOUR_HF_USERNAME/dpo-coding-model dpo-coding-model
```
Verify it worked by visiting `https://huggingface.co/YOUR_HF_USERNAME/dpo-coding-model` — you should see the model files listed.

### Step 6 — Run evaluation (same Colab session, right after training)
```python
import os
os.environ['OPENAI_API_KEY'] = 'your-key-here'
```
```
!python src/evaluate.py
```
This generates responses from both models on the 50 held-out prompts, judges each pair with GPT-4o-mini, and prints your real win-rate with a 95% confidence interval. Results save to `data/evaluation_results.json`.

Commit this file back to GitHub:
```
git add data/evaluation_results.json
git commit -m "Add evaluation results"
git push
```

### Step 7 — Run the live demo (FastAPI + Streamlit)
This needs the trained model available locally or downloaded from Hugging Face. Easiest: run this in Colab too (or any machine with a GPU), in two separate terminals/cells:
```
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
```
streamlit run app/streamlit_app.py
```
Open the Streamlit URL, enter a coding question, and see both models' answers side by side.

---

## Project structure
```
dpo-alignment-project/
├── src/
│   ├── prepare_data.py   # dataset filtering + train/eval split
│   ├── train.py          # DPO + LoRA training
│   └── evaluate.py       # LLM-judge evaluation + win-rate
├── api/
│   └── main.py           # FastAPI backend
├── app/
│   └── streamlit_app.py  # comparison UI
├── data/                 # generated datasets + results (not pre-included)
├── requirements.txt
├── .gitignore
└── README.md
```

## Troubleshooting (issues already fixed in this code)
- **numpy/torch build errors** → caused by very new Python (3.13/3.14). Use Python 3.11.
- **`ModuleNotFoundError: trl`** → re-run `pip install -r requirements.txt` in the same Colab session; if it persists, Runtime → Restart session and reinstall.
- **`DPOTrainer.__init__() got an unexpected keyword argument 'tokenizer'`** → newer `trl` versions use `processing_class=` instead — already fixed in `train.py`.
- **`NotImplementedError: ... not implemented for 'BFloat16'`** → gradient scaler / fp16 dtype clash. Already fixed by setting `fp16=False` in `train.py`.
- **`bitsandbytes` 4-bit quantization ImportError** → run `pip install -U bitsandbytes`.
- **Model "disappeared" after training** → Colab wiped it because the runtime disconnected before you saved it. Always upload to Hugging Face immediately after training completes, in the same session — see Step 5.

## Resume bullet (fill in after Step 6)
"Built an end-to-end preference-alignment pipeline: fine-tuned Qwen2.5-1.5B-Instruct using Direct Preference Optimization (DPO) with LoRA, evaluated via LLM-as-judge, achieving a [XX]% win-rate over the base model on held-out coding-explanation prompts; deployed both models behind a real-time comparison API (FastAPI + Streamlit)."

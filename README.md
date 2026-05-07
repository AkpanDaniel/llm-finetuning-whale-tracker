

#  LLM Fine‑tuning – Blockchain Transaction Parser

Fine tuning FLAN‑T5‑small with LoRA to extract structured JSON data from raw blockchain transaction logs.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Black-html/llm-finetuning-whale-tracker/blob/main/Whale_tracker.ipynb)

Fine‑tune a small language model (FLAN‑T5‑small) with **LoRA** to turn messy blockchain transaction text into clean, structured JSON.

- **Goal:** Extract token, amount, sender, receiver, and date from strings like `"0x5Fbe sent 200,000 BTW to 0x1Ae on May 4 2026"`  
- **Output:** `{"token": "BTW", "amount": 200000, "from": "0x5Fbe", "to": "0x1Ae", "date": "2026-05-04"}`  
- **Why this matters:** Automate on‑chain analysis for whale tracking, transaction monitoring, or audit pipelines.

---

##  Features

- ✅ Synthetic dataset generator (30 examples – easily expandable)  
- ✅ Parameter‑efficient fine‑tuning with **LoRA** (only **0.44%** of parameters trained)  
- ✅ Uses **FLAN‑T5‑small** (runs on free T4 GPU in Colab)  
- ✅ Fully reproducible Colab notebook included  

---

##  Dataset

The notebook generates a toy dataset of 30 Ethereum‑like transaction statements:

| Input | Label (JSON) |
|-------|---------------|
| `0xabc sent 12345 BTW to 0xdef on May 6 2026` | `{"token": "BTW", "amount": 12345, "from": "0xabc", "to": "0xdef", "date": "May 6 2026"}` |

*You can replace the synthetic generator with real on‑chain logs from Etherscan or Dune.*

---

##  Model & PEFT (LoRA)

- **Base model:** `google/flan-t5-small` (80M params)  
- **PEFT method:** LoRA (r=8, alpha=32, dropout=0.1)  
- **Trainable params:** 344,064 (0.445% of all weights)  

Loss after 10 epochs: ~11.0 (expected due to tiny dataset). The pipeline is what matters – you can increase data size for real accuracy.

---

##  Training

Run the notebook in **Google Colab with a T4 GPU** (free).  
Training takes ~3–4 minutes.

```bash
# Install dependencies (inside Colab)
!pip install peft==0.10.0 transformers datasets accelerate torch``` 


## Example Inference
Input:


text
0xabc sent 12345 BTW to 0xdef on May 6 2026
Output (trained on 30 examples):

text
0xabc ne 12345
(The model did not converge; this is expected with such a small dataset. Increase to 500+ examples for usable output.)

## How to Use
Open the notebook in Colab / Jupyter / VS Code.

Run all cells – no extra setup required.

Modify the dataset – change tokens, dates, or add real transaction logs.

Increase num_train_epochs or use a larger base model (flan-t5-base) for better results.

Save your fine‑tuned model with peft_model.save_pretrained("./my_whale_tracker").

## Repository Structure
text
.
├── Whale_tracker.ipynb         # Full training + inference notebook
├── README.md                   # This file
└── requirements.txt            # Python dependencies
⚙️ Requirements
Python 3.10+

PyTorch, Transformers, Datasets, PEFT, Accelerate

Install with:

bash
pip install -r requirements.txt

 ## Limitations (Honest & Important)
Issue	Impact
Only 30 training examples	Model didn't learn the JSON format
FLAN‑T5‑small is weak for structured generation	Use flan-t5-large or lora with bigger base
Loss plateaued at ~11	More data + more epochs needed
This is a demonstration of the workflow, not a production model.
With 500+ examples and a larger LLM, you would get clean JSON extraction.

🔮 Future Improvements
Use real on‑chain data from Etherscan/Dune (pandas → dataset)

Expand to multi‑task: classify tx type (swap/transfer/mint)

Compare LoRA vs QLoRA vs full fine‑tuning

Deploy as a simple API (FastAPI + Hugging Face TGI)

## License
MIT – feel free to use and adapt for your own whale‑tracking tools.

## Acknowledgements
Hugging Face for Transformers & PEFT

Google for FLAN‑T5

The open‑source community for excellent LLM tutorials

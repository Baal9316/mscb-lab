# Fresh-Clone Verification

This procedure verifies the repository works from a **clean clone** with only the
README as guidance (no verbal help). It is the final sanity check before
submission.

## Prerequisites

- Git, Python ≥ 3.10
- Network access to the class services (`dobolyi.com` ports 9001–9005)
- (For `.pptx` uploads only) LibreOffice — see README "Supported formats"
- An **authorized class API key** of your own. Never copy anyone else's `.env`.

## Steps

```bash
git clone <repository-url>
cd mscb-lab
python3 -m venv .venv
source .venv/bin/activate          # use .venv/Scripts/activate on Windows
pip install -r requirements.txt
cp .env.example .env
# edit .env: put YOUR authorized class API key in CLASS_API_KEY
# (keep all endpoint URLs at their .env.example defaults)
```

Then:

```bash
pytest                                # 1) test suite passes
python -m app                         # 2) app starts
```

Open the printed local URL (<http://127.0.0.1:7860>).

## Verification checklist

| # | Check | How to verify |
|---|---|---|
| 1 | README instructions are sufficient | Follow README top-to-bottom without other help |
| 2 | Dependencies install | `pip install -r requirements.txt` completes |
| 3 | Test suite passes | `pytest` → all tests pass (final verification run currently reports `269 passed`) |
| 4 | App starts | `python -m app` → Gradio UI at 127.0.0.1:7860 |
| 5 | Document uploads | Document Manager → upload a course PDF → "Uploaded … pages" |
| 6 | QA works | Ask a Question → a course question → grounded answer appears |
| 7 | Visual source image appears | Ask about a slide with a diagram/meme → supporting slide image shown with citation |
| 8 | Quiz generates | Practice Quiz → Generate Quiz → questions + radio options appear |
| 9 | Quiz answered with radios | Click A/B/C/D per question (no JSON entry) |
| 10 | Submit works | Submit Quiz → per-question ✓/✗ and score appear |
| 11 | Show Answers works | Show Answers → correct answer, explanation, citation, excerpt, image |
| 12 | No credentials appear | Screens of the app, logs, and `git status` show no API key / Bearer / `.env` values |

## Rules

- Use **only your own** authorized class API key; never expose or share anyone
  else's `.env`.
- If any step fails, report the exact command + error; do not silently work
  around it.

## Notes

- The evaluation benchmark is **frozen**: do not rerun or modify
  `evaluation/results.json` / `results.csv`.
- Results are specific to the fixed 8-question set over the Week 2 deck.

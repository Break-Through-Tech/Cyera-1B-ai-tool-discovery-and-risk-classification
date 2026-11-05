# Dataset B scripts

Requires Python 3.10+. Put both Python files in your project's scripts/ folder.
Run commands from the project root. Paths resolve from your working directory,
not the scripts folder.

## Install

    python -m pip install pandas google-genai python-dotenv

## 1. Map the full catalog (no API calls)

    python scripts/map_dataset_b.py --input data/processed/kaggle_clean.csv

Creates data/processed/dataset_b/:
- dataset_b_master.csv: all rows, exactly the 14 Dataset B fields.
- dataset_b_context.csv: original metadata and provisional-label provenance.
- mapping_summary.json: counts and input fingerprint.

No rows are dropped. Existing IDs and descriptions are preserved. is_ai_tool=true
is provisional catalog membership, not verified ground truth. Other new fields
are blank (unknown). No invented vendors, retention policies, approvals or risks.
Existing outputs are protected unless --overwrite is explicitly supplied.

## 2. Preview the sample (no API calls)

    python scripts/sample_review_dataset_b.py --dry-run

Default: 500 candidates, seed 42, proportional category allocation with a minimum
of 10 per available category. Smaller strata contribute all available rows.
Excludes missing descriptions/names, invalid domains, sites flagged dead, non-AI
candidates, and duplicate normalized-name + identity pairs. Short descriptions
are retained. Shared hosts use full URL identity. Matching names on different
domains are not automatically merged; manual product-identity review is needed.

The provided CSV produced 14,966 eligible rows, sampled into 500 candidates.
Do not interpret historical site flags as current availability checks.

## 3. Configure Gemini locally

Create .env in your project root:

    GEMINI_API_KEY=your_key_here

Never commit .env. The scripts never print the key. Environment variables take
precedence over .env. Use --env-file to select another file.
Sampled names, domains, descriptions, tasks, keywords and original categories
are sent to Google's Gemini API when running without --dry-run.

## 4. Try one batch, then resume

    python scripts/sample_review_dataset_b.py --max-calls 1
    python scripts/sample_review_dataset_b.py

The first command processes at most one SDK request (normally 20 candidates).
It exits with code 2 when work remains, intentionally. The second resumes.
The default model is gemini-3.8-flash; override with --model MODEL_ID if needed.
Use the exact same model for cache reuse. Default full run is 25 requests for
500 records at batch size 20, before retries. This is not a dollar-cost estimate.
The --max-calls limit counts SDK invocations, including script retries, not tokens
or any transport retries performed internally by the SDK.

Results are cached per record + model + prompt + schema. Changing a record or
prompt invalidates that cache entry. Use a NEW --out-dir if input files, seed,
sample size or category minimum change. Do not run concurrent processes against
the same output directory. Cache should be kept until final review is complete.

## Outputs in data/processed/dataset_b/sample/

- candidates_dataset_b.csv: the original 500-row sample in Dataset B schema.
- candidates_context.csv: sampled source metadata and flags.
- excluded_rows.csv: preprocessing exclusions and reasons.
- sample_manifest.json: input fingerprints and selected IDs.
- reviewed_dataset_b.csv: every sampled row, including pending/rejected rows;
  category and AI status updated only where a valid response exists.
- accepted_dataset_b.csv: model-accepted usable AI candidates, NOT human-verified.
- review_results.csv: accepted/rejected/needs_review/pending, reasons and function.
- run_summary.json: status counts and sanitized API errors.
- usage.jsonl: successful responses' token-usage metadata.
- cache/: resumable, validated per-record results.

Dry runs only produce candidate/exclusion/manifest files, not new review results.
API failures or call limits leave pending rows; they are never marked accepted.
An interrupted process retains completed per-record cache files; rerun to rebuild
review outputs. The script does not automatically replace rejected candidates or
force the accepted count to 400. Review the results, then select a final diverse
300–400 records. Keep related products together when later splitting train/test.

Description usability is judged only from description. Categorization and AI
status use the additional supplied metadata. Original category is a hint.
Uncertain cases are held for review. Vendor/security claims are not researched.
Approval, retention, SSO, vendor and risk fields remain unchanged for later work.
Synthetic governance attributes must be labeled separately when added.

## Validation performed

Mapping and 500-row dry-run tested on the supplied 16,762-row CSV.
Offline mocked responses tested API parsing, caching/resume, rejected and
uncertain routing, and Dataset B output schema. No live Gemini request was made.

API implementation reference:
https://ai.google.dev/gemini-api/docs/generate-content/structured-output

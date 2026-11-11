# Cyera - AI Tool Discovery & Risk Classification

## Current project status — November 9, 2026

The [vendor-reconciled subset](data/processed/final/dataset_b_final.csv) contains 697 reviewed records and is separate from the full classification source. See its [provenance](data/processed/final/dataset_b_final.provenance.json): vendor research is model-interpreted; governance policy fields are synthetic; six classification labels are unresolved.

The [corrected-label export](docs/corrected_labels.md) applies the approved conflict-review recommendations: **16,740 AI, 516 conventional and 36 unresolved records**. The [full-data binary training guide](docs/corrected_full_binary.md) documents the new grouped splits and retrained Logistic Regression model; the source-membership file below preserves original dataset assignments.

The current labeling dataset is [all_labeled.csv](data/processed/source_membership/all_labeled.csv): **16,762 AI-source records labeled 1 and 530 conventional-source records labeled 0**. All 17,292 records are retained; potential cross-source conflicts are flagged. These are source-membership labels, not independently verified labels. See the [labeling guide](docs/source_membership_labels.md).

The new **description-only Logistic Regression classifier** is in `artifacts/corrected_full_binary/`: 16,674 usable records grouped into fresh train/validation/test splits. Test macro F1 is 0.9243; 70 of 73 conventional test examples are recognized. Labels remain primarily source-assigned and accuracy is not independently verified. See [results, limitations and commands](docs/corrected_full_binary.md). Earlier baseline/embedding artifacts remain historical experiments. No Jev comparison has been run.

The [governance risk engine](docs/governance_risk.md) is available separately, with explained scores and explicitly synthetic policy scenarios. It does not depend on the superseded curated v2 dataset or its fitted models.

PRs #12, #13 and #17 are integrated in this consolidation; the risk-engine changes from #16 are included independently. PRs #14 and #15 are superseded by the team-data direction and retained in Git history rather than added to main.

---


> 💡 **Note for the team:** This is just a template. Update the above title with your AI Studio Challenge Project name. Remove all guidance notes and example text in this template and populate this README with your own content. You can work on this README throughout AI Studio, and get feedback from your AI Studio Coach and Challenge Advisor before finalizing it.  

---

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
|  Shubham Arya   | @Arya25 | Challenge Advisor            |
|  Aram Ramos  | @aramseries  | AI Coach  |
|  Forum Shah    | @shahforum  | Member                 |
|  Sheila Aquino  |  @shem3ra  |  Member  |
|  Sadit Hussain  |  @sadithussain  |  Member  |
|  Min Thaw Zin   |  @Min-13     | Member   |
| Abigail De Leon | @abigail-deleon | Member |
| Joshua Hu | @JoshuaCongHu | Member — Exploratory Data Analysis (EDA) |
| Abdulfattah Ahmed | @abdulfattaha42 | Member |
---

## 🎯 **Project Highlights**

**Example:**

- Developed a machine learning model using `[model type/technique]` to address `[challenge project task]`.
- Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
- Generated actionable insights to inform business decisions at `[host company or stakeholders]`.
- Implemented `[specific methodology]` to address industry constraints or expectations.

---

## 👩🏽‍💻 **Setup and Installation**

```bash
git clone https://github.com/Break-Through-Tech/Cyera-1B-ai-tool-discovery-and-risk-classification.git
cd Cyera-1B-ai-tool-discovery-and-risk-classification
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All datasets are already in `data/`.

**Run the EDA notebook**

```bash
jupyter notebook notebooks/eda.ipynb          # open interactively
# or re-run it headless from the repo root:
jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
```

---

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

> EDA conducted by **Joshua Hu ([@JoshuaCongHu](https://github.com/JoshuaCongHu))**. Full analysis with charts: [`notebooks/eda.ipynb`](notebooks/eda.ipynb)

**Datasets**

| Dataset | File | Size | Contents |
|---|---|---|---|
| AIToolBuzz AI tools (Kaggle), cleaned | `data/cleaned_kaggle.csv` | 13,701 tools (from 16,762 raw) | Name, category, primary task, short description, website, country, year founded |
| Dataset B: AI Tool Metadata (incomplete) | `data/dataset_b_ai_tool_metadata_incomplete.csv` | 13,701 tools | Kaggle mapped to the `VISION.md` schema; governance fields still blank |
| Dataset A: NovaTech usage (synthetic) | `data/novatech_usage (1).csv` | 30,000 events, 500 employees | Sept 2026 app activity across 6 departments, 21 apps, incl. bytes uploaded/downloaded |
| Dataset A: Summit Financial usage (synthetic) | `data/summit_financial_usage (1).csv` | 25,120 events, 400 employees | Sept 2026 app activity across 6 departments, 22 apps, incl. bytes uploaded/downloaded |

**Key insights**

* **Kaggle metadata is sparse beyond the core fields.** Year Founded is missing for 64% of tools, Country 59%, industry and technologies about 52%, and Keywords 40%. Name, description, category and website are the reliable features.
* **Descriptions are short and categories are noisy.** The median description is 8 words, and 17% of tools have 5 words or fewer. The 14 Kaggle categories are imbalanced (Media & Content Creation is 26%) and often wrong for well-known tools: ChatGPT is listed as "Communication And Support" and Grammarly as "Security And Privacy". They need mapping to the `VISION.md` taxonomy and a hand-labeled evaluation set before they can be used as training labels.
* **Dataset B has no negative class.** Every row is `is_ai_tool = True`, so AI-vs-not-AI detection needs non-AI SaaS examples added.
* **A known-tool domain lookup isn't enough.** Matched against the usage logs it gets precision 0.67 and recall 0.60. Some non-AI platforms match because AI add-ons share their domain (`slack.com` → "Slack GPT", `zoom.us` → "Zoom IQ"). Some real AI tools are missed because of stale or path-based domains or because they're absent from Kaggle: Cursor, GitHub Copilot, Midjourney and DocuMind AI. That supports the embedding/LLM approach for unseen tools.
* **The synthetic logs contain a clear shadow-AI case.** DocuMind AI appears only in Summit Financial's Finance department (84 of 129 employees) and isn't in the metadata. It uploads 7.95 MB per event, 2.6× every other app, and it's the only app with uploads over 10 MB. This is the `VISION.md` §4 scenario the pipeline should surface, and it shows upload volume relative to the baseline should be a core risk feature.
* **The synthetic usage is very uniform.** Event types, session durations, upload sizes, hours and weekdays are all flat, and every employee uses AI tools. Signal comes from the department × application mix and from outliers against that flat baseline.

---

## Embeddings and semantic similarity (issue #7)

A frozen, revision-pinned MiniLM sentence-transformer now supports AI detection, category prediction and cosine-similarity matching to training applications. The same held-out applications and source-review exclusions are used as in issue #6. Validation selects the reviewed-only embedding head: test macro F1 is 0.939 versus 0.770 for TF-IDF; category test accuracy is 54.3% versus 28.3%. These are weak-label development results with only four non-AI examples per evaluation split.

```bash
python -m pip install -r modeling/requirements-embeddings.txt
python modeling/predict_embeddings.py --name "ExampleCodeHelper" --description "An AI coding assistant that suggests completions and helps debug software."
```

See [the embedding comparison guide](docs/embedding_classification.md) and [runnable notebook](notebooks/embedding_classification.ipynb). The first use downloads the public encoder; classifier heads, reports and the training-only reference index are included in `artifacts/embeddings/`.

---

## Baseline classification models (issue #6)

Train reproducible TF-IDF + Logistic Regression baselines for AI/non-AI detection and an initial business-purpose category classifier. Both reviewed-only and full-catalog variants are evaluated against a majority-class baseline. Validation macro F1 selects the reviewed-only binary model; the labels remain weak supervision.

```bash
python -m pip install -r modeling/requirements.txt
python modeling/train_baseline.py --out artifacts/reproduced-baseline
python modeling/predict.py --name "Example Writing Assistant" --description "An AI assistant that generates marketing copy from prompts."
python -m unittest discover -s tests -v
```

The [source/code review](docs/source_review_2026-11-04.md) checks the negative seeds, a category sample and original errors, withholds four uncertain training identities, and preserves evaluation labels. Negative labels describe conventional primary purpose rather than an absence of AI capabilities.

See [the baseline guide](docs/baseline_classification.md) for results, settings, prediction output and limitations, or use [the runnable notebook](notebooks/full_classification_baseline.ipynb). Saved models, reports and test predictions are in `artifacts/baseline/`. Detection, category and governance risk are separate tasks; this baseline does not calculate risk.

---

## Classification dataset preparation (issue #5)

The full available-data preparation pipeline accounts for all 16,762 cleaned catalog records and combines eligible weak AI examples with the team's 30 non-AI examples recovered from Git history. It produces 14,792 applications: 14,692 training, 50 validation and 50 test rows.

`is_ai_tool` is the binary detection target (`1` = AI-focused tool, `0` = conventional primary application purpose). Business-function categories are separate; no risk scores or vendor security claims are inferred. Labels remain weak supervision, not independently verified ground truth.

```bash
python3 preprocessing_scripts/prepare_full_classification.py
python3 -m unittest discover -s tests -v
```

Model-reviewed positives and team negatives form the held-out sets; unreviewed catalog labels are used for training only. Connected names, hostnames and identical descriptions stay together, and held-out-related provisional records are quarantined. Generated CSVs, provenance, input hashes and the complete catalog audit are in `data/processed/classification_full/`.

See [the full dataset guide](docs/full_classification_dataset.md) for exclusions, split policy and limitations. The [335-row starter set](docs/classification_dataset.md) is retained for reference. The older EDA paths and counts elsewhere in this README describe a previous data revision. Baseline training is separate work under issue #6.

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.

## Integrated enterprise discovery reports

The [discovery reporting backend](docs/discovery_reports.md) now combines corrected catalog detection, optional description-model predictions, company activity and explained governance priority in one JSON report. Unknown and ambiguous applications remain visible in the review queue. Included NovaTech and Summit reports use synthetic activity; fictional policy overlays are explicitly marked. This is not yet a dashboard or RAG assistant.

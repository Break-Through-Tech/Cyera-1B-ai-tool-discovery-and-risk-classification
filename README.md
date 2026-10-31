# Cyera - AI Tool Discovery & Risk Classification

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

# Internet review of conflicting application records

**Update:** The user-approved recommendations have now been applied to a separate [corrected-label export](corrected_labels.md). Original source labels remain unchanged; no training run has occurred. The review-layer descriptions below describe the preserved evidence records.


Reviewed on **November 5, 2026**. This is an assistant source review, not independent human verification. The original source-membership labels and model artifacts are unchanged. No paid model API or classification model supplied the decisions.

The research examined all **201 flagged records**, including both sides of each possible conflict, using their original product URLs from the context files—not just bare domains. A total of **203 unique research URLs were attempted**, including alternate official documentation for blocked or outdated links. Some requests failed, some pages were accessible only through official search-result excerpts, and some successful pages did not resolve identity. A successful HTTP response alone does not verify a label. Raw downloaded HTML/text stays outside the repository; paraphrased findings, source URLs, retrieval method, date and extracted-text fingerprints are recorded here. Fingerprints cannot reconstruct a changed page.

## Recommendations

| Original source | Recommend 0 | Recommend 1 | Unresolved | Total |
| --- | ---: | ---: | ---: | ---: |
| Conventional source | 71 | 13 | 18 | 102 |
| AI source | 17 | 64 | 18 | 99 |
| Total | 88 | 77 | 36 | 201 |

These are **record counts**, not unique application counts. The 102 conventional records include 99 distinct names. Recommendations refer to primary product purpose under the project's agreed definition, rather than the presence of any AI capability. They are reasoned interpretations of product material, not vendor-certified binary labels. Official descriptions can be marketing claims; functionality was not executed or independently tested.

- `recommendations.csv`: all 201 records, proposed labels/scopes, evidence and rationale.
- `non_ai_recommendations.csv`: simpler view of the 102 conventional-source records.
- `summary.json`: counts, input/output fingerprints and review scope.

A blank `recommended_label` means unresolved, not 0. `source_label` is the untouched original assignment. A recommendation of 0 does **not** mean AI-free. A positive recommendation can describe a specifically linked AI feature whose platform stays conventional. Forty-nine records have a proposed scope/name change; those changes must be reviewed and original descriptions checked before these records can enter a training-ready dataset. Do not copy the recommended labels wholesale onto unchanged broad product names. Missing descriptions still need repair, and duplicate/related identities must still be grouped for splitting.

## Clear conventional-source label changes proposed

- **Feedly → 1:** Official product material describes AI-based threat-intelligence collection and analysis. [Official evidence](https://feedly.com).
- **DeepL → 1:** Official product material describes neural-machine translation and AI language services. [Official evidence](https://deepl.com).
- **Grammarly → 1:** Official product material describes AI writing guidance and text generation. [Official evidence](https://grammarly.com).
- **ChatGPT → 1:** Official product material describes a conversational assistant for reasoning, writing and task assistance. [Official evidence](https://openai.com/chatgpt/overview/).
- **ElevenLabs → 1:** Official product material describes model-generated speech and voice agents. [Official evidence](https://elevenlabs.io).
- **Akismet → 1:** Official product material describes machine-learning spam classification. [Official evidence](https://akismet.com).
- **Auphonic → 1:** Official product material describes AI-based audio enhancement and speech processing. [Official evidence](https://auphonic.com).
- **Jasper → 1:** Official product material describes AI agents for marketing content/workflows. [Official evidence](https://jasper.ai).
- **Otter → 1:** Official product material describes AI meeting transcription and summaries. [Official evidence](https://otter.ai).
- **You → 1:** Official product material describes generative AI research and assistants. Official support search result describes a business GenAI platform. [Official evidence](https://support.you.com/kb/s/article/Getting-started-for-Enterprise-users).
- **Stability AI → 1:** Official product material describes generative image and media models. [Official evidence](https://stability.ai).
- **Midjourney → 1:** Official product material describes generating images and videos from prompts. [Official evidence](https://docs.midjourney.com/hc/en-us/articles/33329261836941-Getting-Started-Guide).
- **LangChain → 1:** Official product material describes AI agent and language-model application infrastructure. [Official evidence](https://langchain.com).

## Examples of why domain-only conflict flags can be misleading

- HubSpot's CRM/Marketing Hub records describe a broad conventional workflow, whereas the AI catalog links to Campaign Assistant and the agent platform. Keep their product scopes distinct.
- ClickUp's work-management platform and the separately linked Brain assistant support different scoped labels.
- GitHub Enterprise and the RootKit AI project hosted on GitHub are not the same application. Repository content was read only; no code was downloaded or executed.
- Lovefield and Magika are different Google-hosted projects: an in-browser relational database versus model-based file detection. Their common hosting domain is not a shared product identity.
- The conventional record named JobPotato describes and links to Tally. It needs an identity correction before use. A separate hosted Tally form advertises an AI pitch generator; this does not make the form-building platform identical to that generator.
- The two Oracle records point to different organizations/hostnames. Neither a name match nor a failed TLS request supports treating them as the same product.

## Boundary cases and unresolved records

Vendor portfolios such as Microsoft, Oracle, Zoho, JetBrains, Freshworks and Intuit need a specific product scope. Broad or changed platforms such as Airtable, Notion, Picsart, Pixlr, Dialpad and Podium need a clearer distinction between conventional workflow and AI-focused service. Missing listings, legacy names, a Discord invite, an affiliate page or a general integration directory may not verify a current application's identity and capabilities.

Wolfram Alpha is a definition-sensitive case: its official FAQ distinguishes core algorithmic computation from traditional AI. This review recommends conventional primary purpose, without claiming the system has no AI-related techniques. Akismet is recommended AI-focused because its official page identifies machine-learning classification as the central spam-filtering mechanism. CrowdStrike remains unresolved at vendor/platform scope rather than being assigned a label solely from AI-native branding. These boundaries should be ratified for the benchmark, not treated as universal truths.

## Before applying

These recommendations are a review layer only. The source file still contains the requested 1/0 labels for every record. If approved, create a separate training-ready version that applies supported corrections, makes feature names/descriptions consistent, sets aside unresolved records, deduplicates and groups related applications. Do not change the frozen source export. Review evidence can change as products change. Independently review final test labels before describing model performance as verified accuracy.

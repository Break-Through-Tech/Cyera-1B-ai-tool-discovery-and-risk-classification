# Source and code review — November 4, 2026

This review uses official product information and the repository code. It covers all 30 negative seed descriptions, a deterministic sample of 12 reviewed AI training examples (one per category), and all three errors in the original selected-model test predictions: 43 distinct applications. It is a source-based review, not independent human verification or a fresh benchmark.

## Outcome

- Retain the 30 negative seed labels only for their documented conventional primary-purpose descriptions. This does not establish that the products lack AI features or that a usage event did not involve AI.
- Eight of the 12 sampled AI records have corroborating current product/function evidence under their recorded scope. Identity changes and coarse category choices are documented below.
- Withhold four training identities: Breve AI, Trendhopper, Craveo and AgentProd. Uncorroborated or stale metadata is not converted into negative labels.
- Retain the original labels for Power BI, GitHub Enterprise and MostlyAI. The three original selected-model errors remain model errors under the documented benchmark scope.
- Preserve all prepared dataset CSVs and evaluation labels. Apply exclusions as a versioned model-training overlay, covering all rows in the affected identity groups.
- Refit with the same fixed model settings and report the results, including decreases. No test-based threshold tuning or relabeling is used.

## Four training exclusions

| Record | Decision and evidence |
|---|---|
| Breve AI (`tool_01528`) | Official endpoint was inaccessible in this review; indexed directories were insufficient primary evidence. This is not a finding that the historical AI label was false. [source 1](https://breveai.com/) |
| Trendhopper (`tool_00986`) | Current official product generates SEO articles; stored trend-forecasting description and Data Analytics category are not corroborated. Needs a versioned metadata refresh, not an automatic non-AI relabel. [source 1](https://www.trendhopper.ai/) |
| Craveo (`tool_00072`) | Domain redirects to a pivot announcement; the former food-ordering AI functionality is not corroborated by the current landing page. The linked business endpoint was unavailable. [source 1](https://remy.craveo.in/); [source 2](https://www.craveo.in/) |
| AgentProd (`tool_00289`) | The catalog domain currently resolves to a domain-sale page. No current application identity was corroborated; do not turn a parked domain into a negative application example. [source 1](https://www.agentprod.com/) |

The unavailable Breve AI endpoint is a retrieval limitation, not proof that a historical product did not exist. AgentProd currently resolves to a sale/parking page; the report does not infer the former application's capabilities from that page. The Craveo business endpoint was unavailable after its current landing page redirected to a pivot notice.

## Complete negative-seed review

Every row below is retained as a scoped conventional-purpose example, not as an assertion of an AI-free product. Evidence may document a particular workflow rather than the entire platform.

| Application | Source-supported function and scope |
|---|---|
| Slack | Channels, messaging, file sharing and collaboration. AI features also appear on this page. [Official source](https://slack.com/features) |
| Salesforce Sales Cloud | CRM records and sales pipeline workflows; the current product also incorporates AI. [Official source](https://www.salesforce.com/sales/cloud/guide/?bc=OTH) |
| Jira | Project and work-item tracking. [Official source](https://www.atlassian.com/software/jira/features/project-tracking-software) |
| Workday | HR, workforce, payroll and reporting functionality, with AI capabilities also advertised. [Official source](https://www.workday.com/en-us/products/human-capital-management.html) |
| Zoom | Meetings and collaboration remain identifiable functions; built-in AI means generic usage is ambiguous. [Official source](https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0060423) |
| Confluence | Knowledge pages, documentation and collaboration; Rovo AI is also offered. [Official source](https://www.atlassian.com/software/confluence/features) |
| Microsoft Teams | Meeting chat, messages and collaboration are documented conventional functions. [Official source](https://learn.microsoft.com/en-us/microsoftteams/manage-meeting-chat) |
| SAP S/4HANA | ERP system covering finance, manufacturing and procurement functions. [Official source](https://learning.sap.com/products/s4hana-cloud) |
| NetSuite | ERP, accounting, inventory and order-management functions; AI is also part of the current product. [Official source](https://www.netsuite.com/portal/products/erp.shtml) |
| ServiceNow | IT service-management workflows; this supports the ITSM description, not a platform-wide absence of AI. [Official source](https://www.servicenow.com/docs/r/it-service-management/r_ITServiceManagement.html) |
| HubSpot Marketing Hub | Marketing Hub lead generation and marketing automation. The broader suite includes AI products. [Official source](https://legal.hubspot.com/hubspot-product-and-services-catalog/products) |
| Asana | Organizing, scheduling and managing project tasks. [Official source](https://help.asana.com/s/article/managing-projects) |
| Trello | Boards, lists and cards for visual work management. [Official source](https://trello.com/guide/trello-101) |
| Box | File storage, sharing and content collaboration. [Official source](https://www.box.com/cloud-content-management) |
| Dropbox | Cloud file synchronization, storage and sharing. [Official source](https://www.dropbox.com/features/sync) |
| DocuSign | Electronic signatures and agreement workflows; AI agreement analytics are a separate capability on the same product page. [Official source](https://www.docusign.com/en-gb/products) |
| Okta | Workforce application access through single sign-on. [Official source](https://www.okta.com/products/single-sign-on-workforce-identity/) |
| GitHub Enterprise | Developer collaboration and software-delivery platform that also provides AI integrations. [Official source](https://docs.github.com/en/enterprise-cloud@latest/admin/overview/about-github-for-enterprises) |
| GitLab | Repositories, merge requests, CI/CD and application lifecycle workflows; Duo is separately listed. [Official source](https://docs.gitlab.com/user/) |
| PagerDuty | Incident response, notification and on-call workflows. [Official source](https://docs.pagerduty.com/incident-management) |
| Splunk Enterprise | Search, transform, analyze and visualize operational data; machine-learning capabilities are also listed. [Official source](https://help.splunk.com/en/splunk-enterprise) |
| Tableau | Analytics and interactive data visualization. [Official source](https://help.tableau.com/current/tableau/en-us/tableau_product_overview.htm) |
| Power BI | Business analytics, reporting, visualization and sharing. Copilot availability does not turn every report-viewing event into confirmed AI use. [Official source](https://learn.microsoft.com/en-us/power-bi/fundamentals/power-bi-overview) |
| Coupa | Procurement service sheets and invoice workflows; evidence covers this conventional workflow, not every Coupa feature. [Official source](https://docs.coupa.com/en/supplier-documentation/coupa-for-suppliers/the-coupa-supplier-portal-or-csp/features-and-processes-in-the-coupa-supplier-portal/service-sheets/invoicing-for-services-in-the-csp) |
| Concur Expense | Expense submission, review and reimbursement workflows. [Official source](https://www.concur.com/products/concur-expense) |
| BambooHR | HR workflows and employee-management software. [Official source](https://www.bamboohr.com/platform/) |
| Greenhouse | Applicant tracking and recruiting workflow; the current ATS also advertises AI capabilities. [Official source](https://www.greenhouse.com/ats) |
| Zendesk | Customer-support ticket management; built-in AI means the whole platform cannot be described as AI-free. [Official source](https://www.zendesk.com/service/ticketing-system/) |
| Mailchimp | Marketing platform covering email, audience tools and automation; AI tools are also explicitly listed. [Official source](https://mailchimp.com/marketing-platform/) |
| Smartsheet | Project, portfolio and process management. [Official source](https://www.smartsheet.com/content/smartsheet-faqs) |

## AI/category spot-check

The sample was selected before checking websites by taking the lowest source_record_id in each category among model-reviewed training positives. Category judgments are interpretation within the project taxonomy; the review does not certify every one of the 305 model-generated category labels.

| Application | Original category | Finding / action |
|---|---|
| Breve AI | AI Infrastructure / Model Platforms | Official endpoint was inaccessible in this review; indexed directories were insufficient primary evidence. This is not a finding that the historical AI label was false. Action: `exclude_training`. [Official endpoint](https://breveai.com/) |
| Code Fundi | Coding / Developer Tools | Current documentation supports AI-enabled codebase search/research and developer tooling. Positioning has evolved from the older debugging description. Action: `keep`. [Official endpoint](https://docs.codefundi.app/docs/introduction) |
| LearningStudioAI | Content Generation | The CourseAI domain documents the LearningStudioAI API, resolving the apparent name/domain mismatch. AI course content generation supports Content Generation. Action: `keep`. [Official endpoint](https://courseai.co/docs) |
| Tovie | Customer Support | AI voice agents and contact-centre workflows support the AI label and Customer Support category. Action: `keep`. [Official endpoint](https://tovie.ai/) |
| Trendhopper | Data Analytics | Current official product generates SEO articles; stored trend-forecasting description and Data Analytics category are not corroborated. Needs a versioned metadata refresh, not an automatic non-AI relabel. Action: `exclude_training`. [Official endpoint](https://www.trendhopper.ai/) |
| ChatSpot | General AI Assistants | The official domain redirects to Breeze Assistant. AI assistant status is supported; General AI Assistants remains a coarse taxonomy choice for the assistant rather than a label for generic HubSpot usage. Action: `keep`. [Official endpoint](https://chatspot.ai/) |
| DraGAN | Image / Video Generation | Authors document generative image manipulation, supporting AI and Image / Video Generation. Action: `keep`. [Official endpoint](https://vcai.mpi-inf.mpg.de/projects/DragGAN/) |
| Metaview | Meeting / Transcription Tools | The current interview-notes product supports AI notetaking. Meeting / Transcription describes the stored feature, not the entire broader recruiting suite. Action: `keep`. [Official endpoint](https://www.metaview.ai/interview-notes) |
| Voicemod | Other | Official page identifies an AI voice changer. Other is a taxonomy fallback because this project has no dedicated audio category. Action: `keep`. [Official endpoint](https://www.voicemod.net/) |
| Craveo | Productivity | Domain redirects to a pivot announcement; the former food-ordering AI functionality is not corroborated by the current landing page. The linked business endpoint was unavailable. Action: `exclude_training`. [Official endpoint](https://remy.craveo.in/) |
| IAsk | Research / Search | Current AI question-answer/search experience supports AI and Research / Search. Action: `keep`. [Official endpoint](https://iask.ai/) |
| AgentProd | Sales / Marketing | The catalog domain currently resolves to a domain-sale page. No current application identity was corroborated; do not turn a parked domain into a negative application example. Action: `exclude_training`. [Official endpoint](https://www.agentprod.com/) |

## Original selected-model errors

| Application | Supplied label | Prediction | AI model score | Review conclusion |
|---|---:|---:|---:|---|
| Power BI | 0 | 1 | 0.5042 | Keep original benchmark label under the documented primary-purpose scope; this remains a model error. [Evidence](https://learn.microsoft.com/en-us/power-bi/fundamentals/power-bi-overview) |
| MostlyAI | 1 | 0 | 0.4737 | Keep original benchmark label under the documented primary-purpose scope; this remains a model error. [Evidence](https://mostly.ai/docs/concepts/what-is-synthetic-data) |
| GitHub Enterprise | 0 | 1 | 0.5162 | Keep original benchmark label under the documented primary-purpose scope; this remains a model error. [Evidence](https://docs.github.com/en/enterprise-cloud@latest/admin/overview/about-github-for-enterprises) |

Microsoft documents Power BI as business analytics and separately documents [Copilot](https://learn.microsoft.com/en-us/power-bi/create-reports/copilot-introduction). GitHub documents both conventional enterprise development workflows and AI integrations. MOSTLY AI explicitly documents training generative AI models to create synthetic data. These support the benchmark's primary-purpose interpretation, not a claim that the platforms cannot be used for AI. All three original errors have scores close to 0.5; that observation is not used to pick a new threshold after seeing the test set.

## Code review and validation

- Checked training, source weighting, validation-based selection, metric computation, artifact persistence and inference. Text vectorizers and classifiers fit training rows only. Source/category/label columns are not model features.
- Added a validated source-review overlay: unknown/duplicate IDs, unsupported actions and exclusions targeting evaluation records are rejected. An excluded training identity removes every training row in its connected group.
- Restricted category training explicitly to model-reviewed AI records. Future provisional category fields cannot silently enter this baseline.
- Saved the review file hash and excluded IDs/groups with the model. Prediction output now explicitly states the primary-purpose scope.
- Preserved the initial metrics in `artifacts/baseline/pre_source_review_metrics.json` instead of replacing the experiment history.
- The suite has 25 passing tests, including whole-group exclusion, frozen evaluation data, source-review coverage, source-file hash matching, model metric reproduction, input validation and vocabulary isolation.
- No additional current-data leakage bug was found in this bounded code review. Identity checks still do not resolve all rebrands, parent/subdomain aliases or semantic near-duplicates; that documented limitation remains.

## Results after applying the source review

| Binary variant | Effective training rows | Validation macro F1 | Test macro F1 |
|---|---:|---:|---:|
| reviewed_only | 231 | 0.823 | 0.770 |
| full_catalog | 14688 | 0.479 | 0.684 |

The selected model remains reviewed_only, with 94% test accuracy and 2/4 non-AI test examples correctly detected. Its confusion matrix is unchanged. Category test accuracy is now 28.3% (13/46), versus 30.4% (14/46) before the exclusions; test macro F1 is 0.205. Category validation accuracy is 34.8%. Excluding doubtful source records is a data-quality action and did not improve the category metric.

The same evaluation set was inspected during this review and reused for the reproduction. These are development results, not a new independent estimate. Four non-AI examples per split and many remaining unreviewed catalog positives still limit conclusions. No vendor security policies, approval status or risk scores were verified by this work.

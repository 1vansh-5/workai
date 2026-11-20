# AgriVision from-scratch data collection plan

## Goal

Build an original, consented, expert-reviewed dataset for a future AgriVision model focused first on Bihar and Jharkhand. The dataset must teach the system to (1) interpret field photos, (2) answer crop questions in farmers' languages, and (3) decline to guess when evidence is weak.

This is a data collection plan, not a claim that the current app can diagnose disease reliably. The repository's 31 knowledge entries are an initial advice catalog; they are not model-training data.

## What “from scratch” requires

Keep two data programs separate:

1. **Vision data:** original field photos paired with crop, plant part, confirmed condition, symptom location, severity and expert review. Include healthy, unclear, look-alike, multiple-problem and non-plant examples so the model learns when not to diagnose.
2. **Language data:** a licensed, documented corpus for the languages the model must read and write, plus agronomy materials and expert-written farmer questions and answers. A small FAQ cannot teach a general language model. The knowledge base should remain a separately searchable source of current vetted facts even after model training.

A model trained from random initialization also needs substantial compute. Use the pilot to prove collection quality and evaluate whether the project has enough data, expertise and compute for that scope before starting pretraining.

## Phase 1: define a narrow pilot

Start with **three crops** selected with local KVK/agronomy partners based on local importance and access to experts. Rice, wheat and maize are reasonable candidates for Bihar/Jharkhand, but have the partners confirm the crop list and the first conditions to include. Do not attempt all 19 crops and 31 knowledge entries at once.

For each crop, create a label list with an agronomist. Include:

- healthy
- each locally important disease, pest or deficiency the reviewer can reliably confirm
- look-alike / uncertain diagnosis
- multiple conditions present
- abiotic stress (for example water or heat stress), only where reviewers can label it
- other / not in current label list
- not enough evidence from photo
- non-target / not a plant

Never force an uncertain case into a disease class. Keep `unknown`, `uncertain`, and `not_visible` as valid outcomes.

## Phase 2: collect the field photo pilot

Recruit a small group of consenting farmers and one or more qualified crop reviewers. Ask the local KVK or agricultural university to nominate reviewers and confirm the label vocabulary.

For every plant case:

1. Assign a random `case_id`; do not put a farmer's name, phone number, precise GPS, or village name into filenames.
2. Photograph the whole plant and close views of affected and unaffected parts. Capture more than one angle when useful. Keep the original image; do not apply filters or synthetic edits.
3. Include normal field variation: sun and shade, different phones, backgrounds, growth stages, distances and symptom severity. Capture healthy plants and difficult negative examples too.
4. Record the crop, growth stage, plant part, date, broad district only if consented, capture conditions, and farmer-reported history separately from expert conclusions.
5. Ask a qualified reviewer to label the case. Record the evidence and review status. For ambiguous cases, request a second independent review; retain disagreement instead of hiding it.
6. Keep follow-up photos from the same plant, farm visit or outbreak linked by `case_id` / `group_id` so they cannot leak across train and test sets.
7. Remove faces, vehicle plates, documents and other identifying content before sharing. Store consent records separately from the model dataset.

### Pilot size and stop/go gate

For the first pilot, aim for **at least 100 distinct plant cases per agreed label**, spread across multiple farms, dates and phones where possible. This is a workflow and label-quality target only; it is **not sufficient evidence** that a from-scratch model will perform well. If any label is rare or hard to confirm, collect it as an explicit gap rather than filling it with duplicates or generated images.

At the end of the pilot, review:

- how many cases have expert-confirmed labels, including second reviews where needed;
- class balance and the number of distinct farms, dates and devices;
- label agreement and unresolved disagreements;
- duplicate/near-duplicate rate;
- whether field photos show enough context to distinguish look-alikes;
- data rights, consent, storage and reviewer capacity.

Only set full training targets after this audit and a learning-curve experiment. Keep a locked, farm-separated field test set that is never used for training or tuning.

## Phase 3: build the language and advice corpus

For each crop condition in the knowledge base, ask an agronomist to review the entry and provide a source reference, region/season relevance, date checked, and any restrictions. Do not treat a model-generated answer as a verified source.

Collect farmer questions and expert answers in English and Hindi first; add Bhojpuri, Maithili or other requested languages only with fluent reviewers. Each example should include:

- farmer wording (including spelling, colloquial terms and transliteration);
- crop and context, if known;
- answer in the requested language;
- supporting knowledge-base entry or authoritative source;
- uncertainty / referral condition;
- whether pesticide or fertilizer advice appears, and its reviewer check;
- reviewer name or protected reviewer ID, review date and status;
- license / permission and allowed use.

Include examples where the correct answer is to ask a follow-up question, say the evidence is insufficient, or refer the farmer to a KVK. Keep chemical rates and product recommendations out of model targets unless a qualified reviewer has confirmed legal, label-specific and local guidance. The product should continue to retrieve vetted current advice at answer time; training data alone gets stale.

## Phase 4: dataset quality and splits

- Deduplicate by exact hash and visual similarity; preserve links to the original case.
- Split by farm/plant/outbreak and collection period, never randomly by individual image. Related images must stay in one split.
- Keep a locked field test set from farms not represented in training. Also report a separate test slice by crop, condition, district/region, season, phone and lighting when sample counts permit.
- Track class counts and missingness. Do not hide imbalance with oversampling in reported test results.
- Record provenance and rights for every external source. Do not scrape images or text unless the source terms allow this use.
- Keep an immutable raw archive and a versioned, cleaned dataset. Log all corrections.
- Report per-class precision/recall, confusion matrix, calibration, abstention/unknown performance and performance on field conditions. Overall accuracy alone is not enough.

## Roles to recruit

- **Agronomy lead:** defines labels, reviews disease and treatment claims, resolves disagreements.
- **Field collectors:** collect photos and context consistently, with consent.
- **Language reviewers:** check farmer wording and translations.
- **Data steward:** permissions, de-identification, versions, backups and split integrity.
- **ML lead:** dataset audits, baselines, leakage checks, evaluation and compute estimates.

One person may hold multiple roles, but high-impact diagnosis labels and treatment advice need qualified review.

## Immediate next steps

1. Ask a local KVK/agricultural university to partner and select the three pilot crops and labels.
2. Confirm consent wording, photo storage, retention period and who may access raw images.
3. Print/export the two CSV templates in this folder; assign random case IDs.
4. Train collectors with 10 practice cases and review them together before field collection.
5. Collect the pilot, review labels, then produce a data audit and a compute/training decision.

## Related public research datasets

These can help with label taxonomy and baseline evaluation, but they do not replace local field data. PlantVillage contains more than 50,000 curated leaf images and was collected under controlled conditions; its original paper explicitly says more diverse data is needed for general accuracy. [PlantVillage dataset paper](https://arxiv.org/abs/1511.08060) · [PlantVillage model study](https://arxiv.org/abs/1604.03169)

PlantDoc was introduced as a more challenging field-image dataset, with 2,598 images across 13 plant species and up to 17 disease classes in the original paper. Check each dataset's license and original terms before using it. [PlantDoc paper](https://arxiv.org/abs/1911.10317)

## Templates

- `image_cases_template.csv`: one row per photo; use the same `case_id` for photos of one plant case and a shared `group_id` for the same farm visit/outbreak.
- `expert_qa_template.csv`: one row per reviewed question-answer example.

Keep actual consent records and any contact/location details outside these files.

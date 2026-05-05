"""
perturbation_audit.py
Builds a perturbation audit dataset for clinical NLP robustness analysis.
- 8 synthetic Alzheimer's clinical notes
- 8 perturbation types x 3 versions = 24 perturbations per note
- GPT-4o summarization with logprobs
- BERTScore semantic disruption measurement
- Analysis and visualization of disruption vs. model confidence
"""

from dotenv import load_dotenv
load_dotenv()
import os
import json
import time
import re
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RANDOM_SEED = 42
DELAY_SECONDS = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Synthetic Clinical Notes ─────────────────────────────────────────────────
CLINICAL_NOTES = [
    {
        "note_id": 1,
        "patient": "Mr. Harold Bennett, 78-year-old male",
        "text": """Patient: Mr. Harold Bennett, 78-year-old male.
Chief Complaint: Progressive memory loss and difficulty managing finances over the past 2 years.
Clinical History:
- January 2022: Family noted increased forgetfulness; patient began misplacing keys and forgetting appointments.
- June 2022: Patient got lost while driving a familiar route for the first time.
- March 2023: Unable to recall names of grandchildren; stopped managing personal finances.
- November 2023: Began experiencing hallucinations (visual) and sleep disturbances.
Cognitive Assessment: MMSE 18/30 (moderate impairment); CDR 1.5 (moderate dementia).
Medications: Donepezil 10mg daily, Memantine 10mg twice daily, Lorazepam 0.5mg at bedtime for sleep.
Lab/Imaging: MRI brain showed hippocampal atrophy bilaterally and cortical thinning in parietal lobes. Vitamin B12 and TSH within normal limits.
Clinical Impression: Moderate Alzheimer's disease with behavioral and psychological symptoms of dementia (BPSD).
Plan: Continue current cholinesterase inhibitor therapy. Refer to neuropsychiatry for BPSD management. Caregiver education and support referral. Reassess in 3 months."""
    },
    {
        "note_id": 2,
        "patient": "Ms. Dorothy Walsh, 82-year-old female",
        "text": """Patient: Ms. Dorothy Walsh, 82-year-old female.
Chief Complaint: Worsening confusion and inability to recognize family members.
Clinical History:
- February 2021: Daughter reported episodes of confusion at night.
- August 2021: Patient frequently asked the same questions repeatedly throughout the day.
- April 2022: Failed to recognize her daughter on two separate occasions.
- September 2022: Wandering behavior reported; patient found outside at night.
Cognitive Assessment: MMSE 12/30 (severe impairment); CDR 2 (severe dementia).
Medications: Rivastigmine patch 9.5mg/24hr, Quetiapine 25mg at bedtime, Sertraline 50mg daily.
Lab/Imaging: CT head demonstrated generalized cerebral atrophy and periventricular white matter changes. ApoE genotyping: ε4/ε4 homozygous.
Clinical Impression: Severe Alzheimer's disease with prominent neuropsychiatric symptoms including agitation and wandering.
Plan: Optimize antipsychotic dosing for night-time agitation. Install home safety measures. Arrange memory care evaluation. Follow-up in 6 weeks."""
    },
    {
        "note_id": 3,
        "patient": "Mr. George Patel, 71-year-old male",
        "text": """Patient: Mr. George Patel, 71-year-old male.
Chief Complaint: Mild forgetfulness affecting work performance noted by patient himself.
Clinical History:
- March 2023: Patient self-referred after noticing difficulty recalling names of colleagues.
- July 2023: Episodes of word-finding difficulty during meetings; compensating with notes.
- October 2023: Missed two important deadlines due to forgetting tasks despite reminders.
- January 2024: Wife noticed subtle personality changes including increased irritability.
Cognitive Assessment: MMSE 26/30 (mild impairment); CDR 0.5 (questionable/mild dementia).
Medications: Donepezil 5mg daily (newly initiated), Vitamin E 1000 IU daily, Omega-3 supplements.
Lab/Imaging: MRI brain with mild hippocampal volume loss, symmetric. FDG-PET: hypometabolism in bilateral posterior cingulate. CSF analysis: Aβ42 decreased, phospho-tau elevated.
Clinical Impression: Early Alzheimer's disease with biomarker confirmation. Mild cognitive impairment progressing to early AD.
Plan: Titrate Donepezil to 10mg daily in 4 weeks. Cognitive rehabilitation referral. Driving assessment recommended. Return in 3 months."""
    },
    {
        "note_id": 4,
        "patient": "Mrs. Eleanor Simmons, 76-year-old female",
        "text": """Patient: Mrs. Eleanor Simmons, 76-year-old female.
Chief Complaint: Severe memory impairment, unable to perform activities of daily living without assistance.
Clinical History:
- May 2020: Diagnosis of mild cognitive impairment established at outside facility.
- December 2020: Progressed to moderate Alzheimer's per outside records transferred to this clinic.
- August 2021: Incontinent of urine; requiring help with dressing and bathing.
- March 2022: No longer able to recognize her spouse; significant caregiver burden reported.
Cognitive Assessment: MMSE 10/30 (severe impairment); CDR 3 (severe dementia).
Medications: Memantine 20mg daily, Donepezil 10mg daily, Escitalopram 10mg daily, Melatonin 3mg at bedtime.
Lab/Imaging: MRI: severe diffuse cortical atrophy, marked hippocampal atrophy. CBC and metabolic panel unremarkable.
Clinical Impression: Severe Alzheimer's disease requiring full-time supervised care.
Plan: Discuss advanced care planning with family. Hospice evaluation may be appropriate. Maintain current medications for symptom management. Family meeting scheduled."""
    },
    {
        "note_id": 5,
        "patient": "Mr. Thomas Nguyen, 68-year-old male",
        "text": """Patient: Mr. Thomas Nguyen, 68-year-old male.
Chief Complaint: Short-term memory problems identified on routine cognitive screening during annual physical.
Clinical History:
- September 2023: Primary care physician noted MMSE score of 24 at annual visit.
- November 2023: Referred to neurology; wife confirmed occasional repetitive questioning at home.
- January 2024: Neuropsychological testing revealed deficits in delayed recall and executive function.
- March 2024: Repeat MRI obtained for baseline comparison.
Cognitive Assessment: MMSE 24/30 (mild impairment); CDR 0.5 (questionable dementia).
Medications: No anti-dementia medications initiated yet. Atorvastatin 20mg daily for hyperlipidemia. Lisinopril 10mg daily for hypertension.
Lab/Imaging: MRI brain: minimal age-appropriate hippocampal atrophy. Amyloid PET: positive for amyloid deposition bilaterally. Lipids and metabolic panel within normal limits.
Clinical Impression: Preclinical to early Alzheimer's disease with positive amyloid biomarker.
Plan: Initiate Donepezil 5mg daily. Aggressive cardiovascular risk factor management. Lifestyle counseling (exercise, diet). Enroll in longitudinal research study. Follow-up in 4 months."""
    },
    {
        "note_id": 6,
        "patient": "Mrs. Frances O'Brien, 84-year-old female",
        "text": """Patient: Mrs. Frances O'Brien, 84-year-old female.
Chief Complaint: Caregiver reports complete dependence for all activities of daily living and loss of speech.
Clinical History:
- January 2018: Diagnosed with Alzheimer's disease at outside institution.
- June 2019: Lost ability to prepare meals and manage medications independently.
- February 2021: Wheelchair-bound; lost ability to walk independently.
- October 2022: Communication limited to single words; unable to follow simple commands.
Cognitive Assessment: MMSE 3/30 (very severe impairment); CDR 5 (very severe dementia).
Medications: Donepezil 10mg daily (continuing for possible symptomatic benefit), Lorazepam 0.25mg PRN for acute agitation, Aspirin 81mg daily for cardiovascular prophylaxis.
Lab/Imaging: Most recent MRI (2023): profound diffuse atrophy. Monthly weight monitoring ongoing; BMI 17.8 (underweight).
Clinical Impression: End-stage Alzheimer's disease. High aspiration risk. Pressure injury risk elevated.
Plan: Transition to comfort-focused care. Palliative care and hospice consultation. Nasogastric tube feeding declined by family per advance directive. Repositioning protocol initiated."""
    },
    {
        "note_id": 7,
        "patient": "Mr. Richard Coleman, 73-year-old male",
        "text": """Patient: Mr. Richard Coleman, 73-year-old male.
Chief Complaint: Difficulty with navigation and spatial tasks; spouse reports increasing memory lapses.
Clinical History:
- April 2022: First reported getting lost in a previously familiar shopping center.
- August 2022: Difficulty assembling furniture; cannot follow multi-step instructions.
- January 2023: Unable to balance checkbook; persistent errors in simple arithmetic.
- July 2023: Began confusing day and night; sleeping at unusual hours.
Cognitive Assessment: MMSE 22/30 (mild-to-moderate impairment); CDR 1 (mild dementia).
Medications: Donepezil 10mg at bedtime, Galantamine ER 16mg daily (transition being considered), Trazodone 50mg at bedtime for sleep disturbance.
Lab/Imaging: MRI brain: parietal lobe atrophy greater than expected for age, hippocampal volumes 15% below age-matched norms. EEG: diffuse slowing consistent with early encephalopathy.
Clinical Impression: Mild-to-moderate Alzheimer's disease with prominent posterior cortical involvement. Parietal features suggest possible posterior cortical atrophy variant.
Plan: Switch from Donepezil to Galantamine ER. Occupational therapy evaluation. Fall risk assessment. Return visit in 3 months with repeat cognitive testing."""
    },
    {
        "note_id": 8,
        "patient": "Ms. Mabel Thornton, 79-year-old female",
        "text": """Patient: Ms. Mabel Thornton, 79-year-old female.
Chief Complaint: Increased agitation, paranoid ideation, and refusal to take medications.
Clinical History:
- February 2021: Diagnosed with mild Alzheimer's disease; Donepezil initiated.
- September 2021: Development of paranoid delusions (believes caregiver is stealing from her).
- March 2022: Agitation episodes escalating; refusing meals on several occasions.
- November 2022: Physical aggression toward caregiver reported twice; currently residing in memory care facility.
Cognitive Assessment: MMSE 15/30 (moderate impairment); CDR 2 (moderate-severe dementia).
Medications: Donepezil 10mg daily, Risperidone 0.5mg twice daily (for agitation and psychosis), Mirtazapine 15mg at bedtime (appetite stimulation and mood), Calcium 600mg + Vitamin D3 1000 IU daily.
Lab/Imaging: CT head: moderate cortical atrophy, no acute intracranial abnormality. Thyroid function normal. Urinalysis negative for infection (agitation precipitant ruled out).
Clinical Impression: Moderate Alzheimer's disease with severe neuropsychiatric symptoms (NPS): psychosis, agitation, and physical aggression.
Plan: Review Risperidone risk-benefit; consider Brexpiprazole if insufficient response. Non-pharmacological interventions (music therapy, sensory stimulation) to be initiated. Caregiver training in de-escalation techniques. Reassess in 6 weeks."""
    }
]

# ── Synonym Map ──────────────────────────────────────────────────────────────
SYNONYM_MAP = {
    "dementia": "cognitive decline",
    "Alzheimer's disease": "neurodegenerative disorder",
    "Alzheimer's": "neurodegenerative condition",
    "medication": "pharmacotherapy",
    "memory loss": "amnestic impairment",
    "confusion": "disorientation",
    "atrophy": "volumetric reduction",
    "hallucinations": "perceptual disturbances",
    "agitation": "psychomotor restlessness",
    "diagnosis": "clinical assessment",
    "cognitive impairment": "neurocognitive dysfunction",
    "MRI": "magnetic resonance imaging",
    "CT": "computed tomography",
    "patient": "individual",
    "treatment": "therapeutic intervention",
    "wandering": "elopement behavior",
    "caregiver": "care provider",
    "follow-up": "subsequent evaluation",
    "moderate": "intermediate-stage",
    "severe": "advanced-stage",
}

# ── Perturbation Functions ───────────────────────────────────────────────────
def perturb_temporal_reorder(note: str, version: int) -> str:
    """Shuffle ALL sentences/lines in the clinical history section completely."""
    rng = random.Random(RANDOM_SEED + version)
    lines = note.split('\n')
    history_start = None
    history_end = None

    # Find bounds of the Clinical History section
    for i, line in enumerate(lines):
        if 'Clinical History:' in line:
            history_start = i + 1  # start after the header line
        elif history_start is not None and history_end is None:
            # Section ends at the next non-bullet, non-empty section header
            if line.strip() and not line.strip().startswith('-') and not line.strip().startswith('\u2022') and ':' in line:
                history_end = i
                break

    if history_start is None:
        return note  # fallback: no history section found
    if history_end is None:
        history_end = len(lines)

    # Collect all non-empty lines in the history block
    history_lines = lines[history_start:history_end]
    content_lines = [l for l in history_lines if l.strip()]
    blank_lines = [l for l in history_lines if not l.strip()]

    if len(content_lines) >= 2:
        # Fully shuffle content lines (different order per version)
        rng.shuffle(content_lines)

    # Reconstruct: put shuffled content back
    lines[history_start:history_end] = content_lines + blank_lines
    return '\n'.join(lines)


def perturb_negation_insert(note: str, version: int) -> str:
    """Insert negation before 3-4 key clinical findings per note (stronger)."""
    rng = random.Random(RANDOM_SEED + version * 10)
    negation_words = ["no", "not", "without", "denies"]

    # Expanded clinical terms to negate
    targets = [
        r'\bprogressive\b', r'\bincreased\b', r'\bworsening\b',
        r'\bdifficulty\b', r'\bimpairment\b', r'\batrophy\b',
        r'\bconfusion\b', r'\bforgetfulness\b', r'\bhallucinations\b',
        r'\bagitation\b', r'\bwandering\b', r'\bparanoid\b',
        r'\bdecline\b', r'\bdisorientation\b', r'\bdepression\b',
        r'\bsevere\b', r'\bmoderate\b', r'\babnormal\b',
    ]

    modified = note
    n_negations = 3 + (version % 2)  # 3 or 4 negations per version
    neg = negation_words[version % len(negation_words)]

    shuffled_targets = targets.copy()
    rng.shuffle(shuffled_targets)
    applied = 0

    for target in shuffled_targets:
        if applied >= n_negations:
            break
        match = re.search(target, modified, re.IGNORECASE)
        if match:
            word = match.group()
            # Avoid double-negating (skip if already preceded by a negation word)
            prefix = modified[max(0, match.start()-10):match.start()].lower()
            if any(n in prefix for n in ['no ', 'not ', 'without', 'denies']):
                continue
            modified = modified[:match.start()] + f"{neg} {word}" + modified[match.end():]
            applied += 1

    return modified


def perturb_value_swap(note: str, version: int) -> str:
    """Replace numerical values with plausible but incorrect alternatives."""
    rng = random.Random(RANDOM_SEED + version * 100)
    modified = note

    # MMSE: valid range 0-30; swap with a different value
    def replace_mmse(m):
        original = int(m.group(1))
        options = [v for v in [8, 12, 16, 18, 20, 22, 24, 26] if v != original]
        return f"MMSE {rng.choice(options)}/30"
    modified = re.sub(r'MMSE (\d+)/30', replace_mmse, modified)

    # CDR: swap with adjacent stage
    cdr_map = {'0': '0.5', '0.5': '1', '1': '1.5', '1.5': '2', '2': '3', '3': '2', '5': '3'}
    def replace_cdr(m):
        val = m.group(1)
        new_val = cdr_map.get(val, '1')
        return f"CDR {new_val}"
    modified = re.sub(r'CDR (\d+(?:\.\d+)?)', replace_cdr, modified)

    # Dosages: multiply by a small factor
    factors = [0.5, 1.5, 2.0]
    factor = factors[version % len(factors)]
    def replace_dose(m):
        val = float(m.group(1))
        new_val = round(val * factor, 1)
        return f"{new_val}{m.group(2)}"
    modified = re.sub(r'(\d+(?:\.\d+)?)(mg)', replace_dose, modified)

    # Age: shift by ±5
    shift = [+5, -5, +8][version % 3]
    def replace_age(m):
        age = int(m.group(1))
        return f"{age + shift}-year-old"
    modified = re.sub(r'(\d{2})-year-old', replace_age, modified)

    return modified


def perturb_synonym_replace(note: str, version: int) -> str:
    """Replace clinical terms with synonyms."""
    modified = note
    items = list(SYNONYM_MAP.items())
    # Each version replaces a different subset
    rng = random.Random(RANDOM_SEED + version)
    chosen = rng.sample(items, min(4 + version, len(items)))
    for original, synonym in chosen:
        modified = re.sub(re.escape(original), synonym, modified, flags=re.IGNORECASE)
    return modified


def perturb_sentence_drop(note: str, version: int) -> str:
    """Remove 1-2 sentences from the note."""
    rng = random.Random(RANDOM_SEED + version)
    # Split into sentences (split on period+space or newline endings)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', note) if len(s.strip()) > 20]
    n_drop = 1 if version == 0 else 2
    # Don't drop the first or last sentence (preserve structure)
    if len(sentences) > n_drop + 2:
        drop_indices = set(rng.sample(range(1, len(sentences) - 1), n_drop))
        sentences = [s for i, s in enumerate(sentences) if i not in drop_indices]
    return ' '.join(sentences)


def perturb_contradiction_insert(note: str, version: int) -> str:
    """Insert a sentence that contradicts a key finding in the note."""
    contradictions = [
        "MMSE score was 30/30, indicating no cognitive impairment at last assessment.",
        "Recent neuroimaging showed no evidence of hippocampal atrophy or cortical thinning.",
        "Patient demonstrated full orientation to time, place, and person during examination.",
        "Biomarker panel including CSF amyloid and tau was entirely within normal limits.",
        "Family reports no observed memory deficits or behavioral changes in the past year.",
        "Cognitive function has returned to baseline following medication adjustment.",
    ]
    contradiction = contradictions[version % len(contradictions)]
    # Insert before the "Plan:" section
    if "Plan:" in note:
        return note.replace("Plan:", f"{contradiction}\nPlan:", 1)
    return note + f"\n{contradiction}"


def perturb_noise_inject(note: str, version: int) -> str:
    """Add irrelevant medical information unrelated to Alzheimer's."""
    noise_sentences = [
        "Patient also reports mild bilateral knee pain consistent with osteoarthritis, managed conservatively with acetaminophen.",
        "Dermatology was consulted for a 2cm sebaceous cyst on the upper back; excision scheduled for next month.",
        "Recent dental records indicate three crowns placed in 2021 with no current dental complaints.",
        "Patient's podiatrist treated a mild left great toe onychomycosis with topical antifungal cream last quarter.",
        "Ophthalmology follow-up for bilateral cataracts is scheduled; visual acuity 20/40 bilaterally.",
        "Patient experienced a minor laceration on the right forearm from a gardening accident; sutured in the ED.",
    ]
    noise = noise_sentences[version % len(noise_sentences)]
    # Insert after the medications section
    if "Lab/Imaging" in note:
        return note.replace("Lab/Imaging", f"{noise}\nLab/Imaging", 1)
    return note + f"\n{noise}"


def perturb_entity_swap(note: str, version: int) -> str:
    """Swap patient demographics (age, gender pronouns)."""
    modified = note
    age_shifts = [+10, -10, +15]
    shift = age_shifts[version % len(age_shifts)]

    # Swap age
    def swap_age(m):
        age = int(m.group(1))
        return f"{age + shift}-year-old"
    modified = re.sub(r'(\d{2})-year-old', swap_age, modified)

    # Swap gender
    if re.search(r'\bmale\b', note, re.IGNORECASE) and 'female' not in note.lower():
        # Male patient → make female
        modified = re.sub(r'\bMr\b\.', 'Ms.', modified)
        modified = re.sub(r'\bmale\b', 'female', modified, flags=re.IGNORECASE)
        modified = re.sub(r'\bHe\b', 'She', modified)
        modified = re.sub(r'\bhis\b', 'her', modified, flags=re.IGNORECASE)
        modified = re.sub(r'\bhim\b', 'her', modified, flags=re.IGNORECASE)
    elif re.search(r'\bfemale\b', note, re.IGNORECASE):
        # Female patient → make male
        modified = re.sub(r'\bMs\b\.', 'Mr.', modified)
        modified = re.sub(r'\bMrs\b\.', 'Mr.', modified)
        modified = re.sub(r'\bfemale\b', 'male', modified, flags=re.IGNORECASE)
        modified = re.sub(r'\bShe\b', 'He', modified)
        modified = re.sub(r'\bher\b', 'his', modified, flags=re.IGNORECASE)
        modified = re.sub(r'\bdaughter\b', 'son', modified, flags=re.IGNORECASE)
        modified = re.sub(r'\bwife\b', 'husband', modified, flags=re.IGNORECASE)

    return modified


PERTURBATION_FUNCTIONS = {
    "TEMPORAL_REORDER": perturb_temporal_reorder,
    "NEGATION_INSERT": perturb_negation_insert,
    "VALUE_SWAP": perturb_value_swap,
    "SYNONYM_REPLACE": perturb_synonym_replace,
    "SENTENCE_DROP": perturb_sentence_drop,
    "CONTRADICTION_INSERT": perturb_contradiction_insert,
    "NOISE_INJECT": perturb_noise_inject,
    "ENTITY_SWAP": perturb_entity_swap,
}

# ── GPT-4o Summarization ─────────────────────────────────────────────────────
SUMMARIZATION_PROMPT = (
    "You are a clinical assistant. Summarize the following patient note in 3-4 sentences, "
    "highlighting key findings relevant to Alzheimer's disease management: {note}"
)

def gpt4o_summarize(note_text: str) -> dict:
    """Call GPT-4o and return summary, mean_logprob, and response_time."""
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": SUMMARIZATION_PROMPT.format(note=note_text)}
        ],
        logprobs=True,
        max_tokens=300,
        temperature=0.0,
    )
    elapsed = time.time() - start

    summary = response.choices[0].message.content.strip()
    token_logprobs = [t.logprob for t in response.choices[0].logprobs.content if t.logprob is not None]
    mean_lp = float(np.mean(token_logprobs)) if token_logprobs else 0.0

    return {
        "generated_summary": summary,
        "mean_logprob": mean_lp,
        "response_time_seconds": round(elapsed, 3),
    }

# ── BERTScore ────────────────────────────────────────────────────────────────
def compute_bertscore(candidates: list, references: list) -> list:
    """Compute BERTScore F1 for a batch of candidates vs references."""
    from bert_score import score as bscore
    P, R, F1 = bscore(candidates, references, lang="en", verbose=False)
    return F1.tolist()

# ── Resume Logic ─────────────────────────────────────────────────────────────
def load_existing_results(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def is_done(existing_df: pd.DataFrame, note_id: int, ptype: str, version: int) -> bool:
    if existing_df.empty:
        return False
    mask = (
        (existing_df['note_id'] == note_id) &
        (existing_df['perturbation_type'] == ptype) &
        (existing_df['perturbation_version'] == version)
    )
    return mask.any()

# ── Main Pipeline ────────────────────────────────────────────────────────────
def main():
    OUTPUT_NOTES = "clinical_notes.json"
    OUTPUT_RESULTS = "perturbation_results_v2.csv"
    OUTPUT_ANALYSIS = "perturbation_analysis_v2.csv"
    OUTPUT_PLOT = "perturbation_plot_v2.png"

    # Step 1: Save clinical notes
    print("Saving clinical notes to", OUTPUT_NOTES)
    with open(OUTPUT_NOTES, 'w') as f:
        json.dump(CLINICAL_NOTES, f, indent=2)
    print(f"  Saved {len(CLINICAL_NOTES)} notes.")

    # Step 2-4: Generate baseline summaries + perturbed summaries
    existing_df = load_existing_results(OUTPUT_RESULTS)
    results = []

    print("\nGenerating baseline summaries for original notes...")
    baseline_summaries = {}  # note_id → summary

    for note in CLINICAL_NOTES:
        nid = note['note_id']
        # Check if baseline already computed
        if not existing_df.empty:
            mask = (
                (existing_df['note_id'] == nid) &
                (existing_df['perturbation_type'] == 'BASELINE') &
                (existing_df['perturbation_version'] == 0)
            )
            if mask.any():
                baseline_summaries[nid] = existing_df[mask]['generated_summary'].iloc[0]
                print(f"  Note {nid}: baseline already exists, skipping API call.")
                continue

        print(f"  Calling GPT-4o for baseline summary of Note {nid}...")
        result_data = gpt4o_summarize(note['text'])
        baseline_summaries[nid] = result_data['generated_summary']
        results.append({
            "note_id": nid,
            "perturbation_type": "BASELINE",
            "perturbation_version": 0,
            "original_note": note['text'],
            "perturbed_note": note['text'],
            "generated_summary": result_data['generated_summary'],
            "mean_logprob": result_data['mean_logprob'],
            "bertscore_f1": 1.0,
            "semantic_disruption": 0.0,
            "response_time_seconds": result_data['response_time_seconds'],
        })
        time.sleep(DELAY_SECONDS)

    # Reload existing to ensure baseline rows also included
    existing_df = load_existing_results(OUTPUT_RESULTS)
    if not existing_df.empty and results:
        combined_df = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
        combined_df.to_csv(OUTPUT_RESULTS, index=False)
        results = []
        existing_df = load_existing_results(OUTPUT_RESULTS)
    elif results:
        pd.DataFrame(results).to_csv(OUTPUT_RESULTS, index=False)
        results = []
        existing_df = load_existing_results(OUTPUT_RESULTS)

    print("\nGenerating perturbed summaries...")
    total = len(CLINICAL_NOTES) * len(PERTURBATION_FUNCTIONS) * 3
    done_count = 0

    for note in CLINICAL_NOTES:
        nid = note['note_id']
        for ptype, pfunc in PERTURBATION_FUNCTIONS.items():
            for version in range(3):
                done_count += 1
                if is_done(existing_df, nid, ptype, version):
                    print(f"  [{done_count}/{total}] Note {nid} | {ptype} v{version}: already done, skipping.")
                    continue

                perturbed = pfunc(note['text'], version)
                print(f"  [{done_count}/{total}] Note {nid} | {ptype} v{version}: calling GPT-4o...")
                try:
                    result_data = gpt4o_summarize(perturbed)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    result_data = {"generated_summary": "", "mean_logprob": 0.0, "response_time_seconds": 0.0}

                results.append({
                    "note_id": nid,
                    "perturbation_type": ptype,
                    "perturbation_version": version,
                    "original_note": note['text'],
                    "perturbed_note": perturbed,
                    "generated_summary": result_data['generated_summary'],
                    "mean_logprob": result_data['mean_logprob'],
                    "bertscore_f1": None,  # computed in batch below
                    "semantic_disruption": None,
                    "response_time_seconds": result_data['response_time_seconds'],
                })

                # Save every 10 calls just in case
                if len(results) % 10 == 0:
                    new_df = pd.DataFrame(results)
                    combined = pd.concat([existing_df, new_df], ignore_index=True)
                    combined.to_csv(OUTPUT_RESULTS, index=False)

                time.sleep(DELAY_SECONDS)

    # Save remaining
    if results:
        new_df = pd.DataFrame(results)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined.to_csv(OUTPUT_RESULTS, index=False)
        existing_df = combined

    # Step 4: Compute BERTScores for rows missing them
    print("\nComputing BERTScores...")
    df = pd.read_csv(OUTPUT_RESULTS)

    # Get rows that need BERTScore (non-baseline, missing values)
    needs_score = df[(df['perturbation_type'] != 'BASELINE') & (df['bertscore_f1'].isna())]
    
    if not needs_score.empty:
        candidates = []
        references = []
        indices = []
        for idx, row in needs_score.iterrows():
            nid = row['note_id']
            baseline_row = df[(df['note_id'] == nid) & (df['perturbation_type'] == 'BASELINE')]
            if not baseline_row.empty:
                ref = baseline_row['generated_summary'].iloc[0]
            else:
                ref = baseline_summaries.get(nid, "")
            candidates.append(str(row['generated_summary']) if pd.notna(row['generated_summary']) else "")
            references.append(ref)
            indices.append(idx)

        if candidates:
            f1_scores = compute_bertscore(candidates, references)
            for idx, f1 in zip(indices, f1_scores):
                df.at[idx, 'bertscore_f1'] = f1
                df.at[idx, 'semantic_disruption'] = round(1 - f1, 4)

        df.to_csv(OUTPUT_RESULTS, index=False)
        print(f"  Computed BERTScores for {len(needs_score)} rows.")
    else:
        print("  All BERTScores already computed.")

    # Step 5: Analysis
    print("\nRunning analysis...")
    df = pd.read_csv(OUTPUT_RESULTS)
    perturbed_df = df[df['perturbation_type'] != 'BASELINE'].copy()

    # Normalize mean_logprob to 0-1 (min-max)
    lp_min = perturbed_df['mean_logprob'].min()
    lp_max = perturbed_df['mean_logprob'].max()
    if lp_max != lp_min:
        perturbed_df['normalized_logprob'] = (perturbed_df['mean_logprob'] - lp_min) / (lp_max - lp_min)
    else:
        perturbed_df['normalized_logprob'] = 1.0

    analysis = perturbed_df.groupby('perturbation_type').agg(
        mean_semantic_disruption=('semantic_disruption', 'mean'),
        mean_logprob=('mean_logprob', 'mean'),
        mean_bertscore_f1=('bertscore_f1', 'mean'),
        mean_normalized_logprob=('normalized_logprob', 'mean'),
    ).reset_index()

    analysis['disruption_confidence_gap'] = (
        analysis['mean_semantic_disruption'] - analysis['mean_normalized_logprob']
    )

    analysis.to_csv(OUTPUT_ANALYSIS, index=False)
    print("\n--- Perturbation Analysis ---")
    print(analysis.to_string(index=False))

    # Step 6: Visualization
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(14, 6))

    ptypes = analysis['perturbation_type'].tolist()
    disruption = analysis['mean_semantic_disruption'].tolist()
    confidence = analysis['mean_normalized_logprob'].tolist()

    x = np.arange(len(ptypes))
    width = 0.35

    bars1 = ax.bar(x - width/2, disruption, width, label='Mean Semantic Disruption', color='#E05C5C', edgecolor='black', alpha=0.85)
    bars2 = ax.bar(x + width/2, confidence, width, label='Mean Model Confidence (Normalized)', color='#4A90D9', edgecolor='black', alpha=0.85)

    ax.set_xlabel('Perturbation Type', fontsize=12)
    ax.set_ylabel('Score (0–1)', fontsize=12)
    ax.set_title('Semantic Disruption vs. Model Confidence by Perturbation Type\n(GPT-4o on Synthetic Alzheimer\'s Clinical Notes)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(ptypes, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Annotate values
    for bar in bars1:
        ax.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"  Plot saved to {OUTPUT_PLOT}")

    print("\n=== DONE ===")
    print(f"  clinical_notes.json: {len(CLINICAL_NOTES)} notes")
    print(f"  perturbation_results.csv: {len(df)} rows")
    print(f"  perturbation_analysis.csv: {len(analysis)} perturbation types")
    print(f"  perturbation_plot.png: saved")

if __name__ == "__main__":
    main()

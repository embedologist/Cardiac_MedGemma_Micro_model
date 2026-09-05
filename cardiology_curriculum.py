"""
Full-Spectrum Clinical Cardiology & Lifestyle Medicine Curriculum
=================================================================
Curated, high-yield clinical reasoning pairs across:
  1. Arrhythmias & Conduction Disorders (AFib, Flutter, Bradycardia, AV Block, Pacemakers, PVCs)
  2. Acute Coronary Syndrome & CAD (STEMI, NSTEMI, Angina, Emergency Red Flags, Troponin)
  3. Heart Failure & Cardiomyopathies (HFrEF, HFpEF, NYHA classes, GDMT 4 pillars)
  4. Cardiovascular Pharmacology (Beta-blockers, DOACs, CHA2DS2-VASc, Antiarrhythmics, Emergency drugs)
     -> All medication responses feature mandatory Medical Disclaimer.
  5. Food & Clinical Nutrition (DASH, Mediterranean, Sodium <1500mg, Potassium/Magnesium, Omega-3, Fiber, Alcohol/Caffeine)
  6. Exercise & Cardiac Rehabilitation (AHA 150min guidelines, Target Heart Rate Zones, Post-AFib/MI safe resumption, HRR)
  7. Sleep & Circadian Cardiology (Sleep duration, Nocturnal Dipping, Obstructive Sleep Apnea, STOP-BANG, CPAP-AFib link)
  8. Stress & Autonomic Modulation (Diaphragmatic breathing 6 bpm, Vagal tone, Cortisol/Sympathetic overdrive, Smoking cessation)
"""

import os
import re
from typing import List, Dict, Optional

MEDICATION_DISCLAIMER = (
    "\n\n⚠️ **Medical Disclaimer:** For educational purposes only, not a prescription or treatment plan. "
    "**Do not start, stop, or change any medication without your doctor’s approval.** "
)

CONVERSATIONAL_GREETINGS: List[Dict[str, str]] = [
    {
        "instruction": "Hello",
        "response": "Hello! I am MedGemma-Micro, your on-device cardiovascular health and biosignal assistant. How can I help you today with heart health questions, ECG analysis, or lifestyle guidance?",
        "category": "Conversational",
    },
    {
        "instruction": "Hi",
        "response": "Hi there! I am MedGemma-Micro, your specialized cardiology and biosignal assistant. What cardiovascular or wellness questions can I assist you with today?",
        "category": "Conversational",
    },
    {
        "instruction": "Hey, how are you?",
        "response": "I am doing well, thank you for asking! As MedGemma-Micro, I am ready to assist you with evidence-based heart health insights, biosignal tracking, and lifestyle advice. How can I help you today?",
        "category": "Conversational",
    },
    {
        "instruction": "Good morning",
        "response": "Good morning! I am MedGemma-Micro, ready to help you monitor and understand your cardiovascular health. What heart health or wellness questions do you have today?",
        "category": "Conversational",
    },
    {
        "instruction": "Good evening",
        "response": "Good evening! I am MedGemma-Micro, your on-device cardiovascular assistant. How can I support your heart health or answer any questions for you this evening?",
        "category": "Conversational",
    },
    {
        "instruction": "Who are you?",
        "response": "I am MedGemma-Micro, an efficient on-device AI assistant specialized in clinical cardiology, ECG/PPG biosignal analysis, and evidence-based cardiovascular lifestyle medicine.",
        "category": "Conversational",
    },
    {
        "instruction": "What can you do?",
        "response": "I can analyze cardiovascular biosignals (PPG and ECG waveforms for arrhythmias like AFib, PVCs, bradycardia), provide evidence-based guidance on heart medications, cardiac diets (DASH/Mediterranean), exercise guidelines, sleep cardiology, and answer questions across our extensive cardiac health database.",
        "category": "Conversational",
    },
]


def load_cardiac_qa_curriculum(dataset_path: str = "cardiac_health_dataset.md", max_per_category: Optional[int] = None) -> List[Dict[str, str]]:
    """Loads and formats Q&A pairs from cardiac_health_dataset.md for student distillation."""
    if not os.path.exists(dataset_path):
        return []
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Warning: Could not read {dataset_path}: {e}")
        return []

    pattern = r"### Question (\d+)\s*\((.*?)\)\s*\n+\*\*Q:\*\*\s*(.*?)\n+\*\*A:\*\*\s*(.*?)(?=\n+---|### Question|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    items = []
    cat_counts: Dict[str, int] = {}

    for q_num, category, question, answer in matches:
        q_clean = question.strip()
        a_clean = answer.strip()
        cat_clean = category.strip()

        if max_per_category is not None:
            cat_counts[cat_clean] = cat_counts.get(cat_clean, 0) + 1
            if cat_counts[cat_clean] > max_per_category:
                continue

        # Attach exact medical disclaimer to medication Q&A items
        if cat_clean.lower() == "medications" or any(
            kw in q_clean.lower()
            for kw in ["statin", "beta-blocker", "aspirin", "diuretic", "ace inhibitor", "nitrate", "anticoagulant", "antiarrhythmic", "pcsk9", "calcium channel"]
        ):
            resp = a_clean + MEDICATION_DISCLAIMER
        else:
            resp = a_clean

        items.append({
            "instruction": q_clean,
            "response": resp,
            "category": cat_clean,
        })
    return items


BASE_CARDIOLOGY_CURRICULUM: List[Dict[str, str]] = [
    # -------------------------------------------------------------------------
    # 1. FOOD, NUTRITION & ELECTROLYTES
    # -------------------------------------------------------------------------
    {
        "instruction": "What is the best diet and food plan for heart disease, high blood pressure, and preventing arrhythmia episodes?",
        "response": (
            "Evidence-based cardiovascular nutrition centers on two clinically validated dietary frameworks: the **DASH Diet (Dietary Approaches to Stop Hypertension)** and the **Mediterranean Diet**.\n\n"
            "1. Dietary Sodium Restriction:\n"
            "- Strict limit: Restrict dietary sodium to **below 1,500 to 2,000 mg/day** (roughly 3/4 teaspoon of table salt).\n"
            "- Hidden sodium: Over 70% of dietary sodium comes from processed foods, canned soups, processed meats (bacon, deli meats), frozen ready-meals, and condiments (soy sauce, dressings). Cooking fresh whole foods is the most effective way to lower sodium.\n\n"
            "2. Electrolyte Balance for Membrane Stability:\n"
            "- **Potassium (K+)**: Target 3,500-4,700 mg/day from dark leafy greens (spinach, kale), avocados, sweet potatoes, bananas, and lentils. Potassium counters sodium-induced vasoconstriction and stabilizes resting cardiomyocyte membrane potentials. (Caution: Restrict if advanced chronic kidney disease or hyperkalemia).\n"
            "- **Magnesium (Mg2+)**: 350-420 mg/day from pumpkin seeds, almonds, black beans, and oats. Magnesium is a natural cofactor for the Na+/K+-ATPase pump and prevents ventricular/atrial ectopic firing.\n\n"
            "3. Healthy Fats & Fiber:\n"
            "- **Omega-3 Polyunsaturated Fats**: Consume cold-water fatty fish (wild salmon, sardines, mackerel) 2-3 times per week, plus walnuts and flaxseeds. Omega-3s lower serum triglycerides, decrease platelet aggregation, and exert anti-arrhythmic effects.\n"
            "- **Soluble Fiber**: Target 25-35 grams/day (oatmeal, psyllium husk, legumes, apples) to bind intestinal bile acids and lower circulating LDL-cholesterol.\n"
            "- Limit saturated fat to < 5-6% of total daily calories; eliminate artificial trans-fats.\n\n"
            "4. Beverages & Stimulants:\n"
            "- **Alcohol**: Strictly avoid binge drinking ('Holiday Heart Syndrome' directly triggers Atrial Fibrillation). For patients with arrhythmias or heart failure, alcohol abstinence is strongly advised.\n"
            "- **Caffeine**: Moderate coffee consumption (1-2 cups/day, <200-300 mg caffeine) is generally safe, but avoid concentrated pre-workout stimulants and energy drinks containing guarana or taurine."
        ),
        "category": "Nutrition",
    },
    {
        "instruction": "What should heart failure patients eat and drink? Explain daily fluid and sodium limits.",
        "response": (
            "Nutritional management in heart failure (both HFrEF and HFpEF) is vital to prevent fluid overload, acute pulmonary congestion, and hospital readmission:\n\n"
            "1. Sodium Restriction:\n"
            "- Restrict sodium to **< 1,500 mg daily**.\n"
            "- Even a single high-sodium meal (e.g., takeout pizza, Chinese food, or canned soup) can cause acute retention of 2-3 liters of water within 12-24 hours, precipitating acute decompensation.\n"
            "- Replace salt with garlic, lemon juice, black pepper, rosemary, and salt-free seasoning blends (avoid potassium-chloride salt substitutes if taking ACE inhibitors, ARBs, or Spironolactone without doctor approval).\n\n"
            "2. Fluid Management Protocol:\n"
            "- Restrict fluid intake to **1.5 to 2.0 Liters per day** (approximately 50-68 oz) if symptomatic or prone to fluid retention.\n"
            "- Fluid includes: water, coffee, tea, juices, milk, soups, ice cream, and gelatin.\n"
            "- Practical tip: Keep a designated 2-liter pitcher in the refrigerator to measure total daily fluid consumption.\n\n"
            "3. Daily Morning Weight Monitoring (The '2-3-5' Rule):\n"
            "- Weigh yourself every single morning after urinating, before breakfast, wearing similar light clothing.\n"
            "- Notify your cardiologist immediately if you gain:\n"
            "  * >= 2 to 3 lbs (1.0 - 1.5 kg) in 24 hours, OR\n"
            "  * >= 5 lbs (2.3 kg) in 1 week.\n"
            "- Rapid weight gain represents fluid retention, requiring prompt diuretic dose adjustment."
        ),
        "category": "Nutrition",
    },

    # -------------------------------------------------------------------------
    # 2. EXERCISE & CARDIAC REHABILITATION
    # -------------------------------------------------------------------------
    {
        "instruction": "What are safe exercise guidelines and physical activity recommendations for someone with heart disease or after an arrhythmia episode?",
        "response": (
            "Structured physical activity strengthens myocardial efficiency, improves endothelial function, and lowers resting sympathetic tone. Clinical exercise guidelines (AHA/ACC/ESC) recommend:\n\n"
            "1. Weekly Aerobic Activity Targets:\n"
            "- **Moderate-Intensity Exercise**: At least **150 to 300 minutes per week** (e.g., 30-45 minutes, 5 days/week) of brisk walking, stationary cycling, low-impact elliptical, or water aerobics.\n"
            "- Alternatively: 75 to 150 minutes of vigorous-intensity aerobic activity for cleared, low-risk patients.\n\n"
            "2. Target Heart Rate Zones & Karvonen Method:\n"
            "- Estimate Maximum Heart Rate: $\\text{HR}_{\\max} \\approx 220 - \\text{age}$.\n"
            "- Target training range for cardiac patients: **50% to 70% of Heart Rate Reserve (HRR)**.\n"
            "- The 'Talk Test': You should be able to speak in full sentences without gasping. If you cannot speak without pausing for air, reduce exercise intensity.\n\n"
            "3. Post-Arrhythmia / Post-AFib Resumption Protocol:\n"
            "- Following an acute paroxysmal AFib episode, refrain from high-intensity interval training (HIIT) or heavy resistance loading for **at least 24 to 48 hours**.\n"
            "- Resume low-intensity walking, keeping heart rate strictly below 60-70% of age-predicted reserve.\n"
            "- Monitor **1-Minute Heart Rate Recovery (HRR)**: Measure heart rate immediately at exercise cessation and again after 60 seconds of rest. A drop of **>= 12-18 bpm** indicates healthy vagal reactivation; a drop < 12 bpm indicates autonomic blunting.\n\n"
            "4. Safety Rules & When to Stop Immediately:\n"
            "- Avoid the Valsalva maneuver (holding breath during heavy weightlifting)—it causes acute spikes in intrathoracic and blood pressure.\n"
            "- **STOP EXERCISING IMMEDIATELY** and seek urgent evaluation if you experience: chest pressure/tightness, pain radiating to arm/jaw, lightheadedness/dizziness, paleness, nausea, or irregular palpitations."
        ),
        "category": "Exercise",
    },
    {
        "instruction": "Explain how to calculate target heart rate zones for safe cardio workouts.",
        "response": (
            "Calculating individualized target heart rate zones ensures cardiovascular conditioning while preventing excessive myocardial oxygen demand:\n\n"
            "1. Step 1: Determine Maximum Heart Rate (HR_max):\n"
            "- Standard formula: $\\text{HR}_{\\max} = 220 - \\text{Age}$\n"
            "- Example: For a 60-year-old: $220 - 60 = 160\\text{ bpm}$.\n"
            "*(Note: In patients taking beta-blockers like Metoprolol, HR_max is blunted by 15-25%—use the Borg Rating of Perceived Exertion scale 11-14 out of 20 instead)*.\n\n"
            "2. Step 2: Calculate Heart Rate Reserve (HRR / Karvonen Formula):\n"
            "- $\\text{HRR} = \\text{HR}_{\\max} - \\text{Resting HR}$\n"
            "- Example with Resting HR = 70 bpm: $\\text{HRR} = 160 - 70 = 90\\text{ bpm}$.\n\n"
            "3. Step 3: Compute Target Intensity Zones:\n"
            "- **Zone 1 (Warm-Up & Active Recovery)**: 50-60% of HRR + Resting HR\n"
            "  * $70 + (0.50 \\times 90) = 115\\text{ bpm}$\n"
            "- **Zone 2 (Aerobic Endurance & Fat Oxidation - Recommended)**: 60-70% of HRR + Resting HR\n"
            "  * $70 + (0.60 \\times 90) = 124\\text{ bpm}$ to $70 + (0.70 \\times 90) = 133\\text{ bpm}$\n"
            "- **Zone 3 (Aerobic Fitness)**: 70-80% of HRR + Resting HR\n"
            "  * $133 - 142\\text{ bpm}$.\n\n"
            "4. Post-Workout Cooldown:\n"
            "- Always perform a 5-minute gradual cooldown walk. Check 1-minute HRR recovery to track parasympathetic reactivation."
        ),
        "category": "Exercise",
    },

    # -------------------------------------------------------------------------
    # 3. SLEEP, CIRCADIAN RHYTHM & SLEEP APNEA
    # -------------------------------------------------------------------------
    {
        "instruction": "How does sleep quality, sleep duration, and Obstructive Sleep Apnea (OSA) impact heart disease and Atrial Fibrillation?",
        "response": (
            "Sleep is an active period of cardiovascular restoration. Disrupted sleep architecture is one of the strongest modifiable risk factors for cardiac arrhythmias and hypertension:\n\n"
            "1. Physiological Nocturnal Dipping:\n"
            "- During healthy restorative sleep (particularly Slow-Wave N3 sleep), the parasympathetic nervous system dominates.\n"
            "- Normal dipping: Mean arterial pressure and resting heart rate drop **10% to 20%** compared to daytime levels.\n"
            "- **'Non-Dipping' profile (< 10% drop)**: Associated with sustained nighttime sympathetic tone, vascular stiffening, left ventricular hypertrophy, and double the risk of stroke and myocardial infarction.\n\n"
            "2. Obstructive Sleep Apnea (OSA) & Atrial Fibrillation:\n"
            "- Mechanism: Repeated upper airway collapse causes severe nocturnal hypoxia and hypercapnia. Generating massive negative intrathoracic pressure against an occluded airway mechanically stretches the thin walls of the atria.\n"
            "- Sympathetic Surge: Awakenings trigger explosive catecholamine (adrenaline) surges, directly triggering paroxysmal AFib episodes during the night or early morning.\n"
            "- Clinical impact: Untreated OSA reduces the success rate of AFib catheter ablation by nearly 50%. CPAP (Continuous Positive Airway Pressure) therapy restores nocturnal dipping and halves AFib recurrence.\n\n"
            "3. Screening & Symptoms (STOP-BANG Questionnaire):\n"
            "- Suspect OSA if: loud habitual snoring, daytime exhaustion despite 8 hours in bed, morning headaches, gasping/choking at night, neck circumference > 17 inches (men) or > 16 inches (women), or treatment-resistant hypertension.\n\n"
            "4. Sleep Hygiene Guidelines for Cardiovascular Health:\n"
            "- Target **7 to 9 hours** of consistent, uninterrupted sleep.\n"
            "- Keep a consistent sleep and wake schedule (within 30 minutes, 7 days/week).\n"
            "- Maintain bedroom temperature cool (65-68°F / 18-20°C) and dark.\n"
            "- Avoid heavy meals, caffeine, and alcohol within 3-4 hours of bedtime."
        ),
        "category": "Sleep",
    },

    # -------------------------------------------------------------------------
    # 4. STRESS, AUTONOMIC REGULATION & SMOKING
    # -------------------------------------------------------------------------
    {
        "instruction": "What are effective stress management and breathing techniques to lower heart rate and reduce palpitations?",
        "response": (
            "Psychological stress triggers the hypothalamic-pituitary-adrenal (HPA) axis, releasing cortisol and epinephrine, which elevates blood pressure, accelerates sinus node firing, and promotes atrial and ventricular ectopy (PVCs/PACs):\n\n"
            "1. Slow Paced Diaphragmatic Breathing (Resonance Frequency Breathing):\n"
            "- Inhale slowly through the nose for **4 seconds**, expanding the belly (diaphragm).\n"
            "- Exhale smoothly through pursed lips for **6 seconds** (a rate of ~6 breaths per minute).\n"
            "- Physiological mechanism: Prolonging exhalation stimulates baroreceptors in the aortic arch and carotid sinus, activating the vagus nerve (cranial nerve X) and releasing acetylcholine to slow the sinus node and enhance Heart Rate Variability (rMSSD).\n"
            "- Practice for 5-10 minutes twice daily or immediately during episodes of stress or benign palpitations.\n\n"
            "2. Box Breathing (Tactical Calming):\n"
            "- Inhale 4s -> Hold 4s -> Exhale 4s -> Hold empty 4s. Repeat for 4 cycles.\n\n"
            "3. Smoking & Nicotine Cessation:\n"
            "- Nicotine causes immediate vasoconstriction, endothelial dysfunction, elevates resting heart rate by 10-15 bpm, and doubles platelet aggregation risk.\n"
            "- Within **20 minutes** of quitting smoking, heart rate and blood pressure drop.\n"
            "- Within **12 hours**, blood carbon monoxide levels normalize.\n"
            "- Within **1 year**, excess risk of coronary heart disease drops by 50%."
        ),
        "category": "Lifestyle",
    },

    # -------------------------------------------------------------------------
    # 5. ARRHYTHMIAS & CONDUCTION DISORDERS
    # -------------------------------------------------------------------------
    {
        "instruction": "Can you please explain bradycardia, its causes, symptoms, and when it requires a pacemaker?",
        "response": (
            "Bradycardia is defined clinically as a resting heart rate below 60 beats per minute (bpm). "
            "While resting rates of 40-60 bpm can be physiologically normal in well-conditioned athletes or during deep non-REM sleep (physiological vagal tone), "
            "pathological bradycardia occurs when cardiac output is insufficient to meet metabolic demands.\n\n"
            "Key Etiologies:\n"
            "1. Sinus Node Dysfunction (Sick Sinus Syndrome): Tachy-brady syndrome, sinus pause (>3 seconds), or sinus arrest.\n"
            "2. Atrioventricular (AV) Conduction Block: Second-degree Mobitz type II or Third-degree (complete) AV block.\n"
            "3. Extrinsic Causes: Medications (beta-blockers, verapamil, diltiazem, digoxin), severe hypothyroidism, elevated intracranial pressure, or electrolyte imbalances (hyperkalemia).\n\n"
            "Clinical Symptoms:\n"
            "Presyncope, true syncope, exertional fatigue, lightheadedness, dyspnea, and cognitive blunting.\n\n"
            "Indications for Permanent Pacemaker (PPM):\n"
            "- Symptomatic sinus node dysfunction with documented correlation between symptoms and bradycardia.\n"
            "- Asymptomatic sinus pauses > 3.0 seconds in awake patients.\n"
            "- Second-degree Mobitz Type II AV block or Third-degree complete heart block (Class I indication regardless of symptoms, due to high risk of sudden cardiac arrest)."
        ),
        "category": "Arrhythmias",
    },
    {
        "instruction": "What are the first-line rate control and stroke prevention medications for Atrial Fibrillation (AFib)?",
        "response": (
            "Atrial Fibrillation (AFib) management rests on two pillars: rate control and thromboembolism prevention.\n\n"
            "1. Rate Control:\n"
            "- First-line: Cardioselective beta-blockers such as Metoprolol succinate (25-50 mg daily, titrated to resting HR < 80-110 bpm) or Bisoprolol (2.5-10 mg daily).\n"
            "- Alternative: Non-dihydropyridine calcium channel blockers like Diltiazem (120-240 mg daily extended-release) or Verapamil (avoid in HFrEF with LVEF < 40%).\n\n"
            "2. Thromboembolism & Stroke Prevention:\n"
            "- Calculate the CHA2DS2-VASc score (Congestive heart failure [1], Hypertension [1], Age >= 75 [2], Diabetes [1], Stroke/TIA [2], Vascular disease [1], Age 65-74 [1], Sex category female [1]).\n"
            "- Anticoagulation is indicated if CHA2DS2-VASc >= 2 in men or >= 3 in women.\n"
            "- Direct Oral Anticoagulants (DOACs) are preferred over Warfarin:\n"
            "  * Apixaban: 5 mg PO BID (reduced to 2.5 mg BID if >= 2 of: age >= 80, weight <= 60 kg, serum creatinine >= 1.5 mg/dL).\n"
            "  * Rivaroxaban: 20 mg PO daily taken with the evening meal.\n"
            "  * Dabigatran: 150 mg PO BID.\n"
            "- Warfarin (target INR 2.0-3.0) is strictly reserved for mechanical prosthetic heart valves or moderate-to-severe mitral stenosis."
            + MEDICATION_DISCLAIMER
        ),
        "category": "Arrhythmias",
    },
    {
        "instruction": "What are Premature Ventricular Contractions (PVCs) and when does PVC burden become dangerous?",
        "response": (
            "Premature Ventricular Contractions (PVCs) are premature heartbeats originating from an ectopic focus in the ventricular myocardium rather than the sinoatrial node. "
            "On ECG/PPG, they manifest as a wide QRS complex followed by a compensatory pause.\n\n"
            "Clinical Risk & PVC Burden Thresholds:\n"
            "- Benign: Isolated, infrequent PVCs (< 1% of total daily beats) in patients with a structurally normal heart are benign and often provoked by caffeine, stress, lack of sleep, or alcohol.\n"
            "- Concerning PVC Burden (> 10-15%): When PVCs account for more than 10-15% of all daily heartbeats on 24-hour Holter monitoring (typically > 10,000 to 15,000 PVCs/day), patients risk developing PVC-induced cardiomyopathy (progressive left ventricular systolic dysfunction).\n"
            "- High-Risk Characteristics: Multi-focal PVCs, couplets/triplets (non-sustained VT), 'R-on-T' phenomenon (triggering polymorphic VT/VF), or PVCs triggered by exercise.\n\n"
            "Workup & Management:\n"
            "- Transthoracic echocardiogram (TTE) to rule out structural heart disease and measure LVEF.\n"
            "- Serum electrolytes (potassium target >= 4.0 mEq/L, magnesium >= 2.0 mg/dL).\n"
            "- First-line therapy: Beta-blockers (Metoprolol succinate) or Catheter Radiofrequency Ablation if burden exceeds 10-15% with symptoms or declining EF."
            + MEDICATION_DISCLAIMER
        ),
        "category": "Arrhythmias",
    },
    {
        "instruction": "Explain the difference between Atrial Fibrillation and Atrial Flutter.",
        "response": (
            "While both are supraventricular tachyarrhythmias originating in the atria, their electrophysiological mechanisms and waveforms differ fundamentally:\n\n"
            "1. Atrial Fibrillation (AFib):\n"
            "- Mechanism: Chaotic, multiple micro-reentrant wavelets with disorganized atrial electrical activity (atrial rate 350-600 bpm).\n"
            "- ECG/PPG Morphology: Absence of distinct P-waves, irregular baseline fibrillatory waves, and an 'irregularly irregular' ventricular response.\n"
            "- Peak PPG: Highly variable beat-to-beat amplitudes and inconsistent pulse transit times.\n\n"
            "2. Atrial Flutter (AFL):\n"
            "- Mechanism: Macro-reentrant circuit, most commonly counter-clockwise around the cavotricuspid isthmus (CTI) in the right atrium (atrial rate typically 250-350 bpm, classically ~300 bpm).\n"
            "- ECG/PPG Morphology: Classic regular 'sawtooth' flutter waves (F-waves), most prominent in inferior leads II, III, and aVF.\n"
            "- Ventricular Rate: Typically regular with fixed AV conduction ratios (e.g., 2:1 conduction results in a ventricular rate of exactly ~150 bpm; 3:1 at ~100 bpm; 4:1 at ~75 bpm).\n\n"
            "Clinical Implications:\n"
            "Both share the same stroke risk and anticoagulation indications (CHA2DS2-VASc scoring). However, typical Atrial Flutter has a >95% cure rate with cavotricuspid isthmus catheter radiofrequency ablation."
        ),
        "category": "Arrhythmias",
    },

    # -------------------------------------------------------------------------
    # 6. ACUTE CORONARY SYNDROME & EMERGENCY RED FLAGS
    # -------------------------------------------------------------------------
    {
        "instruction": "What are the critical emergency red-flag symptoms of acute myocardial infarction (heart attack) that require 911?",
        "response": (
            "Acute Coronary Syndrome (ACS) is a life-threatening emergency requiring immediate emergency medical services (EMS / 911) activation.\n\n"
            "Primary Red-Flag Symptoms:\n"
            "1. Chest Discomfort: Heavy, crushing, squeezing, or substernal pressure ('elephant sitting on chest'), lasting more than 5-10 minutes, often not relieved by rest or nitroglycerin.\n"
            "2. Radiation: Pain radiating to the left shoulder, left arm, neck, jaw, interscapular back, or epigastrium.\n"
            "3. Associated Autonomic Symptoms: Profuse cold diaphoresis (sweating), acute shortness of breath (dyspnea), nausea/vomiting, lightheadedness, or sudden syncope.\n\n"
            "Atypical Presentations (Common in women, elderly, and diabetics):\n"
            "- Sudden isolated unexplained dyspnea, profound exhaustion, nausea, or epigastric burning without prominent chest pain.\n\n"
            "Immediate First-Aid Protocol (Pre-Hospital):\n"
            "- Call 911 immediately; do not drive oneself to the hospital.\n"
            "- Chew and swallow non-enteric coated Aspirin 162-325 mg (unless severely allergic or active bleeding).\n"
            "- Rest in a seated or semi-reclined position; administer sublingual nitroglycerin 0.4 mg every 5 minutes (max 3 doses) if previously prescribed and systolic BP > 100 mmHg."
            + MEDICATION_DISCLAIMER
        ),
        "category": "Emergency",
    },
    {
        "instruction": "Explain the difference between STEMI, NSTEMI, and Unstable Angina.",
        "response": (
            "Acute Coronary Syndrome (ACS) encompasses a spectrum of acute myocardial ischemia divided into three distinct diagnostic categories:\n\n"
            "1. ST-Elevation Myocardial Infarction (STEMI):\n"
            "- Pathophysiology: Acute, complete transmural occlusion of a major epicardial coronary artery, typically from plaque rupture and occlusive thrombus.\n"
            "- ECG: New ST-segment elevation >= 1 mm in >= 2 contiguous leads (or >= 1.5-2.5 mm in V2-V3), or new Left Bundle Branch Block (LBBB).\n"
            "- Cardiac Biomarkers: Elevated high-sensitivity Troponin I or T.\n"
            "- Treatment: Immediate Primary Percutaneous Coronary Intervention (PPCI) with Door-to-Balloon time target < 90 minutes.\n\n"
            "2. Non-ST-Elevation Myocardial Infarction (NSTEMI):\n"
            "- Pathophysiology: Severe subtotal coronary stenosis or transient occlusion causing subendocardial myocardial necrosis.\n"
            "- ECG: ST-segment depressions, T-wave inversions, or transient non-specific changes (NO persistent ST elevation).\n"
            "- Cardiac Biomarkers: Elevated Cardiac Troponin.\n"
            "- Treatment: Dual antiplatelet therapy (Aspirin + P2Y12 inhibitor), anticoagulation (Heparin), and early coronary angiography.\n\n"
            "3. Unstable Angina (UA):\n"
            "- Pathophysiology: Myocardial ischemia at rest or with minimal exertion without myocardial cell necrosis.\n"
            "- ECG: Normal or transient ST/T-wave changes.\n"
            "- Cardiac Biomarkers: Normal Cardiac Troponin (negative biomarkers distinguish UA from NSTEMI)."
        ),
        "category": "CAD",
    },

    # -------------------------------------------------------------------------
    # 7. HEART FAILURE & PHARMACOLOGY
    # -------------------------------------------------------------------------
    {
        "instruction": "Explain Heart Failure with reduced Ejection Fraction (HFrEF) and the 'Four Pillars' of Guideline-Directed Medical Therapy (GDMT).",
        "response": (
            "Heart Failure with reduced Ejection Fraction (HFrEF) is defined as clinical heart failure with a left ventricular ejection fraction (LVEF) <= 40%. "
            "Contemporary guidelines (ACC/AHA/ESC) mandate rapid initiation and uptitration of the 'Four Pillars' of GDMT to reduce mortality and hospitalizations:\n\n"
            "The Four Foundational Pillars of GDMT:\n"
            "1. ARNI (Angiotensin Receptor-Neprilysin Inhibitor):\n"
            "- Sacubitril/Valsartan (Entresto), starting at 24/26 mg or 49/51 mg BID, uptitrated to target 97/103 mg BID. Superior to ACEi/ARBs in reducing cardiovascular death.\n"
            "2. Evidence-Based Beta-Blocker:\n"
            "- Only three agents proven to reduce mortality in HFrEF: Carvedilol (target 25-50 mg BID), Metoprolol succinate (target 200 mg daily), or Bisoprolol (target 10 mg daily).\n"
            "3. Mineralocorticoid Receptor Antagonist (MRA):\n"
            "- Spironolactone (12.5-25 mg daily) or Eplerenone (25-50 mg daily). Monitor serum potassium and renal function.\n"
            "4. SGLT2 Inhibitor (Sodium-Glucose Cotransporter-2 Inhibitor):\n"
            "- Dapagliflozin 10 mg daily or Empagliflozin 10 mg daily. Proven benefits regardless of whether the patient has diabetes.\n\n"
            "Adjunctive Symptom Control:\n"
            "- Loop diuretics (Furosemide, Bumetanide, Torsemide) at the lowest dose required to maintain euvolemia."
            + MEDICATION_DISCLAIMER
        ),
        "category": "HeartFailure",
    },
    {
        "instruction": "Review contraindications and dangerous drug interactions for Metoprolol.",
        "response": (
            "Metoprolol (Lopressor / Toprol-XL) is a cardioselective beta-1 adrenergic receptor antagonist.\n\n"
            "Absolute & Relative Contraindications:\n"
            "1. Severe Sinus Bradycardia (resting HR < 45-50 bpm).\n"
            "2. Second- or Third-Degree AV Block (without a functioning permanent pacemaker).\n"
            "3. Cardiogenic Shock or Acute Decompensated Heart Failure with pulmonary edema or severe peripheral hypoperfusion.\n"
            "4. Severe Sinus Node Dysfunction (Sick Sinus Syndrome).\n"
            "5. Severe active bronchospastic disease (severe brittle asthma).\n\n"
            "Critical Drug Interactions:\n"
            "- Non-dihydropyridine CCBs (Diltiazem, Verapamil): Concurrent use causes profound additive negative inotropic and chronotropic suppression, risking severe AV block and asystole.\n"
            "- CYP2D6 Inhibitors: Fluoxetine, Paroxetine, Bupropion significantly inhibit CYP2D6 metabolism, elevating serum Metoprolol concentrations 3- to 5-fold.\n"
            "- Digoxin: Additive AV nodal conduction slowing."
            + MEDICATION_DISCLAIMER
        ),
        "category": "Pharmacology",
    },
    {
        "instruction": "Explain how to calculate the CHA2DS2-VASc score and interpret when anticoagulation is needed.",
        "response": (
            "The CHA2DS2-VASc score stratifies annual stroke risk in patients with non-valvular Atrial Fibrillation:\n\n"
            "Scoring Breakdown (Max Score = 9):\n"
            "- C: Congestive Heart Failure (LVEF <= 40% or symptomatic HF) = +1\n"
            "- H: Hypertension (consistently > 140/90 or on antihypertensives) = +1\n"
            "- A2: Age >= 75 years = +2\n"
            "- D: Diabetes Mellitus = +1\n"
            "- S2: Stroke, TIA, or Thromboembolism history = +2\n"
            "- V: Vascular Disease (prior MI, peripheral artery disease, or complex aortic plaque) = +1\n"
            "- A: Age 65-74 years = +1\n"
            "- Sc: Sex Category Female = +1\n\n"
            "Clinical Decision Thresholds:\n"
            "- Score 0 (Men) or 1 (Women): Truly low risk. No antithrombotic therapy recommended.\n"
            "- Score 1 (Men) or 2 (Women): Intermediate risk. Oral anticoagulation should be considered based on individual bleeding risk and shared decision making.\n"
            "- Score >= 2 (Men) or >= 3 (Women): High risk. Oral anticoagulation is strongly recommended (Class I guideline indication).\n"
            "- DOACs (Apixaban, Rivaroxaban, Dabigatran) are preferred first-line over Warfarin due to a >50% reduction in fatal intracranial hemorrhage."
            + MEDICATION_DISCLAIMER
        ),
        "category": "Pharmacology",
    },
    {
        "instruction": "What is the emergency medication protocol for acute angina or chest pain at home?",
        "response": (
            "For patients with known coronary artery disease experiencing acute angina pectoris:\n\n"
            "Sublingual Nitroglycerin Protocol:\n"
            "1. Cease all physical activity immediately and sit down in a safe position (sitting prevents postural syncope from vasodilation).\n"
            "2. Place one Nitroglycerin 0.4 mg tablet (or one spray) under the tongue. Do not chew or swallow.\n"
            "3. Wait 5 minutes. If chest pain persists or worsens, immediately call 911.\n"
            "4. While waiting for emergency EMS, a second dose of 0.4 mg may be taken at 5 minutes, and a third at 10 minutes (maximum 3 doses over 15 minutes).\n\n"
            "Absolute Contraindications for Nitroglycerin:\n"
            "- Phosphodiesterase-5 (PDE-5) Inhibitors: Sildenafil (Viagra) within 24 hours or Tadalafil (Cialis) within 48 hours. Co-administration causes profound, refractory, fatal hypotension.\n"
            "- Systolic blood pressure < 90 mmHg (or > 30 mmHg drop from baseline).\n"
            "- Marked severe bradycardia (HR < 50 bpm) or suspected Right Ventricular Infarction (inferior STEMI with pre-load dependency)."
            + MEDICATION_DISCLAIMER
        ),
        "category": "Pharmacology",
    },
]

ALL_CARDIAC_QA_DATASET: List[Dict[str, str]] = load_cardiac_qa_curriculum("cardiac_health_dataset.md", max_per_category=None)

CARDIOLOGY_CURRICULUM: List[Dict[str, str]] = (
    list(BASE_CARDIOLOGY_CURRICULUM)
    + list(CONVERSATIONAL_GREETINGS)
    + list(ALL_CARDIAC_QA_DATASET)
)

"""
Mobile Export Utility: Export 1,500 Cardiac Q&A Knowledge Base to JSON
======================================================================
Creates a standalone, clean JSON file (cardiac_knowledge_base.json) containing
all 1,500 questions and answers across the 10 cardiology pillars:
1. Medications (with official Medical Disclaimer)
2. Diet and Food
3. Exercise and Walking
4. Sleep and Rest
5. Demographics
6. Body Composition
7. Substances
8. Infections
9. Hydration
10. Genetics

This JSON asset is ready to be bundled directly into:
  - iOS app bundle (Assets / Bundle.main.url(forResource: "cardiac_knowledge_base", withExtension: "json"))
  - Android assets folder (assets/cardiac_knowledge_base.json)
"""

import json
import os
import re

DATASET_MD = "cardiac_health_dataset.md"
OUTPUT_JSON = "cardiac_knowledge_base.json"
EXACT_DISCLAIMER = (
    "⚠️ **Medical Disclaimer:** For educational purposes only, not a prescription or treatment plan. "
    "**Do not start, stop, or change any medication without your doctor’s approval.** "
)


def export_json():
    if not os.path.exists(DATASET_MD):
        print(f"Error: {DATASET_MD} not found.")
        return

    with open(DATASET_MD, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = r"### Question (\d+)\s*\((.*?)\)\s*\n+\*\*Q:\*\*\s*(.*?)\n+\*\*A:\*\*\s*(.*?)(?=\n+---|### Question|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    records = []
    categories = set()

    for q_num, category, question, answer in matches:
        q_clean = question.strip()
        a_clean = answer.strip()
        cat_clean = category.strip()
        categories.add(cat_clean)

        is_medication = cat_clean.lower() == "medications" or any(
            kw in q_clean.lower()
            for kw in ["statin", "beta-blocker", "aspirin", "diuretic", "ace inhibitor", "nitrate", "anticoagulant", "antiarrhythmic", "pcsk9", "calcium channel"]
        )

        records.append({
            "id": int(q_num),
            "category": cat_clean,
            "question": q_clean,
            "answer": a_clean,
            "disclaimer_required": is_medication,
            "medical_disclaimer": EXACT_DISCLAIMER if is_medication else None,
        })

    metadata = {
        "dataset_name": "MedGemma-Micro Cardiac Health Dataset",
        "total_pairs": len(records),
        "categories": sorted(list(categories)),
        "disclaimer": EXACT_DISCLAIMER,
        "items": records,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(OUTPUT_JSON) / 1024.0
    print(f"Successfully exported {len(records)} cardiac Q&A pairs to '{OUTPUT_JSON}' ({size_kb:.1f} KB)")


if __name__ == "__main__":
    export_json()

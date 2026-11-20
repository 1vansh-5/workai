"""Interactive local agronomy review; only an expert-confirmed note enters the KB."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PENDING = ROOT / "data" / "pending_contributions.jsonl"
KB = ROOT / "data" / "knowledge.json"


def main():
    if not PENDING.exists():
        print("No contributions have been submitted yet.")
        return
    records = [json.loads(x) for x in PENDING.read_text(encoding="utf-8").splitlines() if x.strip()]
    changed = False
    for item in records:
        if item.get("status") != "pending_review":
            continue
        print("\n--- Pending contribution ---")
        for key in ("id", "crop", "problem", "observation", "suggested_advice", "source", "created_at"):
            print(f"{key}: {item.get(key, '')}")
        decision = input("Expert reviewed and approve this note? [y/N] ").strip().lower()
        if decision != "y":
            if decision == "n":
                item["status"] = "rejected"
                changed = True
            continue
        # The reviewer must supply search terms and the exact approved wording.
        aliases = [x.strip().lower() for x in input("Crop aliases, comma separated: ").split(",") if x.strip()]
        keywords = [x.strip().lower() for x in input("Search keywords, comma separated: ").split(",") if x.strip()]
        approved = input("Approved treatment notes, separated by | : ").strip()
        if not aliases or not keywords or not approved:
            print("Skipped: aliases, keywords and reviewed treatment text are required.")
            continue
        prevention = input("Prevention notes, separated by | (optional): ").strip()
        quality = input("Quality notes, separated by | (optional): ").strip()
        reviewer = input("Reviewer name or protected ID: ").strip()
        if not reviewer:
            print("Skipped: reviewer ID is required.")
            continue
        entry = {"c": aliases, "k": keywords, "n": item["problem"],
                 "cure": [x.strip() for x in approved.split("|") if x.strip()],
                 "prev": [x.strip() for x in prevention.split("|") if x.strip()],
                 "q": [x.strip() for x in quality.split("|") if x.strip()],
                 "review": {"reviewer": reviewer, "source": item["source"], "reviewed_at": __import__("datetime").date.today().isoformat()}}
        entries = json.loads(KB.read_text(encoding="utf-8-sig"))
        entries.append(entry)
        KB.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        item["status"] = "approved"
        item["reviewer"] = reviewer
        changed = True
        print("Approved note added to the local knowledge base.")
    if changed:
        PENDING.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

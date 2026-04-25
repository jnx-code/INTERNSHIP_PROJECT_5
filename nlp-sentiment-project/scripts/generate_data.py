"""
Generate synthetic sentiment dataset for training.
Run: python scripts/generate_data.py
"""
import random
import csv
import os

random.seed(42)

POSITIVE_TEMPLATES = [
    "This product is absolutely {adj}! I {verb} it so much.",
    "Totally {adj}, would {verb} to all my friends.",
    "{adj} quality, {adj} service. Highly recommend!",
    "I was {adj}ly surprised — {verb}d it completely.",
    "Best purchase I've made in years. Truly {adj}.",
    "Five stars! Everything about this is {adj}.",
    "Exceeded all my expectations. Incredibly {adj}.",
    "Amazing! {adj} and {verb}d every bit of it.",
    "Could not be happier. {adj} and {adj} experience.",
    "Outstanding! The {adj} quality speaks for itself.",
]

NEUTRAL_TEMPLATES = [
    "It's okay, does what it says. Nothing {adj}.",
    "Average product, not {adj} but not bad either.",
    "Decent enough. Could be {adj}er but works fine.",
    "It arrived on time. Seems {adj} quality for the price.",
    "Not bad, not great. Pretty {adj} overall.",
    "Does the job. Would've preferred something more {adj}.",
    "Standard quality, {adj} packaging, average delivery.",
    "Works as described. Nothing {adj} about it.",
    "I expected more. It's just {adj} at best.",
    "Acceptable. Nothing stands out — quite {adj}.",
]

NEGATIVE_TEMPLATES = [
    "Terrible product! Completely {adj} and {verb}d my expectations.",
    "Do NOT buy this. Absolutely {adj} quality.",
    "Worst purchase ever. {adj} construction, {adj} support.",
    "I'm {adj}ly disappointed. Broke after a week.",
    "Total waste of money. {adj} from day one.",
    "Horrible experience from start to finish. Very {adj}.",
    "Returned immediately. {adj} smell and {adj} finish.",
    "Zero stars if I could. {adj} all around.",
    "Cheap and {adj}. Fell apart immediately.",
    "Regret buying this. {adj} quality for any price.",
]

POS_ADJ = ["amazing", "fantastic", "brilliant", "wonderful", "superb", "excellent", "outstanding"]
POS_VERB = ["recommend", "love", "enjoy", "adore", "appreciate"]
NEU_ADJ = ["average", "ordinary", "standard", "basic", "plain", "moderate"]
NEG_ADJ = ["terrible", "awful", "dreadful", "horrible", "poor", "defective", "flimsy"]
NEG_VERB = ["shattered", "ruined", "destroyed", "betrayed", "crushed"]


def generate_review(sentiment):
    if sentiment == 2:  # positive
        tmpl = random.choice(POSITIVE_TEMPLATES)
        return tmpl.format(adj=random.choice(POS_ADJ), verb=random.choice(POS_VERB))
    elif sentiment == 1:  # neutral
        tmpl = random.choice(NEUTRAL_TEMPLATES)
        return tmpl.format(adj=random.choice(NEU_ADJ), verb=random.choice(POS_VERB))
    else:  # negative
        tmpl = random.choice(NEGATIVE_TEMPLATES)
        return tmpl.format(adj=random.choice(NEG_ADJ), verb=random.choice(NEG_VERB))


def generate_dataset(n=3000, path="data/reviews.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for _ in range(n):
        label = random.choice([0, 1, 2])
        text = generate_review(label)
        rows.append({"text": text, "label": label})

    random.shuffle(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Generated {n} reviews → {path}")
    counts = {0: 0, 1: 0, 2: 0}
    for r in rows:
        counts[r["label"]] += 1
    print(f"   Negative: {counts[0]} | Neutral: {counts[1]} | Positive: {counts[2]}")


if __name__ == "__main__":
    generate_dataset()

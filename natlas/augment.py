"""v2 training data: worker-style cases covering all ten danger signs, plus hard negatives.

Why: v1 trained only on hospital records whose single danger sign is confusion. The benchmark
showed the model then stopped recognising the other nine signs on Ibà's Nigerian vignettes
(refer-now recall 58% -> 25%). These cases teach every sign, in English and Nigerian Pidgin,
and include negated mentions ("no convulsions") so the model does not over-flag.

Every phrase is checked against Ibà's deterministic danger-sign rules
(app/backend/app/rules/danger_signs.py): a positive phrase must trigger exactly its sign, a
negative phrase must trigger none. Generated cases are screened against the 61 benchmark
vignettes so no test case (or near-copy) is trained on.
"""

import random
import re

from natlas.prompts import DANGER_SIGNS

# sign -> (English phrases, Pidgin phrases). Written from the rule patterns, not the vignettes.
POSITIVE: dict[str, tuple[list[str], list[str]]] = {
    "convulsions": (
        ["convulsed twice this morning", "had a fit at home", "jerking of the arms and legs",
         "seizure last night"],
        ["e don convulse two times", "e dey jerk for body", "e don get fits this morning"],
    ),
    "unable_to_drink_or_breastfeed": (
        ["unable to drink", "refusing to breastfeed", "not able to take fluids", "drinking poorly"],
        ["e no fit drink water", "pikin no gree suck breast", "e no dey drink at all"],
    ),
    "vomiting_everything": (
        ["vomits everything", "cannot keep anything down", "persistent vomiting since yesterday",
         "throws up all food and drugs"],
        ["e dey vomit everything", "anything wey e chop, e dey vomit am",
         "e no fit hold anything down"],
    ),
    "lethargy_or_unconsciousness": (
        ["very drowsy", "difficult to wake", "lethargic and floppy", "passed out at home",
         "not responding when called"],
        ["e dey sleep anyhow", "e no dey respond when we call am", "e don pass out"],
    ),
    "prostration": (
        ["too weak to sit", "cannot stand without support", "unable to walk since morning"],
        ["e no fit sit down", "body weak, e no fit stand", "e no fit waka"],
    ),
    "respiratory_distress": (
        ["difficulty breathing", "fast and labored breathing", "chest indrawing", "grunting",
         "shortness of breath"],
        ["breath dey cut", "e dey pant", "e dey struggle to breathe"],
    ),
    "dark_urine": (
        ["dark cola-coloured urine", "passing black urine", "urine is very dark"],
        ["e piss dey black", "urine don turn like coke"],
    ),
    "jaundice": (
        ["yellow eyes", "jaundice noticed today", "skin has turned yellow"],
        ["eye don yellow", "body dey yellow"],
    ),
    "bleeding": (
        ["bleeding from the gums", "nose bleeding", "vomiting blood", "blood in the vomit"],
        ["blood dey comot from nose", "e dey vomit blood"],
    ),
    "severe_pallor": (
        ["severe pallor", "very pale palms", "palms look white"],
        ["e palm don white", "no blood for im body"],
    ),
}

# No danger sign: negated or reassuring mentions of the same features.
NEGATIVE: tuple[list[str], list[str]] = (
    ["no convulsions", "drinking well", "eating and drinking well", "alert and playful",
     "no vomiting", "breathing normally", "urine normal colour", "no bleeding",
     "walking without help", "fully awake and talking"],
    ["e dey drink well", "e dey play", "e no vomit", "e dey waka well", "e dey talk well",
     "e dey chop well"],
)

OTHER_EN = ["headache", "body pains", "joint pains", "chills", "loss of appetite", "cough",
            "sore throat", "abdominal pain", "diarrhoea", "sweating at night"]
OTHER_PCM = ["head dey pain am", "body dey pain am", "e dey shiver", "e no dey chop",
             "e dey cough", "belle dey pain am", "e dey purge"]

RDT_EN = {"positive": "RDT positive", "negative": "RDT negative", "not_done": "no RDT done"}
RDT_PCM = {"positive": "RDT show positive", "negative": "RDT show negative",
           "not_done": "dem never do RDT"}


def _person(rng: random.Random, pidgin: bool) -> tuple[str, int]:
    child = rng.random() < 0.5
    age = rng.randint(1, 12) if child else rng.randint(16, 70)
    if pidgin:
        noun = rng.choice(["Pikin", "Small girl", "Small boy"]) if child else rng.choice(
            ["Woman", "Man", "Mama", "Oga"])
        return f"{noun} wey be {age} years", age
    noun = rng.choice(["Girl", "Boy", "Child"]) if child else rng.choice(
        ["Woman", "Man", "Adult female", "Adult male"])
    return f"{noun} {age} years", age


def _label(signs: list[str], rdt: str) -> dict:
    if signs:
        level, reason = "refer_now", "Danger sign present: refer now."
        risk = "not_applicable" if rdt == "negative" else "high"
    elif rdt == "negative":
        level, reason = "treat_monitor", "No danger signs and negative RDT: look for another cause and review."
        risk = "not_applicable"
    else:
        level, reason = "treat_monitor", "No danger signs: test, treat at PHC and review."
        risk = "standard"
    return {"triage_level": level, "danger_signs": signs, "malaria_risk": risk, "reason": reason}


def make_case(rng: random.Random, signs: list[str], pidgin: bool) -> tuple[str, dict]:
    who, _ = _person(rng, pidgin)
    days = rng.randint(1, 7)
    fever = f"body dey hot {days} days" if pidgin else f"fever {days} days"
    rdt = rng.choice(["positive", "positive", "negative", "not_done"])
    parts = [fever, (RDT_PCM if pidgin else RDT_EN)[rdt]]
    parts += rng.sample(OTHER_PCM if pidgin else OTHER_EN, k=rng.randint(0, 2))
    lang = 1 if pidgin else 0
    parts += [rng.choice(POSITIVE[s][lang]) for s in signs]
    if not signs or rng.random() < 0.3:
        parts.append(rng.choice(NEGATIVE[lang]))
    head, tail = parts[:2], parts[2:]
    rng.shuffle(tail)
    return f"{who}, " + ", ".join(head + tail), _label(signs, rdt)


def generate(n_single: int = 36, n_double: int = 60, n_none: int = 160,
             seed: int = 2027) -> list[tuple[str, dict]]:
    """n_single cases per sign, n_double two-sign cases, n_none cases without danger signs."""
    rng = random.Random(seed)
    cases = []
    for sign in DANGER_SIGNS:
        for _ in range(n_single):
            cases.append(make_case(rng, [sign], rng.random() < 0.35))
    for _ in range(n_double):
        cases.append(make_case(rng, sorted(rng.sample(DANGER_SIGNS, 2)), rng.random() < 0.35))
    for _ in range(n_none):
        cases.append(make_case(rng, [], rng.random() < 0.35))
    rng.shuffle(cases)
    return cases


_TOKEN = re.compile(r"[a-z]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def too_close(text: str, held_out: list[str], threshold: float = 0.6) -> bool:
    """Jaccard word overlap with any held-out text above `threshold`."""
    a = _tokens(text)
    for other in held_out:
        b = _tokens(other)
        if a and b and len(a & b) / len(a | b) >= threshold:
            return True
    return False

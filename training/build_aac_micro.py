"""Build a balanced AAC micro-SFT dataset for surgical recovery.

Source: train_v18aac.jsonl (4517 rows, caregiver-heavy)
Output: train_aac_micro.jsonl with ~1200 rows balanced across the 5 AAC tasks.

Composition:
  - 300 caregiver (random sample from 3440)
  - 294 text_correct (all)
  - 496 emergency (all)
  - synthesized translate (~50)
  - synthesized ask_ai (~50)

Format already matches Qwen <|im_start|>...<|im_end|> chat template, which is
identical between Qwen2.5-Coder and Qwen3-8B (both use the same special tokens
when no separate tools= arg is passed).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SRC = Path("/Users/admin/prism/training/data/train_v18aac.jsonl")
OUT = Path("/Users/admin/prism/training/data/train_aac_micro.jsonl")

CAREGIVER_KEEP = 300

TRANSLATE_SYSTEM = (
    "You are a translation engine for an AAC (augmentative and alternative "
    "communication) app. Translate the input text from the source language to "
    "the target language. Respond ONLY with the translated text, no extra "
    "commentary."
)

# Bilingual-quality phrasebook for AAC. Each row has a few accepted variants;
# we use the first as the SFT target.
TRANSLATE_PAIRS = [
    ("I am hungry", "English", "Spanish", "tengo hambre"),
    ("Help me please", "English", "Spanish", "ayúdame por favor"),
    ("Where is the bathroom", "English", "Spanish", "dónde está el baño"),
    ("I need water", "English", "Spanish", "necesito agua"),
    ("I am tired", "English", "Spanish", "estoy cansado"),
    ("I love you", "English", "Spanish", "te amo"),
    ("Thank you", "English", "Spanish", "gracias"),
    ("Stop", "English", "Spanish", "para"),
    ("More please", "English", "Spanish", "más por favor"),
    ("I am scared", "English", "Spanish", "tengo miedo"),
    ("It hurts", "English", "Spanish", "duele"),
    ("Yes", "English", "Spanish", "sí"),
    ("No", "English", "Spanish", "no"),
    ("Help", "English", "Spanish", "ayuda"),
    ("Where is mom", "English", "Spanish", "dónde está mamá"),
    ("I want to go home", "English", "Spanish", "quiero ir a casa"),
    ("I love you", "English", "French", "je t'aime"),
    ("I need water", "English", "French", "j'ai besoin d'eau"),
    ("I am hungry", "English", "French", "j'ai faim"),
    ("Help me please", "English", "French", "aidez-moi s'il vous plaît"),
    ("Where is the bathroom", "English", "French", "où sont les toilettes"),
    ("Thank you", "English", "French", "merci"),
    ("Stop", "English", "French", "arrête"),
    ("It hurts", "English", "French", "ça fait mal"),
    ("I am tired", "English", "French", "je suis fatigué"),
    ("Yes", "English", "French", "oui"),
    ("No", "English", "French", "non"),
    ("More", "English", "French", "encore"),
    ("I am scared", "English", "French", "j'ai peur"),
    ("Where is mom", "English", "French", "où est maman"),
    ("Estoy cansado", "Spanish", "English", "I am tired"),
    ("Tengo hambre", "Spanish", "English", "I am hungry"),
    ("Quiero agua", "Spanish", "English", "I want water"),
    ("Ayúdame", "Spanish", "English", "Help me"),
    ("Dónde está mi mamá", "Spanish", "English", "Where is my mom"),
    ("Me duele", "Spanish", "English", "It hurts"),
    ("Tengo miedo", "Spanish", "English", "I am scared"),
    ("Más por favor", "Spanish", "English", "More please"),
    ("Je t'aime", "French", "English", "I love you"),
    ("J'ai faim", "French", "English", "I am hungry"),
    ("Aide-moi", "French", "English", "Help me"),
    ("Où est papa", "French", "English", "Where is dad"),
    ("Ça fait mal", "French", "English", "It hurts"),
    ("Plus s'il vous plaît", "French", "English", "More please"),
    ("Je suis fatigué", "French", "English", "I am tired"),
    ("Help", "English", "Spanish", "ayuda"),
    ("Help", "English", "French", "à l'aide"),
    ("Yes", "English", "French", "oui"),
    ("Thank you", "English", "Mandarin", "谢谢"),
    ("I love you", "English", "Mandarin", "我爱你"),
    ("Help", "English", "Mandarin", "帮助"),
]

ASK_AI_SYSTEM = (
    "You are an AAC assistant answering simple factual or conversational "
    "questions for a user with motor or speech impairments. Give a short, "
    "direct answer in plain language. No tool calls, no JSON, no markdown."
)

ASK_AI_PAIRS = [
    ("What is 5 + 3?", "8."),
    ("What is 2 + 2?", "4."),
    ("What is 10 + 7?", "17."),
    ("What is 20 - 5?", "15."),
    ("What is 6 times 4?", "24."),
    ("What color is the sky?", "Blue."),
    ("What color is grass?", "Green."),
    ("What color is a banana?", "Yellow."),
    ("How many legs does a dog have?", "Four."),
    ("How many legs does a spider have?", "Eight."),
    ("How many wheels does a car have?", "Four."),
    ("What sound does a cat make?", "Cats say meow."),
    ("What sound does a dog make?", "Dogs bark — woof."),
    ("What sound does a cow make?", "Cows moo."),
    ("What sound does a duck make?", "Ducks quack."),
    ("Is fire hot or cold?", "Fire is hot."),
    ("Is ice hot or cold?", "Ice is cold."),
    ("Is the sun hot or cold?", "The sun is very hot."),
    ("Is snow hot or cold?", "Snow is cold."),
    ("Where do fish live?", "Fish live in water."),
    ("Where do birds live?", "Birds live in nests in trees."),
    ("What do bees make?", "Bees make honey."),
    ("What do cows give us?", "Cows give us milk."),
    ("What do chickens lay?", "Chickens lay eggs."),
    ("What is the opposite of hot?", "Cold."),
    ("What is the opposite of big?", "Small."),
    ("What is the opposite of happy?", "Sad."),
    ("What is the opposite of fast?", "Slow."),
    ("What is the opposite of up?", "Down."),
    ("What comes after Monday?", "Tuesday."),
    ("What comes after Friday?", "Saturday."),
    ("What month comes after January?", "February."),
    ("What month is Christmas?", "December."),
    ("Who is older, a baby or grandma?", "Grandma is older."),
    ("Who delivers mail?", "A mail carrier delivers mail."),
    ("Who teaches kids at school?", "A teacher teaches kids at school."),
    ("Who helps when you are sick?", "A doctor or nurse helps."),
    ("What do you wear on your feet?", "Shoes."),
    ("What do you wear in winter?", "A coat."),
    ("What do you eat for breakfast?", "Common breakfast foods include cereal, toast, eggs, or fruit."),
    ("Are you hungry or full after eating a big meal?", "You are full."),
    ("How many days are in a week?", "Seven."),
    ("How many hours are in a day?", "Twenty-four."),
    ("What is water made of?", "Water is made of hydrogen and oxygen."),
    ("Is the moon round or square?", "The moon is round."),
    ("What is the biggest planet?", "Jupiter."),
    ("What planet do we live on?", "Earth."),
    ("Is a whale a fish or a mammal?", "A mammal."),
    ("Do penguins fly?", "No, penguins do not fly."),
    ("What season has snow?", "Winter."),
]


def render(messages: list[dict]) -> str:
    """Render a list of role/content messages in Qwen <|im_start|>...<|im_end|> format."""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(parts)


def main():
    random.seed(42)

    # Bucket existing rows by task
    buckets = {"caregiver": [], "text_correct": [], "emergency": [], "other": []}
    with SRC.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text", "")
            i = text.find("<|im_start|>system")
            j = text.find("<|im_end|>", i)
            sys_low = text[i:j].lower() if i >= 0 else ""
            if "aac app configuration" in sys_low or "caregiver" in sys_low:
                buckets["caregiver"].append(text)
            elif "text-cleanup" in sys_low or "text correction" in sys_low:
                buckets["text_correct"].append(text)
            elif "emergency" in sys_low or "911" in sys_low:
                buckets["emergency"].append(text)
            else:
                buckets["other"].append(text)

    out_rows = []

    # Caregiver: down-sample to CAREGIVER_KEEP for balance
    caregiver_subset = random.sample(buckets["caregiver"], min(CAREGIVER_KEEP, len(buckets["caregiver"])))
    out_rows.extend(caregiver_subset)
    print(f"caregiver: {len(caregiver_subset)} of {len(buckets['caregiver'])}")

    # text_correct: all
    out_rows.extend(buckets["text_correct"])
    print(f"text_correct: {len(buckets['text_correct'])}")

    # emergency: all
    out_rows.extend(buckets["emergency"])
    print(f"emergency: {len(buckets['emergency'])}")

    # Synthesize translate examples
    n_translate = 0
    for src, from_lang, to_lang, target in TRANSLATE_PAIRS:
        sys = TRANSLATE_SYSTEM
        user = f"Translate from {from_lang} to {to_lang}: {src}"
        out_rows.append(render([
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ]))
        n_translate += 1
    print(f"translate: {n_translate}")

    # Synthesize ask_ai examples
    n_ask = 0
    for q, a in ASK_AI_PAIRS:
        out_rows.append(render([
            {"role": "system", "content": ASK_AI_SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]))
        n_ask += 1
    print(f"ask_ai: {n_ask}")

    # Shuffle
    random.shuffle(out_rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for text in out_rows:
            f.write(json.dumps({"text": text}) + "\n")

    print(f"\ntotal rows: {len(out_rows)} -> {OUT}")


if __name__ == "__main__":
    main()

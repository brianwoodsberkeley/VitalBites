"""
avoid_ingredients.py
====================
Maps health conditions to sets of actual ingredient names that should be
excluded during recipe recommendation.

Usage
-----
    from avoid_ingredients import get_avoid_ingredients

    avoid = get_avoid_ingredients(
        health_conditions={"Hypertension", "Diabetes (Type 2)"},
        ingredient_list=unique_ingredients,   # list[str] from your dataset
    )
    # avoid["unified"]  → flat set of all ingredient strings to filter out
    # avoid["by_condition"] → dict[str, list[str]] per-condition breakdown

Design
------
Each condition owns a ConditionRule that declares:
  • include_patterns : regex patterns — an ingredient matches if ANY fires
  • exclude_patterns : regex patterns — matched ingredient is dropped if ANY fires
  • qualitative_notes: human-readable rationale (for logging / explainability)

The matcher runs case-insensitively on the lower-cased ingredient name.

Sources
-------
AHA 2021 Dietary Guidance; DASH trial (Sacks, NEJM 2001); ADA Standards of
Care 2024-2026; KDOQI 2020 / KDIGO 2024; ACOG Nutrition in Pregnancy;
MIND diet (Morris, Alzheimers Dement 2015); SMILES trial (Jacka, BMC Med 2017);
WFSBP/CANMAT 2022; NPUAP/EPUAP 2019; ESPEN 2014; NIH ODS fact sheets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


# ── helpers ────────────────────────────────────────────────────────────────

def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]

def _matches(text: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


# ── rule container ─────────────────────────────────────────────────────────

@dataclass
class ConditionRule:
    include_patterns: list[str]          # ingredient is a candidate if ANY matches
    exclude_patterns: list[str] = field(default_factory=list)   # drop if ANY matches
    notes: str = ""                      # rationale for explainability

    def match(self, ingredient: str) -> bool:
        inc = _compile(self.include_patterns)
        exc = _compile(self.exclude_patterns)
        return _matches(ingredient, inc) and not _matches(ingredient, exc)


# ── condition rule bank ────────────────────────────────────────────────────
# Each key must match the strings used in health_conditions exactly.
# Multiple ConditionRules per condition are OR-ed (union).

CONDITION_RULES: dict[str, list[ConditionRule]] = {

    # ────────────────────────────────────────────────────────────────────
    # HYPERTENSION
    # DASH diet; AHA ideal sodium target ≤1 500 mg; Sacks NEJM 2001.
    # Avoid: high-sodium foods, processed meats, excessive alcohol.
    # ────────────────────────────────────────────────────────────────────
    "Hypertension": [
        ConditionRule(
            # Table/seasoning salts (low-sodium variants kept)
            include_patterns=[
                r"\bsalt\b",
                r"\bsea salt\b",
                r"\bkosher salt\b",
                r"\bseasoned salt\b",
                r"\bpickling salt\b",
            ],
            exclude_patterns=[
                r"no.?salt", r"low.sodium", r"reduced.sodium",
                r"salt substitute", r"unsalted", r"salt.free",
            ],
            notes="High-sodium table/seasoning salts — DASH/AHA",
        ),
        ConditionRule(
            include_patterns=[r"soy sauce"],
            exclude_patterns=[r"reduced.sodium", r"low.sodium", r"lite soy"],
            notes="Regular soy sauce is extremely high in sodium",
        ),
        ConditionRule(
            include_patterns=[
                r"\bbacon\b", r"\bprosciutto\b", r"\bpancetta\b",
                r"\bsalami\b", r"\bpepperoni\b",
                r"\bchorizo\b", r"\bkielbasa\b",
                r"\bpastrami\b", r"\bcorned beef\b",
                r"\bliverwurst\b", r"\bmortadella\b",
                r"\bcapicola\b", r"\bcapacola\b",
                r"andouille sausage",
                r"italian sausage",
                r"breakfast sausage",
                r"pork sausage",
                r"\bbologne?se?\b",          # bologna
            ],
            exclude_patterns=[
                r"turkey sausage", r"chicken sausage",
                r"low.fat", r"reduced.fat", r"vegetarian",
                r"low.sodium", r"reduced.sodium",
            ],
            notes="Processed meats are top sodium sources (AHA)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bham\b",
            ],
            exclude_patterns=[
                r"graham", r"low.salt", r"low.sodium",
                r"reduced.sodium", r"vegetarian",
            ],
            notes="Ham is a high-sodium cured meat",
        ),
        ConditionRule(
            include_patterns=[r"canned soup", r"cream of mushroom soup",
                               r"cream of chicken soup", r"onion soup mix",
                               r"french onion soup"],
            exclude_patterns=[r"low.sodium", r"reduced.sodium", r"no.salt"],
            notes="Canned soups are major hidden sodium sources",
        ),
        ConditionRule(
            include_patterns=[
                r"\bvodka\b", r"\bwhiskey\b", r"\bwhisky\b",
                r"\brum\b", r"\bbourbon\b", r"\bbrandy\b",
                r"\btequila\b", r"\bgin\b", r"\bschnapps\b",
                r"\babsinthe\b", r"\bcognac\b", r"\bamaretto\b",
                r"\bkahlua\b", r"\bbaileys\b", r"\bvermouth\b",
                r"\bchampagne\b", r"\bprosecco\b", r"\bsake\b",
                r"\blager\b", r"\bale\b",
            ],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"extract",
                               r"ginger ale"],
            notes="Alcohol elevates blood pressure (AHA)",
        ),
        ConditionRule(
            include_patterns=[r"\bbeer\b", r"\bwine\b"],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"ginger ale",
                               r"ginger wine"],
            notes="Alcohol elevates blood pressure (AHA)",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # HEART DISEASE
    # AHA 2021; TLC diet; Lichtenstein Circ 2006; Sacks Circ 2017.
    # Avoid: trans fats, saturated fat > 7 % kcal, sodium, refined carbs.
    # ────────────────────────────────────────────────────────────────────
    "Heart Disease": [
        ConditionRule(
            include_patterns=[
                r"\bmargarine\b", r"stick margarine", r"solid margarine",
            ],
            exclude_patterns=[
                r"light margarine", r"reduced.fat margarine",
                r"reduced.calorie", r"fat.free", r"low.fat",
                r"soft margarine", r"non.hydrogenated",
            ],
            notes="Stick margarine may contain trans fats (AHA)",
        ),
        ConditionRule(
            include_patterns=[r"shortening", r"\blard\b", r"\bsuet\b",
                               r"\btallow\b"],
            exclude_patterns=[r"butter flavor shortening", r"all.vegetable"],
            notes="Solid fats high in saturated/trans fat",
        ),
        ConditionRule(
            include_patterns=[
                r"\bbutter\b", r"heavy cream", r"heavy whipping cream",
                r"creme fraiche", r"full.fat sour cream",
            ],
            exclude_patterns=[
                r"peanut butter", r"almond butter", r"nut butter",
                r"hazelnut butter", r"cashew butter", r"sunflower butter",
                r"apple butter", r"buttercream", r"buttermilk",
                r"butternut", r"butterfly", r"cocoa butter",
                r"shea butter", r"butter bean", r"butter lettuce",
                r"low.fat", r"reduced.fat", r"light butter",
                r"butter flavor", r"butter extract",
            ],
            notes="Full-fat dairy fats raise LDL-C (TLC diet; AHA)",
        ),
        ConditionRule(
            include_patterns=[
                r"whole milk(?! cottage| mozzarella| ricotta)",
                r"whole milk mozzarella", r"whole milk ricotta",
                r"whole milk cottage",
            ],
            exclude_patterns=[r"non.dairy", r"almond", r"soy", r"oat", r"coconut milk"],
            notes="Full-fat dairy elevates LDL-C (AHA 2021)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bbacon\b", r"\bprosciutto\b", r"\bpancetta\b",
                r"\bsalami\b", r"\bpepperoni\b", r"\bchorizo\b",
                r"\bkielbasa\b", r"\bpastrami\b", r"\bcorned beef\b",
                r"\bliverwurst\b", r"\bmortadella\b",
                r"processed meat", r"deli meat",
            ],
            exclude_patterns=[r"turkey", r"chicken", r"low.fat", r"vegetarian"],
            notes="Processed red meats increase CVD risk (AHA)",
        ),
        ConditionRule(
            include_patterns=[r"\bham\b"],
            exclude_patterns=[r"graham", r"low.sodium", r"vegetarian"],
            notes="Cured ham: high sodium + saturated fat",
        ),
        ConditionRule(
            include_patterns=[r"soy sauce"],
            exclude_patterns=[r"reduced.sodium", r"low.sodium"],
            notes="Regular soy sauce: very high sodium (AHA <1500 mg)",
        ),
        ConditionRule(
            include_patterns=[
                r"palm oil", r"palm kernel", r"coconut oil",
            ],
            notes="Tropical oils high in saturated fat (AHA)",
        ),
        ConditionRule(
            include_patterns=[
                r"white bread(?! flour)", r"wonder bread",
                r"white (sandwich )?bread",
            ],
            notes="Refined carbs raise triglycerides (AHA 2021)",
        ),
        ConditionRule(
            include_patterns=[r"\bbeer\b", r"\bwine\b"],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"ginger ale"],
            notes="Alcohol in excess worsens CVD outcomes",
        ),
        ConditionRule(
            include_patterns=[
                r"\bvodka\b", r"\bwhiskey\b", r"\bwhisky\b",
                r"\brum\b", r"\bbourbon\b", r"\bbrandy\b",
                r"\btequila\b", r"\bgin\b",
            ],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"extract"],
            notes="Alcohol in excess worsens CVD outcomes",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # DIABETES (TYPE 2)
    # ADA Standards of Care 2024-2026, Section 5.
    # Avoid: added sugars, refined carbs, high-GI foods, juices.
    # ────────────────────────────────────────────────────────────────────
    "Diabetes (Type 2)": [
        ConditionRule(
            include_patterns=[
                r"\bwhite sugar\b", r"\bgranulated sugar\b",
                r"\bcane sugar\b", r"\bicing sugar\b",
                r"\bpowdered sugar\b", r"\bconfectioners.? sugar\b",
                r"\bbrown sugar\b", r"\bturbinado\b", r"\braw sugar\b",
                r"\bdemerara\b", r"\bsucanat\b",
            ],
            exclude_patterns=[r"sugar substitute", r"sugar.free",
                               r"no.sugar", r"brown sugar twin"],
            notes="Added sugars — ADA Rec 5.22: minimise",
        ),
        ConditionRule(
            include_patterns=[
                r"\bcorn syrup\b", r"high fructose corn syrup",
                r"\bmolasses\b",
            ],
            notes="High-sugar syrups spike blood glucose (ADA)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bwhite rice\b", r"cooked white rice",
                r"instant rice", r"minute rice",
            ],
            notes="High-GI white rice (ADA; glycemic index data)",
        ),
        ConditionRule(
            include_patterns=[
                r"white bread", r"white sandwich bread",
                r"wonder bread",
            ],
            exclude_patterns=[r"white bread flour"],
            notes="High-GI white bread (ADA)",
        ),
        ConditionRule(
            include_patterns=[
                r"flour tortilla", r"soft taco.size flour",
                r"flour tortilla mix",
            ],
            notes="Refined-flour tortillas: high GI (ADA)",
        ),
        ConditionRule(
            include_patterns=[
                r"fruit juice", r"orange juice", r"apple juice",
                r"grape juice", r"cranberry juice",
                r"pineapple juice", r"mango juice",
                r"juice cocktail", r"juice drink",
            ],
            exclude_patterns=[r"lemon juice", r"lime juice",
                               r"sugar.free", r"unsweetened"],
            notes="Fruit juices: fast-acting sugars without fiber (ADA Rec 5.25)",
        ),
        ConditionRule(
            include_patterns=[
                r"regular soda", r"cola soda",
                r"sweetened soda",
            ],
            notes="Sugar-sweetened sodas (ADA)",
        ),
        ConditionRule(
            include_patterns=[
                r"sweetened condensed milk",
                r"sweetened coconut cream",
            ],
            notes="Very high sugar, concentrated (ADA)",
        ),
        ConditionRule(
            include_patterns=[r"candy\b", r"\bcandy ", r"candied"],
            exclude_patterns=[r"sugar.free", r"yam", r"sweet potato"],
            notes="Candy / confectionery (ADA)",
        ),
        ConditionRule(
            include_patterns=[
                r"instant oatmeal",       # processed; quick oats raise GI
                r"instant mashed potato",
                r"potato flakes",
            ],
            notes="Highly processed starches with elevated GI (ADA)",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # KIDNEY DISEASE (CKD non-dialysis stages 3-5)
    # KDOQI 2020; KDIGO 2024. Avoid: excess protein, high phosphorus
    # (processed), high potassium (if hyperkalemic), high sodium.
    # ────────────────────────────────────────────────────────────────────
    "Kidney Disease": [
        # High-phosphorus: processed cheeses, dark colas, fast-food additives
        ConditionRule(
            include_patterns=[
                r"processed cheese", r"\bvelveeta\b",
                r"american cheese", r"kraft (singles|cheese slices)",
                r"cheese spread",
            ],
            notes="Phosphorus additives in processed cheese (KDOQI 2020)",
        ),
        ConditionRule(
            include_patterns=[r"salt substitute", r"lite salt", r"nu.?salt",
                               r"no.?salt brand", r"morton lite salt"],
            notes="Salt substitutes = KCl → dangerous in CKD (KDOQI)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bbanana\b", r"\bbananas\b", r"banana chips",
                r"banana pulp", r"banana flakes",
            ],
            exclude_patterns=[r"banana pepper", r"banana extract",
                               r"banana rum", r"banana liqueur",
                               r"banana pudding", r"banana cream",
                               r"banana instant"],
            notes="High potassium — avoid in hyperkalemia (KDOQI)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bavocado\b", r"\bavocados\b",
            ],
            exclude_patterns=[r"avocado oil"],
            notes="High potassium per serving (KDOQI)",
        ),
        ConditionRule(
            include_patterns=[r"tomato juice", r"v8 juice", r"vegetable juice"],
            notes="High potassium liquid concentrate (KDOQI)",
        ),
        ConditionRule(
            include_patterns=[r"dried apricot", r"prune juice", r"\bprunes\b",
                               r"beet juice", r"pomegranate juice"],
            notes="Concentrated potassium sources (KDOQI)",
        ),
        ConditionRule(
            include_patterns=[r"soy sauce"],
            exclude_patterns=[r"reduced.sodium", r"low.sodium"],
            notes="Extreme sodium — CKD target <2000 mg (KDOQI 2020)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bbacon\b", r"\bsalami\b", r"\bchorizo\b",
                r"\bpepperoni\b", r"deli meat", r"processed meat",
            ],
            exclude_patterns=[r"vegetarian", r"low.sodium"],
            notes="Phosphorus additives + sodium in cured meats (KDOQI)",
        ),
        # Phosphorus: nuts and seeds are moderately high; limit in CKD
        ConditionRule(
            include_patterns=[
                r"\bpumpkin seeds\b", r"\bsunflower seeds\b",
                r"\bflaxseed\b", r"\bchia seeds\b",
                r"\bcashews\b", r"\bpeanuts\b",
                r"\balmonds\b", r"\bwalnuts\b",
            ],
            notes="High phosphorus nuts/seeds — moderate portions in CKD (KDOQI)",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # WEIGHT MANAGEMENT
    # NHLBI Obesity Guidelines; Morton Br J Sports Med 2018.
    # Avoid: ultra-processed snacks, SSBs, refined carbs, excess alcohol.
    # ────────────────────────────────────────────────────────────────────
    "Weight Management": [
        ConditionRule(
            include_patterns=[
                r"\bcorn syrup\b", r"high fructose",
                r"sweetened condensed milk",
            ],
            notes="High-calorie low-nutrient sweeteners (NHLBI)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bwhite sugar\b", r"\bgranulated sugar\b",
                r"\bcane sugar\b", r"\bpowdered sugar\b",
                r"\bconfectioners.? sugar\b", r"\bbrown sugar\b",
            ],
            exclude_patterns=[r"sugar substitute", r"sugar.free"],
            notes="Added sugars contribute excess empty calories",
        ),
        ConditionRule(
            include_patterns=[
                r"shortening", r"\blard\b", r"palm oil",
                r"coconut oil",
            ],
            notes="Solid fats — calorie-dense, low nutritional value (NHLBI)",
        ),
        ConditionRule(
            include_patterns=[
                r"potato chips", r"tortilla chips", r"corn chips",
                r"cheese puffs", r"cheese doodles", r"cheetos",
                r"pork rinds", r"cracklins",
            ],
            notes="Ultra-processed snacks — high kcal density (NHLBI)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bbeer\b", r"\bwine\b",
            ],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"ginger ale"],
            notes="Alcohol: 7 kcal/g, reduces satiety signals (NHLBI)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bvodka\b", r"\bwhiskey\b", r"\bwhisky\b",
                r"\brum\b", r"\bbourbon\b", r"\bbrandy\b",
                r"\btequila\b", r"\bgin\b", r"\bliqueur\b",
            ],
            exclude_patterns=[r"extract", r"non.alcoholic"],
            notes="Alcohol: 7 kcal/g (NHLBI)",
        ),
        ConditionRule(
            include_patterns=[r"fruit juice", r"orange juice",
                               r"apple juice", r"grape juice",
                               r"cranberry juice", r"juice cocktail",
                               r"juice drink"],
            exclude_patterns=[r"lemon juice", r"lime juice",
                               r"sugar.free", r"unsweetened"],
            notes="Liquid calories — no satiety benefit (NHLBI)",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # OSTEOPOROSIS
    # Bone Health & Osteoporosis Foundation; IOM 2011 DRIs.
    # Avoid: excess sodium/caffeine (Ca excretion), alcohol, cola.
    # ────────────────────────────────────────────────────────────────────
    "Osteoporosis": [
        ConditionRule(
            include_patterns=[r"\bsalt\b", r"sea salt", r"kosher salt",
                               r"seasoned salt"],
            exclude_patterns=[r"no.salt", r"low.sodium", r"salt substitute",
                               r"unsalted", r"salt.free"],
            notes="Excess sodium increases urinary calcium excretion (BHOF)",
        ),
        ConditionRule(
            include_patterns=[r"\bcoffee\b", r"\bespresso\b"],
            exclude_patterns=[r"decaf", r"sugar.free coffee",
                               r"extract", r"chocolate"],
            notes=">3 cups coffee/day increases urinary Ca (BHOF)",
        ),
        ConditionRule(
            include_patterns=[r"\bbeer\b", r"\bwine\b"],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"ginger ale"],
            notes="Alcohol impairs bone formation (BHOF)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bvodka\b", r"\bwhiskey\b", r"\bwhisky\b",
                r"\brum\b", r"\bbourbon\b", r"\bbrandy\b",
                r"\btequila\b", r"\bgin\b", r"\bliqueur\b",
            ],
            exclude_patterns=[r"extract", r"non.alcoholic"],
            notes=">2 drinks/day impairs osteoblast activity (BHOF)",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # ANEMIA
    # NIH ODS Iron; WHO Iron-Deficiency Guidelines.
    # Avoid: tannins/phytates with iron-rich meals (flag the inhibitors).
    # ────────────────────────────────────────────────────────────────────
    "Anemia": [
        ConditionRule(
            include_patterns=[r"\bcoffee\b", r"\bespresso\b", r"\btea\b",
                               r"black tea", r"green tea"],
            exclude_patterns=[r"decaf", r"extract", r"chocolate",
                               r"tea tree"],
            notes="Tannins inhibit non-heme iron absorption (NIH ODS)",
        ),
        ConditionRule(
            include_patterns=[r"wheat bran", r"oat bran",
                               r"bran cereal", r"all.bran"],
            notes="High phytate content binds iron (NIH ODS)",
        ),
        ConditionRule(
            include_patterns=[r"raw egg white", r"egg white substitute"],
            notes="Avidin in raw egg white blocks biotin; high intake may antagonise iron",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # THYROID DISORDER
    # ATA statement; Endocrine Society hypothyroidism guidelines;
    # PMC10951571 selenium review 2024.
    # Avoid: excess iodine, goitrogens, soy (medication interference).
    # ────────────────────────────────────────────────────────────────────
    "Thyroid Disorder": [
        ConditionRule(
            include_patterns=[r"\bkelp\b", r"\bdulse\b", r"\bnori\b",
                               r"\bwakame\b", r"\bkombu\b",
                               r"seaweed", r"sea vegetable"],
            notes="Very high iodine — Wolff-Chaikoff effect (ATA)",
        ),
        ConditionRule(
            include_patterns=[r"tofu", r"tempeh", r"edamame",
                               r"\bsoymilk\b", r"soy milk",
                               r"soy protein", r"\bmiso\b"],
            notes="Soy isoflavones impair thyroid hormone absorption (Endocrine Soc.)",
        ),
        ConditionRule(
            include_patterns=[r"\bkale\b", r"\bbok choy\b",
                               r"raw broccoli", r"raw cauliflower",
                               r"raw cabbage", r"\brunabaga\b",
                               r"\bturnip\b", r"\brussels sprout"],
            notes="Raw cruciferous vegetables contain goitrogens (ATA)",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # IMMUNE DEFICIENCY
    # NIH ODS; Gombart, Nutrients 2020.
    # Avoid: excess sugar (neutrophil suppression), alcohol.
    # ────────────────────────────────────────────────────────────────────
    "Immune Deficiency": [
        ConditionRule(
            include_patterns=[
                r"\bwhite sugar\b", r"\bgranulated sugar\b",
                r"\bcane sugar\b", r"\bcorn syrup\b",
                r"high fructose", r"sweetened condensed milk",
            ],
            exclude_patterns=[r"sugar substitute", r"sugar.free"],
            notes="Excess sugar suppresses neutrophil function (NIH ODS)",
        ),
        ConditionRule(
            include_patterns=[r"\bbeer\b", r"\bwine\b"],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"ginger ale"],
            notes="Alcohol suppresses both innate and adaptive immunity",
        ),
        ConditionRule(
            include_patterns=[
                r"\bvodka\b", r"\bwhiskey\b", r"\bwhisky\b",
                r"\brum\b", r"\bbourbon\b", r"\bbrandy\b",
                r"\btequila\b", r"\bgin\b", r"\bliqueur\b",
            ],
            exclude_patterns=[r"extract", r"non.alcoholic"],
            notes="Alcohol suppresses immune response (Gombart 2020)",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # INFLAMMATION
    # Koelman, Adv Nutr 2022; AHA 2021; Calder, AJCN 2006.
    # Avoid: trans fats, refined sugars, excess omega-6 oils, processed meats.
    # ────────────────────────────────────────────────────────────────────
    "Inflammation": [
        ConditionRule(
            include_patterns=[r"shortening", r"\blard\b", r"palm oil"],
            notes="Saturated/trans-fat sources raise inflammatory markers (AHA)",
        ),
        ConditionRule(
            include_patterns=[r"margarine"],
            exclude_patterns=[r"light", r"reduced.fat", r"fat.free",
                               r"low.fat", r"non.hydrogenated"],
            notes="Trans fats in stick margarine (AHA 2021)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bwhite sugar\b", r"\bgranulated sugar\b",
                r"\bcane sugar\b", r"\bcorn syrup\b",
                r"high fructose",
            ],
            exclude_patterns=[r"sugar substitute", r"sugar.free"],
            notes="Added sugars promote pro-inflammatory cytokines (Koelman 2022)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bbacon\b", r"\bsalami\b", r"\bpepperoni\b",
                r"\bchorizo\b", r"\bkielbasa\b", r"\bpancetta\b",
                r"processed meat", r"deli meat",
            ],
            exclude_patterns=[r"turkey", r"chicken", r"low.fat", r"vegetarian"],
            notes="AGEs and nitrites in processed meats fuel inflammation",
        ),
        ConditionRule(
            include_patterns=[r"vegetable oil(?! spray)", r"corn oil",
                               r"soybean oil", r"sunflower oil",
                               r"safflower oil", r"cottonseed oil"],
            notes="High omega-6 oils skew omega-6:3 ratio (Simopoulos 2002)",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # COGNITIVE DECLINE
    # MIND diet: Morris, Alzheimers Dement 2015. MIND limits red meat,
    # butter, cheese, fried food, pastries.
    # ────────────────────────────────────────────────────────────────────
    "Cognitive Decline": [
        ConditionRule(
            include_patterns=[r"\bbutter\b"],
            exclude_patterns=[
                r"peanut butter", r"almond butter", r"nut butter",
                r"hazelnut butter", r"cashew butter", r"sunflower butter",
                r"apple butter", r"buttercream", r"buttermilk",
                r"butternut", r"butterfly", r"cocoa butter",
                r"butter bean", r"butter lettuce",
                r"butter flavor", r"butter extract",
                r"light butter", r"low.fat",
            ],
            notes="MIND diet: butter <1 tbsp/day",
        ),
        ConditionRule(
            include_patterns=[r"margarine"],
            exclude_patterns=[r"light", r"non.hydrogenated", r"reduced.fat"],
            notes="MIND diet: avoid margarine (trans-fat proxy)",
        ),
        ConditionRule(
            include_patterns=[r"shortening", r"\blard\b"],
            notes="MIND diet: saturated/trans fat avoided",
        ),
        ConditionRule(
            include_patterns=[
                r"\bbacon\b", r"\bsalami\b", r"\bchorizo\b",
                r"\bkielbasa\b", r"\bpancetta\b",
                r"\bpepperoni\b", r"processed meat",
            ],
            exclude_patterns=[r"turkey", r"chicken", r"low.fat", r"vegetarian"],
            notes="MIND diet: red/processed meat <4 servings/week",
        ),
        ConditionRule(
            include_patterns=[
                r"\bwhite sugar\b", r"\bgranulated sugar\b",
                r"\bcane sugar\b", r"\bcorn syrup\b",
                r"high fructose",
            ],
            exclude_patterns=[r"sugar substitute", r"sugar.free"],
            notes="MIND diet: pastries/sweets <5/week — proxy on added sugars",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # DEPRESSION
    # SMILES trial (Jacka, BMC Med 2017); WFSBP/CANMAT 2022; Liao 2019.
    # Avoid: ultra-processed foods, excess sugar, alcohol.
    # ────────────────────────────────────────────────────────────────────
    "Depression": [
        ConditionRule(
            include_patterns=[
                r"\bwhite sugar\b", r"\bgranulated sugar\b",
                r"\bcane sugar\b", r"\bcorn syrup\b",
                r"high fructose", r"sweetened condensed milk",
            ],
            exclude_patterns=[r"sugar substitute", r"sugar.free"],
            notes="Refined sugar increases depression risk (SMILES trial)",
        ),
        ConditionRule(
            include_patterns=[r"\bbeer\b", r"\bwine\b"],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"ginger ale"],
            notes="Alcohol depletes folate, B12, zinc (WFSBP 2022)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bvodka\b", r"\bwhiskey\b", r"\bwhisky\b",
                r"\brum\b", r"\bbourbon\b", r"\bbrandy\b",
                r"\btequila\b", r"\bgin\b", r"\bliqueur\b",
            ],
            exclude_patterns=[r"extract", r"non.alcoholic"],
            notes="Alcohol depletes mood-relevant B vitamins (WFSBP 2022)",
        ),
        ConditionRule(
            include_patterns=[
                r"margarine", r"shortening", r"\blard\b",
                r"palm oil",
            ],
            exclude_patterns=[r"light", r"reduced.fat", r"non.hydrogenated"],
            notes="Trans/saturated fats linked to worse depression outcomes",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # FATIGUE
    # Tardy, Nutrients 2020; Harvard Health (B12, Mg, Fe).
    # Avoid: refined sugar energy crashes, extreme caffeine, heavy fats.
    # ────────────────────────────────────────────────────────────────────
    "Fatigue": [
        ConditionRule(
            include_patterns=[
                r"\bwhite sugar\b", r"\bgranulated sugar\b",
                r"\bcane sugar\b", r"\bcorn syrup\b",
                r"high fructose",
            ],
            exclude_patterns=[r"sugar substitute", r"sugar.free"],
            notes="Refined sugars cause energy spike-and-crash (Tardy 2020)",
        ),
        ConditionRule(
            include_patterns=[r"\bcoffee\b", r"\bespresso\b"],
            exclude_patterns=[r"decaf", r"chocolate", r"extract"],
            notes="Excessive caffeine causes rebound fatigue (>400 mg/day)",
        ),
        ConditionRule(
            include_patterns=[r"\bbeer\b", r"\bwine\b"],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"ginger ale"],
            notes="Alcohol is a CNS depressant — worsens fatigue (Tardy 2020)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bvodka\b", r"\bwhiskey\b", r"\bwhisky\b",
                r"\brum\b", r"\bbourbon\b", r"\bbrandy\b",
                r"\btequila\b", r"\bgin\b", r"\bliqueur\b",
            ],
            exclude_patterns=[r"extract", r"non.alcoholic"],
            notes="Alcohol worsens fatigue and sleep quality",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # CRAMPS
    # Nosaka, JISSN 2021; NIH ODS Mg, K, Ca fact sheets.
    # Avoid: diuretics that deplete electrolytes (caffeine, alcohol).
    # ────────────────────────────────────────────────────────────────────
    "Cramps": [
        ConditionRule(
            include_patterns=[r"\bcoffee\b", r"\bespresso\b"],
            exclude_patterns=[r"decaf", r"chocolate", r"extract"],
            notes="Caffeine is a diuretic — promotes electrolyte loss (Nosaka 2021)",
        ),
        ConditionRule(
            include_patterns=[r"\bbeer\b", r"\bwine\b"],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"ginger ale"],
            notes="Alcohol depletes magnesium and potassium (NIH ODS)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bvodka\b", r"\bwhiskey\b", r"\bwhisky\b",
                r"\brum\b", r"\bbourbon\b", r"\bbrandy\b",
                r"\btequila\b", r"\bgin\b", r"\bliqueur\b",
            ],
            exclude_patterns=[r"extract", r"non.alcoholic"],
            notes="Alcohol diuretic effect depletes Mg/K (NIH ODS)",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # MUSCLE WEAKNESS
    # ESPEN 2014; EWGSOP2 2019 sarcopenia criteria.
    # Avoid: alcohol (myopathy), crash-diet proxies.
    # ────────────────────────────────────────────────────────────────────
    "Muscle Weakness": [
        ConditionRule(
            include_patterns=[r"\bbeer\b", r"\bwine\b"],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"ginger ale"],
            notes="Alcohol myopathy accelerates muscle breakdown (ESPEN)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bvodka\b", r"\bwhiskey\b", r"\bwhisky\b",
                r"\brum\b", r"\bbourbon\b", r"\bbrandy\b",
                r"\btequila\b", r"\bgin\b", r"\bliqueur\b",
            ],
            exclude_patterns=[r"extract", r"non.alcoholic"],
            notes="Alcoholic beverages cause alcoholic myopathy (ESPEN 2014)",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # WOUND HEALING
    # NPUAP/EPUAP 2019; ASPEN. Avoid: excess sugar, alcohol.
    # ────────────────────────────────────────────────────────────────────
    "Wound Healing": [
        ConditionRule(
            include_patterns=[
                r"\bwhite sugar\b", r"\bgranulated sugar\b",
                r"\bcane sugar\b", r"\bcorn syrup\b",
                r"high fructose", r"sweetened condensed milk",
            ],
            exclude_patterns=[r"sugar substitute", r"sugar.free"],
            notes="Hyperglycaemia impairs immune-mediated wound healing (NPUAP 2019)",
        ),
        ConditionRule(
            include_patterns=[r"\bbeer\b", r"\bwine\b"],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"ginger ale"],
            notes="Alcohol depletes zinc/selenium; impairs collagen synthesis (ASPEN)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bvodka\b", r"\bwhiskey\b", r"\bwhisky\b",
                r"\brum\b", r"\bbourbon\b", r"\bbrandy\b",
                r"\btequila\b", r"\bgin\b", r"\bliqueur\b",
            ],
            exclude_patterns=[r"extract", r"non.alcoholic"],
            notes="Alcohol impairs wound healing (ASPEN)",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # HAIR LOSS
    # PMC6380979; PMC5315033; AAD guidelines.
    # Avoid: excess vitamin A (preformed), raw egg whites (avidin/biotin).
    # ────────────────────────────────────────────────────────────────────
    "Hair Loss": [
        ConditionRule(
            include_patterns=[r"\bliver\b", r"calf liver", r"beef liver",
                               r"chicken liver", r"duck liver",
                               r"baby beef liver", r"cod liver oil"],
            exclude_patterns=[r"liver sausage", r"liverwurst"],
            notes="Excess preformed vitamin A (retinol) triggers telogen effluvium (AAD)",
        ),
        ConditionRule(
            include_patterns=[r"raw egg white", r"egg white substitute"],
            notes="Avidin in raw egg whites competitively inhibits biotin (NIH ODS)",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # SKIN CONDITIONS
    # AAD / JAAD Int 2022 (dairy + high-GI → acne);
    # PMC7892455 (omega-3 in psoriasis/AD); Dermatology Times 2025.
    # ────────────────────────────────────────────────────────────────────
    "Skin Conditions": [
        ConditionRule(
            include_patterns=[
                r"\bskim milk\b", r"\bmilk\b", r"\bdairy\b",
                r"\byogurt\b", r"\byoghurt\b",
            ],
            exclude_patterns=[
                r"non.dairy", r"almond milk", r"soy milk",
                r"oat milk", r"coconut milk", r"rice milk",
                r"coconut yogurt", r"soy yogurt", r"almond yogurt",
            ],
            notes="Dairy (especially skim milk) associated with acne (JAAD 2022)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bwhite sugar\b", r"\bgranulated sugar\b",
                r"\bcane sugar\b", r"\bcorn syrup\b",
                r"high fructose",
            ],
            exclude_patterns=[r"sugar substitute", r"sugar.free"],
            notes="High-GI/high-sugar diet promotes IGF-1 → acne (JAAD 2022)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bwhite rice\b", r"cooked white rice",
                r"white bread", r"flour tortilla",
            ],
            notes="High-GI foods spike insulin/IGF-1 (JAAD 2022)",
        ),
        ConditionRule(
            include_patterns=[r"\bbeer\b", r"\bwine\b"],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"ginger ale"],
            notes="Alcohol triggers rosacea flares; pro-inflammatory (AAD)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bvodka\b", r"\bwhiskey\b", r"\bwhisky\b",
                r"\brum\b", r"\bbourbon\b", r"\bbrandy\b",
                r"\btequila\b", r"\bgin\b", r"\bliqueur\b",
            ],
            exclude_patterns=[r"extract", r"non.alcoholic"],
            notes="Alcohol triggers rosacea; pro-inflammatory (AAD)",
        ),
    ],

    # ────────────────────────────────────────────────────────────────────
    # PREGNANCY NUTRITION
    # ACOG; IOM 2006; FDA/EPA fish advisory 2024.
    # Avoid: raw proteins (Listeria/Salmonella), high-mercury fish, alcohol.
    # ────────────────────────────────────────────────────────────────────
    "Pregnancy Nutrition": [
        ConditionRule(
            include_patterns=[r"\bbeer\b", r"\bwine\b"],
            exclude_patterns=[r"non.alcoholic", r"vinegar", r"ginger ale",
                               r"ginger beer", r"ginger wine"],
            notes="No safe amount of alcohol in pregnancy (ACOG)",
        ),
        ConditionRule(
            include_patterns=[
                r"\bvodka\b", r"\bwhiskey\b", r"\bwhisky\b",
                r"\brum\b", r"\bbourbon\b", r"\bbrandy\b",
                r"\btequila\b", r"\bgin\b", r"\bliqueur\b",
                r"\bchampagne\b", r"\bprosecco\b",
            ],
            exclude_patterns=[
                r"extract", r"non.alcoholic",
                r"champagne vinegar", r"champagne yeast",
                r"champagne grape", r"ginger beer",
            ],
            notes="No safe amount of alcohol in pregnancy (ACOG)",
        ),
        ConditionRule(
            include_patterns=[
                r"swordfish", r"shark", r"king mackerel",
                r"tilefish", r"bigeye tuna", r"orange roughy",
                r"marlin",
            ],
            notes="High-mercury fish — FDA/EPA 2024 Advice for Pregnant Women",
        ),
        ConditionRule(
            include_patterns=[r"roquefort", r"camembert", r"brie",
                               r"blue cheese", r"gorgonzola",
                               r"unpasteurized"],
            notes="Unpasteurised / mould-ripened cheeses — Listeria risk (ACOG)",
        ),
        ConditionRule(
            include_patterns=[r"raw egg", r"raw egg yolk"],
            notes="Raw eggs — Salmonella risk in pregnancy (ACOG)",
        ),
        ConditionRule(
            include_patterns=[r"\bliver\b", r"calf liver", r"beef liver",
                               r"chicken liver", r"duck liver",
                               r"baby beef liver", r"cod liver oil"],
            exclude_patterns=[r"liverwurst", r"liver sausage"],
            notes="Preformed vitamin A >10 000 IU is teratogenic (ACOG)",
        ),
    ],
}


# ── public API ─────────────────────────────────────────────────────────────

def get_avoid_ingredients(
    health_conditions: set[str],
    ingredient_list: Iterable[str],
) -> dict:
    """
    Map health conditions to actual ingredients from the recipe dataset
    that should be excluded during recommendation.

    Parameters
    ----------
    health_conditions : set[str]
        Subset of recognised condition keys (same strings as
        compute_nutrient_targets).
    ingredient_list : Iterable[str]
        All unique ingredient names in your dataset (raw case preserved).

    Returns
    -------
    dict with two keys:
        "by_condition" : dict[str, list[str]]
            Per-condition lists of matching ingredient names.
        "unified" : set[str]
            Union of all condition lists — use this as the filter set
            in your recommendation pipeline.
    """
    # Normalise to lowercase for matching; preserve originals for output
    norm_to_orig: dict[str, str] = {}
    for ing in ingredient_list:
        norm_to_orig[ing.strip().lower().strip('"')] = ing.strip().strip('"')

    normed: list[str] = list(norm_to_orig.keys())

    by_condition: dict[str, list[str]] = {}

    for condition in health_conditions:
        rules = CONDITION_RULES.get(condition, [])
        matched_normed: set[str] = set()

        for rule in rules:
            for ing_norm in normed:
                if rule.match(ing_norm):
                    matched_normed.add(ing_norm)

        # Map back to original-case names
        by_condition[condition] = sorted(
            norm_to_orig[n] for n in matched_normed
        )

    unified: set[str] = set()
    for lst in by_condition.values():
        unified.update(lst)

    return {"by_condition": by_condition, "unified": unified}

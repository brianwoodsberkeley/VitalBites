// All ailments organized by category
export const AILMENTS_BY_CATEGORY = {
  "Cardiovascular": [
    { id: 1, name: "Hypertension (high blood pressure)", restrictions: "sodium, alcohol" },
    { id: 2, name: "Heart disease / CHF", restrictions: "sodium, saturated fat, cholesterol" },
    { id: 3, name: "High cholesterol / dyslipidemia", restrictions: "saturated fat, trans fat" },
  ],
  "Metabolic / Endocrine": [
    { id: 4, name: "Pre-diabetes / diabetes", restrictions: "carbs, sugar, glycemic index" },
    { id: 5, name: "Obesity", restrictions: "calories, portion control" },
    { id: 6, name: "Hypothyroidism", restrictions: "goitrogens (soy, cruciferous in excess), iodine balance" },
    { id: 7, name: "Hyperthyroidism", restrictions: "iodine, caffeine" },
  ],
  "Kidney": [
    { id: 8, name: "Chronic kidney disease (CKD)", restrictions: "protein, sodium, potassium, phosphorus" },
    { id: 9, name: "Kidney stones", restrictions: "oxalates, purines, sodium, animal protein" },
    { id: 10, name: "Gout", restrictions: "purines, alcohol, fructose" },
  ],
  "Gastrointestinal": [
    { id: 11, name: "Celiac disease", restrictions: "gluten (strict elimination)" },
    { id: 12, name: "Crohn's / ulcerative colitis (IBD)", restrictions: "varies, often fiber during flares" },
    { id: 13, name: "IBS", restrictions: "FODMAPs, trigger foods" },
    { id: 14, name: "GERD / acid reflux", restrictions: "acidic foods, caffeine, alcohol, fatty foods" },
    { id: 15, name: "Diverticulitis", restrictions: "fiber management" },
    { id: 16, name: "Gastroparesis", restrictions: "fat, fiber, large meals" },
    { id: 17, name: "Fatty liver (NAFLD)", restrictions: "sugar, alcohol, saturated fat" },
  ],
  "Allergies & Intolerances": [
    { id: 18, name: "Peanut allergy", restrictions: "peanuts (strict avoidance)" },
    { id: 19, name: "Tree nut allergy", restrictions: "tree nuts (strict avoidance)" },
    { id: 20, name: "Shellfish allergy", restrictions: "shellfish (strict avoidance)" },
    { id: 21, name: "Egg allergy", restrictions: "eggs (strict avoidance)" },
    { id: 22, name: "Dairy allergy", restrictions: "dairy (strict avoidance)" },
    { id: 23, name: "Soy allergy", restrictions: "soy (strict avoidance)" },
    { id: 24, name: "Wheat allergy", restrictions: "wheat (strict avoidance)" },
    { id: 25, name: "Fish allergy", restrictions: "fish (strict avoidance)" },
    { id: 26, name: "Lactose intolerance", restrictions: "dairy/lactose" },
    { id: 27, name: "Histamine intolerance", restrictions: "aged foods, fermented foods" },
  ],
  "Autoimmune / Inflammatory": [
    { id: 28, name: "Rheumatoid arthritis", restrictions: "some follow anti-inflammatory diets" },
    { id: 29, name: "Lupus", restrictions: "varies, sometimes sodium, alcohol" },
    { id: 30, name: "Multiple sclerosis", restrictions: "some follow Wahls or similar protocols" },
  ],
  "Neurological": [
    { id: 31, name: "Epilepsy", restrictions: "ketogenic diet (therapeutic)" },
    { id: 32, name: "Migraines", restrictions: "tyramine, MSG, alcohol, caffeine triggers" },
    { id: 33, name: "Phenylketonuria (PKU)", restrictions: "phenylalanine (strict)" },
  ],
  "Other": [
    { id: 34, name: "Osteoporosis", restrictions: "calcium, vitamin D, limit sodium/caffeine" },
    { id: 35, name: "Anemia", restrictions: "iron, B12, vitamin C for absorption" },
    { id: 36, name: "Wilson's disease", restrictions: "copper" },
    { id: 37, name: "Hemochromatosis", restrictions: "iron" },
    { id: 38, name: "Cancer (during treatment)", restrictions: "varies widely, often immune-compromised diet" },
    { id: 39, name: "Eating disorders (recovery)", restrictions: "structured meal plans" },
  ],
  "Medication Interactions": [
    { id: 40, name: "Warfarin", restrictions: "vitamin K consistency" },
    { id: 41, name: "MAOIs", restrictions: "tyramine" },
    { id: 42, name: "Some statins", restrictions: "grapefruit" },
  ],
};

// Flat list of all ailments
export const ALL_AILMENTS = Object.entries(AILMENTS_BY_CATEGORY).flatMap(
  ([category, ailments]) => ailments.map(a => ({ ...a, category }))
);

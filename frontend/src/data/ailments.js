// All ailments organized by category
export const AILMENTS_BY_CATEGORY = {
  "Cardiovascular": [
    { id: 1, name: "Hypertension", needs: "potassium, magnesium, calcium, fiber", avoid: "sodium, saturated fat" },
    { id: 2, name: "Heart Disease", needs: "fiber, potassium, magnesium, monounsaturated fat, polyunsaturated fat", avoid: "sodium, saturated fat" },
  ],
  "Metabolic / Endocrine": [
    { id: 3, name: "Diabetes (Type 2)", needs: "fiber, magnesium, chromium", avoid: "carbohydrate, saturated fat" },
    { id: 4, name: "Thyroid Disorder", needs: "selenium, zinc, iron" },
    { id: 5, name: "Weight Management", needs: "protein, fiber", avoid: "saturated fat, carbohydrate" },
  ],
  "Kidney & Bone": [
    { id: 6, name: "Kidney Disease", needs: "protein", avoid: "sodium, potassium, phosphorus" },
    { id: 7, name: "Osteoporosis", needs: "calcium, vitamin D, magnesium, phosphorus, vitamin K1", avoid: "sodium" },
  ],
  "Digestive": [
    { id: 8, name: "Digestive Issues", needs: "fiber, zinc, protein", avoid: "saturated fat" },
  ],
  "Blood & Immunity": [
    { id: 9, name: "Anemia", needs: "iron, vitamin B12, folate, vitamin C, copper", avoid: "calcium" },
    { id: 10, name: "Immune Deficiency", needs: "zinc, selenium, vitamin C, vitamin D, iron, protein" },
    { id: 11, name: "Inflammation", needs: "polyunsaturated fat, selenium, zinc, vitamin C, vitamin E", avoid: "saturated fat" },
  ],
  "Neurological & Mental Health": [
    { id: 12, name: "Cognitive Decline", needs: "polyunsaturated fat, vitamin B12, iron, zinc, selenium", avoid: "saturated fat" },
    { id: 13, name: "Depression", needs: "polyunsaturated fat, magnesium, zinc, iron, selenium, vitamin B12" },
    { id: 14, name: "Fatigue", needs: "iron, vitamin B12, magnesium, protein, carbohydrate" },
  ],
  "Musculoskeletal": [
    { id: 15, name: "Cramps", needs: "magnesium, potassium, calcium, sodium" },
    { id: 16, name: "Muscle Weakness", needs: "protein, magnesium, potassium, calcium, vitamin D" },
    { id: 17, name: "Wound Healing", needs: "zinc, protein, vitamin C, iron, copper" },
  ],
  "Skin & Hair": [
    { id: 18, name: "Hair Loss", needs: "iron, zinc, protein, selenium, vitamin B12" },
    { id: 19, name: "Skin Conditions", needs: "zinc, selenium, vitamin C, vitamin E, protein" },
  ],
  "Women's Health": [
    { id: 20, name: "Pregnancy Nutrition", needs: "iron, calcium, folate, protein, zinc, magnesium" },
  ],
};

// Flat list of all ailments
export const ALL_AILMENTS = Object.entries(AILMENTS_BY_CATEGORY).flatMap(
  ([category, ailments]) => ailments.map(a => ({ ...a, category }))
);

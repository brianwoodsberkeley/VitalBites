AILMENTS_SEED_DATA = [
    # Cardiovascular
    {"name": "Hypertension", "category": "Cardiovascular", "needs": "potassium,magnesium,calcium,fiber", "avoid": "sodium,saturated_fat"},
    {"name": "Heart Disease", "category": "Cardiovascular", "needs": "fiber,potassium,magnesium,monounsaturated_fat,polyunsaturated_fat", "avoid": "sodium,saturated_fat"},

    # Metabolic / Endocrine
    {"name": "Diabetes (Type 2)", "category": "Metabolic / Endocrine", "needs": "fiber,magnesium,chromium", "avoid": "carbohydrate,saturated_fat"},
    {"name": "Thyroid Disorder", "category": "Metabolic / Endocrine", "needs": "selenium,zinc,iron", "avoid": ""},
    {"name": "Weight Management", "category": "Metabolic / Endocrine", "needs": "protein,fiber", "avoid": "saturated_fat,carbohydrate"},

    # Kidney & Bone
    {"name": "Kidney Disease", "category": "Kidney & Bone", "needs": "protein", "avoid": "sodium,potassium,phosphorus"},
    {"name": "Osteoporosis", "category": "Kidney & Bone", "needs": "calcium,vitamin_d,magnesium,phosphorus,vitamin_k1", "avoid": "sodium"},

    # Digestive
    {"name": "Digestive Issues", "category": "Digestive", "needs": "fiber,zinc,protein", "avoid": "saturated_fat"},

    # Blood & Immunity
    {"name": "Anemia", "category": "Blood & Immunity", "needs": "iron,vitamin_b12,vitamin_b9_folate,vitamin_c,copper", "avoid": "calcium"},
    {"name": "Immune Deficiency", "category": "Blood & Immunity", "needs": "zinc,selenium,vitamin_c,vitamin_d,iron,protein", "avoid": ""},
    {"name": "Inflammation", "category": "Blood & Immunity", "needs": "polyunsaturated_fat,selenium,zinc,vitamin_c,vitamin_e", "avoid": "saturated_fat"},

    # Neurological & Mental Health
    {"name": "Cognitive Decline", "category": "Neurological & Mental Health", "needs": "polyunsaturated_fat,vitamin_b12,iron,zinc,selenium", "avoid": "saturated_fat"},
    {"name": "Depression", "category": "Neurological & Mental Health", "needs": "polyunsaturated_fat,magnesium,zinc,iron,selenium,vitamin_b12", "avoid": ""},
    {"name": "Fatigue", "category": "Neurological & Mental Health", "needs": "iron,vitamin_b12,magnesium,protein,carbohydrate", "avoid": ""},

    # Musculoskeletal
    {"name": "Cramps", "category": "Musculoskeletal", "needs": "magnesium,potassium,calcium,sodium", "avoid": ""},
    {"name": "Muscle Weakness", "category": "Musculoskeletal", "needs": "protein,magnesium,potassium,calcium,vitamin_d", "avoid": ""},
    {"name": "Wound Healing", "category": "Musculoskeletal", "needs": "zinc,protein,vitamin_c,iron,copper", "avoid": ""},

    # Skin & Hair
    {"name": "Hair Loss", "category": "Skin & Hair", "needs": "iron,zinc,protein,selenium,vitamin_b12", "avoid": ""},
    {"name": "Skin Conditions", "category": "Skin & Hair", "needs": "zinc,selenium,vitamin_c,vitamin_e,protein", "avoid": ""},

    # Women's Health
    {"name": "Pregnancy Nutrition", "category": "Women's Health", "needs": "iron,calcium,vitamin_b9_folate,protein,zinc,magnesium", "avoid": ""},
]

def seed_ailments(db):
    from .models import Ailment

    # Check if already seeded
    existing = db.query(Ailment).first()
    if existing:
        # If old data exists (42 ailments), clear and re-seed
        count = db.query(Ailment).count()
        if count != len(AILMENTS_SEED_DATA):
            db.query(Ailment).delete()
            db.commit()
        else:
            return

    for ailment_data in AILMENTS_SEED_DATA:
        ailment = Ailment(**ailment_data)
        db.add(ailment)

    db.commit()

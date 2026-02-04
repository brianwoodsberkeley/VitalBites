AILMENTS_SEED_DATA = [
    # Cardiovascular
    {"name": "Hypertension (high blood pressure)", "category": "Cardiovascular", "dietary_restrictions": "sodium,alcohol"},
    {"name": "Heart disease / CHF", "category": "Cardiovascular", "dietary_restrictions": "sodium,saturated_fat,cholesterol"},
    {"name": "High cholesterol / dyslipidemia", "category": "Cardiovascular", "dietary_restrictions": "saturated_fat,trans_fat"},
    
    # Metabolic / Endocrine
    {"name": "Pre-diabetes / diabetes", "category": "Metabolic / Endocrine", "dietary_restrictions": "carbs,sugar,high_glycemic"},
    {"name": "Obesity", "category": "Metabolic / Endocrine", "dietary_restrictions": "calories,large_portions"},
    {"name": "Hypothyroidism", "category": "Metabolic / Endocrine", "dietary_restrictions": "goitrogens,soy,excess_cruciferous,iodine_imbalance"},
    {"name": "Hyperthyroidism", "category": "Metabolic / Endocrine", "dietary_restrictions": "iodine,caffeine"},
    
    # Kidney
    {"name": "Chronic kidney disease (CKD)", "category": "Kidney", "dietary_restrictions": "protein,sodium,potassium,phosphorus"},
    {"name": "Kidney stones", "category": "Kidney", "dietary_restrictions": "oxalates,purines,sodium,animal_protein"},
    {"name": "Gout", "category": "Kidney", "dietary_restrictions": "purines,alcohol,fructose"},
    
    # Gastrointestinal
    {"name": "Celiac disease", "category": "Gastrointestinal", "dietary_restrictions": "gluten"},
    {"name": "Crohn's / ulcerative colitis (IBD)", "category": "Gastrointestinal", "dietary_restrictions": "fiber_during_flares"},
    {"name": "IBS", "category": "Gastrointestinal", "dietary_restrictions": "fodmaps,trigger_foods"},
    {"name": "GERD / acid reflux", "category": "Gastrointestinal", "dietary_restrictions": "acidic_foods,caffeine,alcohol,fatty_foods"},
    {"name": "Diverticulitis", "category": "Gastrointestinal", "dietary_restrictions": "fiber_management"},
    {"name": "Gastroparesis", "category": "Gastrointestinal", "dietary_restrictions": "fat,fiber,large_meals"},
    {"name": "Fatty liver (NAFLD)", "category": "Gastrointestinal", "dietary_restrictions": "sugar,alcohol,saturated_fat"},
    
    # Allergies & Intolerances
    {"name": "Peanut allergy", "category": "Allergies & Intolerances", "dietary_restrictions": "peanuts"},
    {"name": "Tree nut allergy", "category": "Allergies & Intolerances", "dietary_restrictions": "tree_nuts"},
    {"name": "Shellfish allergy", "category": "Allergies & Intolerances", "dietary_restrictions": "shellfish"},
    {"name": "Egg allergy", "category": "Allergies & Intolerances", "dietary_restrictions": "eggs"},
    {"name": "Dairy allergy", "category": "Allergies & Intolerances", "dietary_restrictions": "dairy"},
    {"name": "Soy allergy", "category": "Allergies & Intolerances", "dietary_restrictions": "soy"},
    {"name": "Wheat allergy", "category": "Allergies & Intolerances", "dietary_restrictions": "wheat"},
    {"name": "Fish allergy", "category": "Allergies & Intolerances", "dietary_restrictions": "fish"},
    {"name": "Lactose intolerance", "category": "Allergies & Intolerances", "dietary_restrictions": "dairy,lactose"},
    {"name": "Histamine intolerance", "category": "Allergies & Intolerances", "dietary_restrictions": "aged_foods,fermented_foods"},
    
    # Autoimmune / Inflammatory
    {"name": "Rheumatoid arthritis", "category": "Autoimmune / Inflammatory", "dietary_restrictions": "inflammatory_foods"},
    {"name": "Lupus", "category": "Autoimmune / Inflammatory", "dietary_restrictions": "sodium,alcohol"},
    {"name": "Multiple sclerosis", "category": "Autoimmune / Inflammatory", "dietary_restrictions": "wahls_protocol"},
    
    # Neurological
    {"name": "Epilepsy", "category": "Neurological", "dietary_restrictions": "requires_ketogenic"},
    {"name": "Migraines", "category": "Neurological", "dietary_restrictions": "tyramine,msg,alcohol,caffeine"},
    {"name": "Phenylketonuria (PKU)", "category": "Neurological", "dietary_restrictions": "phenylalanine"},
    
    # Other
    {"name": "Osteoporosis", "category": "Other", "dietary_restrictions": "needs_calcium,needs_vitamin_d,sodium,caffeine"},
    {"name": "Anemia", "category": "Other", "dietary_restrictions": "needs_iron,needs_b12,needs_vitamin_c"},
    {"name": "Wilson's disease", "category": "Other", "dietary_restrictions": "copper"},
    {"name": "Hemochromatosis", "category": "Other", "dietary_restrictions": "iron"},
    {"name": "Cancer (during treatment)", "category": "Other", "dietary_restrictions": "immune_compromised_diet"},
    {"name": "Eating disorders (recovery)", "category": "Other", "dietary_restrictions": "structured_meal_plans"},
    
    # Medication Interactions
    {"name": "Warfarin", "category": "Medication Interactions", "dietary_restrictions": "vitamin_k_consistency"},
    {"name": "MAOIs", "category": "Medication Interactions", "dietary_restrictions": "tyramine"},
    {"name": "Some statins", "category": "Medication Interactions", "dietary_restrictions": "grapefruit"},
]

def seed_ailments(db):
    from .models import Ailment
    
    # Check if already seeded
    if db.query(Ailment).first():
        return
    
    for ailment_data in AILMENTS_SEED_DATA:
        ailment = Ailment(**ailment_data)
        db.add(ailment)
    
    db.commit()

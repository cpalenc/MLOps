from pydantic import BaseModel

class PatientInfo(BaseModel):
    race: str
    gender: str
    age: int = 0
    weight: float = 0.0
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 1
    time_in_hospital: int = 0
    num_lab_procedures: int = 0
    num_procedures: int = 0
    num_medications: int = 0
    number_outpatient: int = 0
    number_emergency: int = 0
    number_inpatient: int = 0
    diag_1: str = ""
    diag_2: str = ""
    diag_3: str = ""
    number_diagnoses: int = 0
    max_glu_serum: str = ""
    A1Cresult: str = ""
    metformin: str = ""
    repaglinide: str = ""
    nateglinide: str = ""
    chlorpropamide: str = ""
    glimepiride: str = ""
    acetohexamide: str = ""
    glipizide: str = ""
    glyburide: str = ""
    tolbutamide: str = ""
    pioglitazone: str = ""
    rosiglitazone: str = ""
    acarbose: str = ""
    miglitol: str = ""
    troglitazone: str = ""
    tolazamide: str = ""
    insulin: str = ""
    glyburide_metformin: str = ""
    glipizide_metformin: str = ""
    glimepiride_pioglitazone: str = ""
    metformin_rosiglitazone: str = ""
    metformin_pioglitazone: str = ""
    change: str = ""
    diabetesMed: str = ""
    readmitted: str = ""

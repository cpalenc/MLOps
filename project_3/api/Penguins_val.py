# 1. Library imports
from pydantic import BaseModel

# 2. Class for models.
class Penguins(BaseModel):
    age:int
    discharge_disposition_id:int
    time_in_hospital:int
    num_lab_procedures:int
    num_procedures:int
    number_inpatient:int
    diag_1:float
    diag_2:float
    diag_3:float
    number_diagnoses:int

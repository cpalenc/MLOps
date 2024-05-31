# 1. Library imports
from pydantic import BaseModel

# 2. Class for models.
class BienesRaices(BaseModel):
    bed:float = 3.0
    bath:float = 2.0
    acre_lot:float = 0.09
    states:int = 10
    house_size:float = 1409.0

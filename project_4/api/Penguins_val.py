# 1. Library imports
from pydantic import BaseModel

# 2. Class for models.
class Penguins(BaseModel):
    bed:float = 8.0
    bath:float = 2.0
    acre_lot:float = 0.09
    state:int = 10
    house_size:float = 1409.0

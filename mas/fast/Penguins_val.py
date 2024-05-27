# 1. Library imports
from pydantic import BaseModel

# 2. Class for models.
class Penguins(BaseModel):
    bed:float = 3.0
    bath:float = 2.0
    acre_lot:float = 0.09
    street:float = 892999.0
    house_size:float = 1409.0

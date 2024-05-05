# 1. Library imports
from pydantic import BaseModel

# 2. Class for models.
class Penguins(BaseModel):
    elevation:int
    horizontal_distance_to_roadways:int
    hillshade_9am:int
    horizontal_distance_to_fire_points:int

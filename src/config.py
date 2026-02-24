from typing import List

class Config:
    DATA_PATH: str = "./data/covtype.csv"
    OUTPUT_DIR: str = "plots"
    TEST_SIZE: float = 0.25
    RANDOM_STATE: int = 42
    LR_MAX_ITER: int = 5000

    CONTINUOUS_FEATURES: List[str] = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
    ]

    WILDERNESS_FEATURES: List[str] = [f"Wilderness_Area{i}" for i in range(1, 5)]

    SOIL_FEATURES: List[str] = [f"Soil_Type{i}" for i in range(1, 41)]

    TARGET: str = "Cover_Type"
    LABELS: List[str] = [
        "Spruce/Fir",
        "Lodgepole Pine",
        "Ponderosa Pine",
        "Cottonwood/Willow",
        "Aspen",
        "Douglas-fir",
        "Krummholz"
    ]

    # Regression target (count-like, non-negative)
    REGRESSION_TARGET: str = "Horizontal_Distance_To_Hydrology"

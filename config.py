EXCEL_FILE = "Pune_Grand_Summary.xlsx"
PICKLE_FILE = "Pune_Grand_Summary.pkl"

# Sheet configuration for different comparison types
SHEET_CONFIG = {
    "Location": {"sheet": "Location_YOY", "id_col": "final location"},
    "City": {"sheet": "City_YOY", "id_col": "city"},  # Using "city" column for recommendations
    "Project": {"sheet": "Total_project_wise", "id_col": "project name"}  # Using "project name" column for recommendations
}

# Import the mappings from mapping.py
from mapping import (
    CATEGORY_MAPPING_Location,
    CATEGORY_MAPPING_City,
    CATEGORY_MAPPING_Project,
    COLUMN_MAPPING_Location,
    COLUMN_MAPPING_City,
    COLUMN_MAPPING_Project
)

# Function to get category mapping based on comparison type
def get_category_mapping(comparison_type):
    return {
        "Location": CATEGORY_MAPPING_Location,
        "City": CATEGORY_MAPPING_City,
        "Project": CATEGORY_MAPPING_Project
    }[comparison_type]

# Function to get column mapping based on comparison type
def get_column_mapping(comparison_type):
    return {
        "Location": COLUMN_MAPPING_Location,
        "City": COLUMN_MAPPING_City,
        "Project": COLUMN_MAPPING_Project
    }[comparison_type]
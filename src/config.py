from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_COUNTRY_QUESTIONNAIRES_DIR = RAW_DIR / "country_questionnaires"
RAW_METHODOLOGY_DIR = RAW_DIR / "methodology"
RAW_QUESTIONNAIRES_DIR = RAW_DIR / "questionnaires"

OUTPUTS_DIR = BASE_DIR / "outputs"
GENERATION_DIR = OUTPUTS_DIR / "generation"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

UI_RUNTIME_DIR = INTERIM_DIR / "ui_runtime"

COUNTRY_FILTER = os.getenv("COUNTRY_FILTER", "").strip()

SELECTED_COUNTRIES = [
    "France",
    "Estonia",
    "Germany",
    "Austria",
    "Belgium",
    "Bulgaria",
]

SELECTED_QUESTION_IDS = [
    "P1", "P2", "P3", "P4", "P13", "P16", "P22", "P24",
    "I1", "I2", "I9", "I10", "I12", "I17", "I27",
]

COUNTRY_FILE_MAP = {
    "France": RAW_COUNTRY_QUESTIONNAIRES_DIR / "2025_odm_questionnaire_france.xlsx",
    "Estonia": RAW_COUNTRY_QUESTIONNAIRES_DIR / "2025_odm_questionnaire_estonia.xlsx",
    "Germany": RAW_COUNTRY_QUESTIONNAIRES_DIR / "2025_odm_questionnaire_germany.xlsx",
    "Austria": RAW_COUNTRY_QUESTIONNAIRES_DIR / "2025_odm_questionnaire_austria.xlsx",
    "Belgium": RAW_COUNTRY_QUESTIONNAIRES_DIR / "2025_odm_questionnaire_belgium.xlsx",
    "Bulgaria": RAW_COUNTRY_QUESTIONNAIRES_DIR / "2025_odm_questionnaire_bulgaria.xlsx",
}

MASTER_QUESTIONNAIRE_FILE = RAW_QUESTIONNAIRES_DIR / "2025_odm_questionnaire_data.xlsx"
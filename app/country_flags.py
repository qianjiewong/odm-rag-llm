COUNTRY_FLAG_CODES = {
    "Albania": "al",
    "Austria": "at",
    "Belgium": "be",
    "Bosnia and Herzegovina": "ba",
    "Bulgaria": "bg",
    "Canada": "ca",
    "Croatia": "hr",
    "Cyprus": "cy",
    "Czechia": "cz",
    "Denmark": "dk",
    "Estonia": "ee",
    "Finland": "fi",
    "France": "fr",
    "Germany": "de",
    "Greece": "gr",
    "Hungary": "hu",
    "Iceland": "is",
    "Ireland": "ie",
    "Italy": "it",
    "Latvia": "lv",
    "Lithuania": "lt",
    "Luxembourg": "lu",
    "Malta": "mt",
    "Montenegro": "me",
    "Netherlands": "nl",
    "North Macedonia": "mk",
    "Norway": "no",
    "Poland": "pl",
    "Portugal": "pt",
    "Romania": "ro",
    "Serbia": "rs",
    "Slovakia": "sk",
    "Slovenia": "si",
    "Spain": "es",
    "Sweden": "se",
    "Switzerland": "ch",
    "Ukraine": "ua",
}


def get_flag_url(country_name: str) -> str:
    code = COUNTRY_FLAG_CODES.get(str(country_name or "").strip())
    if not code:
        return ""
    return f"https://flagcdn.com/w40/{code}.png"
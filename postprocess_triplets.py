import re

UNITS_PATTERN = re.compile(
    r"(?:" 
    r"(?<=\d)\s*" 
    r"|[,;]\s*"          
    r")"
    r"(?:" 
    r"мкм/год|мм/год"
    r"|м/с|км/ч|мм/мин"
    r"|г/м[23²³]?|кг/м[23²³]?|т/м[23²³]?"
    r"|Н/мм[23²³]?|Дж/см[23²³]?"
    r"|кг|мг|г|т"
    r"|мкм|нм|км|мм|см|дм|м"
    r"|м[23²³]|см[23²³]|мм[23²³]|км[23²³]"
    r"|м³|см³|мм³|дм³|мл|л"
    r"|кПа|МПа|ГПа|Па"
    r"|кВт|МВт|ГВт|Вт"
    r"|кДж|МДж|ГДж|Дж"
    r"|°\s*|℃|К|K"
    r"|мс|мин|ч|сут"
    r"|%|‰"
    r"|×10[-−]?\d+"
    r")"
    r"\s*$",
    re.IGNORECASE
)

def strip_units(text: str) -> str:
    return UNITS_PATTERN.sub("", text).strip()

def remove_measurement_units(string):
    check_known_units = strip_units(string)
    if check_known_units != string:
        return check_known_units
    # print(check_known_units)
    comma_splitted = list(check_known_units.split(','))
    if len(comma_splitted) > 1 and len(comma_splitted[-1]) <= 10:
        return ",".join(comma_splitted[:-1:])
    return check_known_units
    
def remove_me_from_triplets(triplets):
    formatted_triplets = []
    print(len(triplets))
    for triplet in triplets:
        print(triplet)
        subj = triplet.get("subject", None).copy()
        pred = triplet.get("predicate", None).copy()
        obj = triplet.get("object", None).copy()
        subj["text"] = remove_measurement_units(subj["text"])
        obj["text"] = remove_measurement_units(obj["text"])
        formatted_triplets.append({"subject": subj, "predicate": pred, "object": obj})
    return formatted_triplets


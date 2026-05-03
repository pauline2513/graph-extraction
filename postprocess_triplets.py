import re

UNITS_PATTERN = re.compile(
    r",?\s*"
    r"(?:"
    r"к?г|мг|т"                               # масса
    r"|м[мслк]?|км|мкм|нм"                    # длина
    r"|м[23]|см[23]|мм[23]|км[23]"            # площадь/объём
    r"|л|мл|м³|см³|мм³|дм³"                   # объём
    r"|[кМГ]?Па|МПа|ГПа"                      # давление
    r"|[кМГ]?Вт|кВт"                          # мощность
    r"|[кМГ]?Дж"                              # энергия
    r"|°?[СCF]|К"                             # температура
    r"|с|мс|мин|ч|сут"                        # время
    r"|м/с|км/ч|мм/мин"                       # скорость
    r"|г/м[23]?|кг/м[23]?|т/м[23]?"           # плотность
    r"|%|‰"                                   # проценты
    r"|мкм/год|мм/год"                        # коррозия
    r"|[А-Яа-яA-Za-z]+/[А-Яа-яA-Za-z\d]+"   # любое X/Y
    r"|×10[-−]?\d+"                           # ×10-3
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


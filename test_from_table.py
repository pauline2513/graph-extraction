import csv
import re


def read_csv(filepath, sep=";"):
    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f, delimiter=sep, quotechar='"', skipinitialspace=True))


def fix_decimal_commas(row, expected_len):
    while len(row) > expected_len:
        for i in range(1, len(row) - 1):
            if re.fullmatch(r"[+-]?\d+", row[i].strip()) and re.fullmatch(r"\d+", row[i + 1].strip()):
                row[i] = row[i].strip() + "," + row[i + 1].strip()
                del row[i + 1]
                break
        else:
            break
    return row


def check_ordinary_table(filepath, sep=";"):
    """ 
    обычная таблица с 1 заголовком и 1 столбцом названий
    ;скорость обработки;масса
    сталь;14;2 медь;12;1 
    """
    
    rows = read_csv(filepath, sep)
    headers = [x.strip() for x in rows[0][1:]]
    expected_len = len(headers) + 1

    triplets = []

    for row in rows[1:]:
        row = fix_decimal_commas(row, expected_len)

        subj = row[0].strip().replace('"', "")
        values = row[1:]

        for obj, value in zip(headers, values):
            triplets.append((
                subj,
                value.strip().replace('"', ""),
                obj.strip().replace('"', "")
            ))

    print(triplets)
    return triplets


def check_name_value_horizontal_table(filepath, sep=";"):
    rows = read_csv(filepath, sep)
    headers = [x.strip() for x in rows[0]]
    values = [x.strip() for x in rows[1]]
    triplets = []
    for i, subj in enumerate(headers):
        triplets.append((subj.replace('"', ""), values[i], ""))
    return triplets

def check_name_value_vertical_table(filepath, sep=";"):
    rows = read_csv(filepath, sep)
    headers = [x[0].strip() for x in rows]
    values = [x[1].strip() for x in rows]
    triplets = []
    for i, subj in enumerate(headers):
        triplets.append((subj.replace('"', ""), values[i], ""))
    return triplets


# text = """
# Тип стана,Значение kb
# Блюминг,1.15
# НЗС,1.2
# Крупносортный,1.25
# Среднесортный,1.3
# Мелкосортный,"1,35-1,45"

# """
print(check_ordinary_table("tables_for_test\\Т1.6.csv"))
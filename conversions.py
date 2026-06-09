import re

MEETING_YEAR_DICT = {
    1: 1961, 2: 1962, 3: 1964, 4: 1966, 5: 1968,
    6: 1970, 7: 1972, 8: 1975, 9: 1977, 10: 1979,
    11: 1981, 12: 1983, 13: 1985, 14: 1987, 15: 1989,
    16: 1991, 17: 1992, 18: 1994, 19: 1995, 20: 1996,
    21: 1997, 22: 1998, 23: 1999, 24: 2001, 25: 2002,
    26: 2003, 27: 2004, 28: 2005, 29: 2006, 30: 2007,
    31: 2008, 32: 2009, 33: 2010, 34: 2011, 35: 2012,
    36: 2013, 37: 2014, 38: 2015, 39: 2016, 40: 2017,
    41: 2018, 42: 2019, 43: 2021, 44: 2022,
}

def roman_to_int(roman):
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50}

    if not roman or not isinstance(roman, str):
        return None

    result = 0
    for i in range(len(roman)):
        if i > 0 and roman_values[roman[i]] > roman_values[roman[i-1]]:
            result += roman_values[roman[i]] - 2 * roman_values[roman[i-1]]
        else:
            result += roman_values[roman[i]]
    return result

def actm_meeting_to_year(atcm_name: str) -> int | None:
    num_match = re.search(r'ATCM\s+([IVXL]+)', atcm_name)
    if num_match:
        meeting_number = roman_to_int(num_match.group(1))
        if meeting_number is not None:
            return MEETING_YEAR_DICT.get(meeting_number, 2022 + (meeting_number - 44))

    arabic_match = re.search(r'ATCM\s+(\d+)', atcm_name)
    if arabic_match:
        meeting_number = int(arabic_match.group(1))
        return MEETING_YEAR_DICT.get(meeting_number, 2022 + (meeting_number - 44))

    year_match = (
        re.search(r'\((\d{4})\)', atcm_name) or
        re.search(r'ATCM .+?\(.*?,\s*(\d{4})\)', atcm_name) or
        re.search(r'.*?\(.*?,\s*(\d{4})\)', atcm_name)
    )
    if year_match:
        return int(year_match.group(1))

    return None

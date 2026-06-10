country_alternative_names = {
    "Republic of Korea": ["South Korea", "Korea", "S Korea"],
    "Czechia": ["Czech Republic"],
    "Russia": ["Russian Federation"],
    "United States": ["United States of America", "USA", "United States", "US"],
    "New Zealand": ["NZ"],
    "United Kingdom": ["UK"]
}

def get_country_value_from_dict(country_dict, country_name):
    s = 0

    # First we try the country name as-is.
    if country_name in country_dict:
        s += country_dict[country_name]
    
    # Next we try any alternative names.
    for alt_name in country_alternative_names.get(country_name, []):
        if alt_name in country_dict:
            s += country_dict[alt_name]
    
    return s

def check_dict_coverage(country_dict, countries):
    matched_keys = set()
    not_found = []

    for country in countries:
        found = False
        if country in country_dict:
            matched_keys.add(country)
            found = True
        for alt_name in country_alternative_names.get(country, []):
            if alt_name in country_dict:
                matched_keys.add(alt_name)
                found = True
        if not found:
            not_found.append(country)

    unused_keys = [k for k in country_dict if k not in matched_keys]
    return unused_keys, not_found
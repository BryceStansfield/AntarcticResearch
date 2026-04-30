country_alternative_names = {
    "Republic of Korea": ["South Korea"],
    "Czechia": ["Czech Republic"],
    "Russia": ["Russian Federation"],
    "Turkey": ["Türkiye"],
    "United States": ["United States of America", "USA"],
}

def get_country_value_from_dict(country_dict, country_name):
    # First we try the country name as-is.
    if country_name in country_dict:
        return country_dict[country_name]
    
    # Next we try any alternative names.
    for alt_name in country_alternative_names.get(country_name, []):
        if alt_name in country_dict:
            return country_dict[alt_name]
    
    # If all else fails, we return 0.
    return 0
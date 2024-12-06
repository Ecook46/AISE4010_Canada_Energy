import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("forecast_2029_2030.csv") # read in csv
data["emissions"] = 0 # for emissions data
data = data[~data["month"].str.contains("2030")] # taking the year 2029 to evaluate if target met by 2030
data.loc[data["megawatt_hours"] < 0, "megawatt_hours"] = 0 # remove negatives

# dictionary of reference emissions
emissions = {"hydraulic turbine":0.023, "nuclear steam turbine":0.006, "combustible fuels":0.202, "tidal power turbine":0.022,"wind power turbine":0.012,"solar":0.048,"other types of electricity generation":0.130}

# threshold for 2030 objective.
total2005 = 116.7*pow(10,6) # https://www.canada.ca/en/environment-climate-change/services/environmental-indicators/greenhouse-gas-emissions.html#electricity
thresh2030 = total2005*0.55

# isolate provinces
ab = data[data["province"] == "Alberta"]
nb = data[data["province"] == "New Brunswick"]
bc = data[data["province"] == "British Columbia"]
nl = data[data["province"] == "Newfoundland And Labrador"]
ns = data[data["province"] == "Nova Scotia"]
on = data[data["province"] == "Ontario"]
pei = data[data["province"] == "Prince Edward Island"]
qc = data[data["province"] == "Quebec"]
mb = data[data["province"] == "Manitoba"]
sk = data[data["province"] == "Saskatchewan"]
yk = data[data["province"] == "Yukon"]
nwt = data[data["province"] == "Northwest Territories"]
nv = data[data["province"] == "Nunavut"]

# store as list
provinces = [ab, nb, bc, nl, ns, on, pei, qc, mb, sk, yk, nwt, nv]

# Function to calculate emissions
def calculate_emissions(row):
    emission_factor = emissions[row['generation_type']]  
    return row['megawatt_hours'] * emission_factor

# apply emissions function to each province
for province in provinces:
    province['emissions'] = province.apply(calculate_emissions, axis=1)

total = 0 # var to hold total emissions

for province in provinces: # sum total emissions
    total += sum(province["emissions"])

print(sum(ab["emissions"])) # print emission by chosen province (change the province code)
print(total) # print the overall emissions
print(thresh2030) # print the threshold

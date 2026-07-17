import numpy as np

MIN_OVERRIDE = 0
MAX_OVERRIDE = 140
UNIT_OVERRIDE = "%"

MIN_COOLING_POWER = 0
MAX_COOLING_POWER = 30
UNIT_COOLING_POWER = "kW"

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1000.0
UNIT_TEMPERATURE = "°C"

MIN_CUTTING_SPEED = 0
MAX_CUTTING_SPEED = 300
UNIT_CUTTING_SPEED = "m/min"

MIN_COOLANT_FLOW_RATE = 0
MAX_COOLANT_FLOW_RATE = 100
UNIT_COOLANT_FLOW_RATE = "l/min"

MIN_TOOL_WEAR = 0
MAX_TOOL_WEAR = 1.6
UNIT_TOOL_WEAR = "μm/s"

MIN_VIBRATION = 0
MAX_VIBRATION = 52
UNIT_VIBRATION = "μm"

MIN_ENERGY = 0
MAX_ENERGY = 7.5
UNIT_ENERGY = "kW"

MIN_SCRAP_RATIO = 0.0
MAX_SCRAP_RATIO = 1.0
UNIT_SCRAP_RATIO = "%"

# input
def calc_override(percent):
    override_range = MAX_OVERRIDE - MIN_OVERRIDE
    return MIN_OVERRIDE + percent * override_range

def calc_cooling_power(percent):
    cooling_range = MAX_COOLING_POWER - MIN_COOLING_POWER
    return MIN_COOLING_POWER + percent * cooling_range

# helper
def _calc_cutting_speed_from_override(override):
    cutting_speed_range = MAX_CUTTING_SPEED - MIN_CUTTING_SPEED
    override_range = MAX_OVERRIDE - MIN_OVERRIDE
    return MIN_CUTTING_SPEED + (override - MIN_OVERRIDE) * cutting_speed_range / override_range

def _calc_coolant_flow_rate_from_cooling_power(cooling_power):
    coolant_flow_rate_range = MAX_COOLANT_FLOW_RATE - MIN_COOLANT_FLOW_RATE
    cooling_range = MAX_COOLING_POWER - MIN_COOLING_POWER
    return MIN_COOLANT_FLOW_RATE + (cooling_power - MIN_COOLING_POWER) * coolant_flow_rate_range / cooling_range

# output
def calc_temperature(override:float, cooling_power:float):
    cutting_speed = _calc_cutting_speed_from_override(override)

    temperature = 20.0 + 6.0 * cutting_speed**0.898 - 15.0 * cooling_power
    temperature = np.maximum(MIN_TEMPERATURE, temperature)
    temperature = np.minimum(MAX_TEMPERATURE, temperature)
    return temperature

def calc_vibration(override, cooling_power):
    coolant_flow_rate = _calc_coolant_flow_rate_from_cooling_power(cooling_power)
    cutting_speed = _calc_cutting_speed_from_override(override)
    vibration = 10 * (cutting_speed / 100) ** 1.5 * np.exp(-0.02 * coolant_flow_rate)
    vibration = np.maximum(MIN_VIBRATION, vibration)
    vibration = np.minimum(MAX_VIBRATION, vibration)
    return vibration

def calc_tool_wear(override, cooling_power):
    coolant_flow_rate = _calc_coolant_flow_rate_from_cooling_power(cooling_power)
    cutting_speed = _calc_cutting_speed_from_override(override)
    wear_rate = 1e-9 * (cutting_speed ** 2.5) * np.exp(-0.03 * coolant_flow_rate)
    wear_rate = wear_rate*1000.0 # mm -> micrometer
    wear_rate = np.maximum(MIN_TOOL_WEAR, wear_rate)
    wear_rate = np.minimum(MAX_TOOL_WEAR, wear_rate)
    return wear_rate

def calc_energy(override, cooling_power):
    coolant_flow_rate = _calc_coolant_flow_rate_from_cooling_power(cooling_power)
    cutting_speed = _calc_cutting_speed_from_override(override)
    energy = 0.02 * cutting_speed + 0.015 * coolant_flow_rate
    energy = np.maximum(MIN_ENERGY, energy)
    energy = np.minimum(MAX_ENERGY, energy)
    return energy

def calc_override2(override, cooling_power):
    override2 = override
    if isinstance(override2, float) and not isinstance(cooling_power, float) :
        override2 = override * np.ones(len(cooling_power))
    return override2

def calc_scrap_ratio(override, cooling_power):
    coolant_flow_rate = _calc_coolant_flow_rate_from_cooling_power(cooling_power)
    cutting_speed = _calc_cutting_speed_from_override(override)
    scrap_ratio  = np.maximum(0, np.minimum(100, 1 + 0.02 * cutting_speed - 0.1 * coolant_flow_rate))
    scrap_ratio  = np.maximum(MIN_SCRAP_RATIO, scrap_ratio )
    scrap_ratio  = np.minimum(MAX_SCRAP_RATIO, scrap_ratio )
    return scrap_ratio 

# reward
def calc_reward_temperature(override:float, cooling_power:float):
    temperature = calc_temperature(override, cooling_power)
    return 1.0-(temperature/200.0 - 2) ** 2

def calc_reward_vibration(override, cooling_power):
    vibration = calc_vibration(override, cooling_power)
    return 1 - (vibration - MIN_VIBRATION) / (MAX_VIBRATION - MIN_VIBRATION)

def calc_reward_tool_wear(override, cooling_power):
    wear_rate = calc_tool_wear(override, cooling_power)
    return 1 - (wear_rate - MAX_TOOL_WEAR) / (MAX_TOOL_WEAR - MIN_TOOL_WEAR)

def calc_reward_energy(override, cooling_power):
    energy = calc_energy(override, cooling_power)
    return 1 - (energy - MAX_ENERGY) / (MAX_ENERGY - MIN_ENERGY)

def calc_reward_override2(override, cooling_power):
    override2 = calc_override2(override, cooling_power)
    return (override2 - MIN_OVERRIDE) / (MAX_OVERRIDE - MIN_OVERRIDE)

def calc_reward_scrap_ratio(override, cooling_power):
    scrap_ratio = calc_scrap_ratio(override, cooling_power)
    return 1 - (scrap_ratio - MIN_SCRAP_RATIO) / (MAX_SCRAP_RATIO - MIN_SCRAP_RATIO)

# reward triangle
def calc_reward_coast(override, cooling_power):
    reward_energy = calc_reward_energy(override, cooling_power)
    reward_tool_wear = calc_reward_tool_wear(override, cooling_power)
    return reward_energy + reward_tool_wear

def calc_reward_quality(override, cooling_power):
    reward_temperature = calc_reward_temperature(override, cooling_power)
    reward_vibration = calc_reward_vibration(override, cooling_power)
    return reward_temperature + reward_vibration

def calc_reward_time(override, cooling_power):
    reward_override2 = calc_reward_override2(override, cooling_power)
    reward_scrap_ratio = calc_reward_scrap_ratio(override, cooling_power)
    return reward_override2 + reward_scrap_ratio

def calc_reward(weight_coast, weight_time, weight_quality, override, cooling_power):
    reward_quality = calc_reward_quality(override, cooling_power)
    reward_time = calc_reward_time(override, cooling_power)
    reward_coast = calc_reward_coast(override, cooling_power)
    return weight_coast * reward_coast + weight_time * reward_time + weight_quality * reward_quality

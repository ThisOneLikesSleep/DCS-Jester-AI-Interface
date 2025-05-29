favored_words = [
    # Ground Operations
    "Wheel", "Chocks", "Place", "Remove",
    "External", "Power", "Connect", "Disconnect",
    "Air", "Source", "Right", "Left", "Engine", "Start", "Stop", "Flow",
    "Load", "Cartridges", "Boarding", "Ladder", "Steps", "Extend", "Retract",

    # Radio Operations
    "Radio", "Frequency", "Manual", "Megahertz", "Kilohertz", "Decimal", "Dot",
    "Comm", "Aux", "Auxillery",
    "ATC", "Tanker", "Carrier",
    "TR", "ADF", "G", "Plus", "CMD", "Command",

    # Radar Operations
    "Radar", "Active", "Standby", "Auto-focus", "IFF", "Scan", "Elevation", "Identify",
    "Friendly", "Hostile", "Wide", "Narrow", "Center",
    "Focus", "Target", "Lock", "Feet", "Mile", "Nautical",

    # Air to Ground Weapons
    "Weapons", "Pave", "Spike", "Video", "TV",

    # Navigation
    "Go", "Resume", "Next", "Turn", "Point", "Waypoint", "Hold", "Activate",
    "Primary", "Secondary", "Flight", "Plan", "Divert", "Lat", "Long",
    "Edit", "Delete", "Insert", "Map", "Marker", "Airfield", 'Latitude', 'Longitude',
    'North', 'East', 'West', 'East',

    # TACAN
    "TACAN", "Select", "Off", "AAR", "AATR", "Alpha", "Receive",
    "Digit", "Band", "Ground", "Station", "X", "Y",

    # Countermeasures and Systems
    "Chaff", "Flare", "Quantity", "Jettison", "Jammer",
    "Transmit", "Recorder", "AVTR", "Record",

    # Jester Management
    "Jester", "Presence", "Auto", "Force", "Disable",
    "Talking", "Silence", "Ejection", "Selector", "WSO",
    "Countermeasures", "Dispensing",
    "Batumi", "Senaki-Kolkhi", "Kobuleti", "Kutaisi", "Sukhumi-Babushara",
    "Gudauta", "Sochi-Adler", "Mineralnye Vody", "Nalchik", "Mozdok",
    "Beslan", "Anapa-Vityazevo", "Krasnodar-Center", "Krasnodar-Pashkovsky", "Maykop-Khanskaya",
    "Gelendzhik", "Krymsk", "Novorossiysk", "Tbilisi-Lochini", "Vaziani",
    "Soganlug", "Shirak", "Yerevan", "Ganja", "Larnaca",
    "Paphos", "Akrotiri", "Gecitkale", "Incirlik", "Ramat David",
    "Hatzerim", "Ramat Aviv", "Beirut-Rafic Hariri", "Damascus", "Mezzeh",
    "Marj Ruhayyil", "Khalkhalah", "Sayqal", "Tiyas", "Hama",
    "Shayrat", "Bassel Al-Assad", "Abu al-Duhur", "Tabqa", "Deir ez-Zor",
    "Qamishli", "Aleppo", "Rene Mouawad", "King Hussein Air College", "H4",
    "Wadi Al Jandali", "King Hussein Air Base", "King Abdullah II Air Base", "Prince Hassan Air Base", "King Faisal Air Base",
    "Al Safa", "Tabuk", "King Khalid Military City", "Prince Sultan Air Base", "Al Qaysumah/Hafr Al Batin",
    "Al Dhafra", "Liwa", "Al Ain", "Dubai", "Fujairah",
    "Sharjah", "Ras Al Khaimah", "Sohar", "Khasab", "Muscat",
    "Thumrait", "Salalah", "Seeb", "Masirah", "Al Udeid",
    "Doha", "Hamad", "Dukhan", "Isa Air Base", "Sakhir Air Base",
    "Bahrain", "Mina Salman", "Kuwait", "Ali Al Salem", "Ahmed Al Jaber",
    "Jaber Al Ahmed", "Umm Al Maradim", "Al Asad", "Balad", "Erbil",
    "Sulaymaniyah", "Kirkuk", "Taji", "Baghdad", "Tadji",
    "Ali Al Salem", "Qayyarah", "Ayn al-Asad", "Tall Afar", "Al Taqaddum",
    "Al Taji", "Karbala", "Al Rashid", "Ubaydah Bin Al Jarrah", "H3",
    "H3 Northwest", "Al Walid", "Shaibah", "Basrah", "Nasiriyah",
    "Al Kut", "Umm Qasr", "Muthanna", "Qalat Sikar", "Suwairah",
    "Hillah", "H2", "H1", "Az Zubayr", "Al Amarah",
    "Ali Air Base", "Saddam Air Base", "Tikrit", "Ar Ramadi", "Sinjar",
    "Makhmur", "Khanaqin", "Baqubah", "An Nasiriyah", "Bagram",
    "Shindand", "Kandahar", "Jalalabad", "Bagram Airfield", "Farkhor Air Base",
    "Termez", "Kulyab", "Aini", "Kandahar International Airport", "Bagram",
    "Mazar-e-Sharif", "Herat", "Farah", "Chakcharan", "Maimana",
    "Sheberghan", "Kunduz", "Fayzabad", "Bamiyan", "Sharana",
    "Khost", "Gardez", "Ghazni", "Tarinkot", "Lashkar Gah",
    "Zaranj", "Asadabad", "Khost", "Salerno"
]

ground_ops_command = [
    "Chief, place wheel chocks.",
    "Let's get those wheel chocks on.",
    "Take the wheel chocks off.",
    "Chocks in.",
    "Remove the chocks.",
    "Connect external power.",
    "Disconnect external power, chief.",
    "Kill the external power.",
    "Unplug the external power.",
    "Connect air source to right engine.",
    "Air source to left engine, now.",
    "Hook up air to right engine.",
    "Start airflow.",
    "Chief, airflow on.",
    "Cut the airflow.",
    "Disconnect air source.",
    "Air source disconnected.",
    "Chief, air source off.",
    "Unhook the air source.",
    "Load start cartridges.",
    "Start cartridges in.",
    "Take out the start cartridges.",
    "Place the ladder.",
    "Take the ladder away.",
    "Extend the steps."
]

nav_commands = [
    "Go to next turn point.",
    "Resume to waypoint.",
    "Go to waypoint.",
    "Resume navigation to the next turn point.",
    "Hold at current turn point.",
    "Activate hold at current turn point.",
    "Set hold for primary flight plan at turn point.",
    "Hold at turn point in the secondary flight plan.",
    "Deactivate planned hold at primary turn point.",
    "Cancel hold for secondary flight plan at turn point.",
    "Deactivate planned hold at primary flight plan turn point.",
    "Divert to coordinates.",
    "Switch to primary flight plan and navigate to turn point.",
    "Set navigation for map marker.",
    "Navigate to.",
    "Divert to.",
    "Edit primary flight plan at waypoint.",
    "Edit secondary flight plan at waypoint.",
    "Modify primary flight plan waypoint.",
    "Insert new turn point after waypoint in primary flight plan.",
    "Delete turn point at waypoint in primary flight plan.",
    "Edit secondary flight plan turn point at waypoint.",
    "Delete turn point at waypoint in secondary flight plan.",
    "Select TACAN mode.",
    "Set TACAN to channel."
]

radio_commands = [
    "Set UHF radio to manual frequency MHz KHz.",
    "Select UHF radio channel Comm.",
    "Switch to UHF radio auxiliary channel.",
    "Tune UHF radio to ATC at airfield.",
    "Tune UHF radio to.",
    "Select UHF radio mode.",
    "Set manual frequency MHz KHz.",
    "Adjust UHF frequency to MHz KHz.",
    "Set Comm channel on UHF radio.",
    "Select Comm channel on UHF.",
    "Tune to Comm channel.",
    "Switch to auxiliary channel on UHF.",
    "Select auxiliary channel.",
    "Set UHF auxiliary channel.",
    "Switch to auxiliary channel on UHF.",
    "Tune to ATC at airfield.",
    "Tune UHF to.",
    "Set UHF mode to.",
    "Select  mode on UHF.",
    "Set UHF to.",
    "Select guard mode on UHF.",
    "Adjust UHF to ADF mode.",
    "Select ADF mode on UHF.",
    "Set UHF to TR.",
    "Switch UHF to ADF."
]

radar_commands = [
    "Set radar to active mode.",
    "Switch radar to standby mode.",
    "Activate radar auto-focus.",
    "Deactivate radar auto-focus.",
    "Perform IFF check.",
    "Identify contact.",
    "Set radar scan elevation to angels to.",
    "Adjust radar scan elevation to to feet.",
    "Adjust radar scan elevation to angels to ",
    "Center radar scan elevation.",
    "Adjust radar scan elevation to minus feet.",
    "Adjust radar scan elevation to below feet.",
    "Increase radar scan elevation to  feet.",
    "Switch radar scan to nautical miles wide.",
    "Change radar scan to miles narrow.",
    "Set radar scan to miles wide.",
    "Adjust radar scan to nautical miles narrow.",
    "Focus radar on target.",
    "Lock radar on target.",
    "Radar to active mode.",
    "Set radar to standby.",
    "Turn on auto-focus.",
    "Turn off auto-focus.",
    "Perform radar IFF.",
    "Scan feet below us."
]

countermeasure_and_systems = [
    "Set chaff mode to .",
    "Set flare mode to .",
    "Report chaff/flare quantity.",
    "Jettison flares.",
    "Set jammer to .",
    "Turn AVTR recorder .",
    "Set AVTR recorder to .",
    "Activate chaff mode .",
    "Activate flare mode .",
    "Report current chaff flare quantity.",
    "Jettison all flares.",
    "Switch jammer to .",
    "Turn off AVTR recorder.",
    "Switch AVTR recorder to .",
    "Select chaff mode .",
    "Select flare mode .",
    "Set TV Video to Weapons.",
    "Switch TV Video to Pave Spike.",
    "Activate TV Video for Weapons.",
    "Activate TV Video for Pave Spike.",
    "Select TV Video mode Weapons.",
    "Select TV Video mode Pave Spike.",
    "TV Video set to Weapons.",
    "Switch TV mode to Weapons.",
    "Switch TV mode to Pave Spike."
]

# MAYBE FIX THIS SOON.
air_to_ground_commands = [
    "Set TV Video to Weapons.",
    "Switch TV Video to Pave Spike.",
    "Activate TV Video for Weapons.",
    "Activate TV Video for Pave Spike.",
    "Select TV Video mode Weapons.",
    "Select TV Video mode Pave Spike.",
    "TV Video set to Weapons.",
    "TV Video set to Pave Spike.",
    "Switch TV mode to Weapons.",
    "Switch TV mode to Pave Spike."
]

jester_management_commands = [
    "Set Jester presence to auto.",
    "Force Jester presence.",
    "Disable Jester presence.",
    "Jester, talk with me.",
    "Jester, silence please.",
    "Set ejection selector to WSO.",
    "I'll be dispensing the countermeasures.",
    "Jester, you dispense the countermeasures.",
    "Activate Jester presence auto.",
    "Stay with me Jester.",
    "Jester, go away.",
    "Jester, I need you to talk with me.",
    "Jester, please be silent.",
    "Switch ejection selector to WSO.",
    "Switch ejection selector to both.",
    "Change countermeasures dispensing to Jester.",
    "Turn Jester presence to auto.",
    "Turn Jester presence to force.",
    "Turn Jester presence to disable.",
    "Change Jester presence to force.",
    "Enable Jester presence disable.",
    "Jester, start talking with me.",
    "Jester, stop talking please.",
    "Set ejection selector mode to both.",
    "Change countermeasures to manual dispensing."
]

root_class = {
    '0': ground_ops_command, # Ground Operations
    '1': radio_commands, # Radio Operations
    '2': radar_commands, # Radar Operations
    '3': nav_commands, # Navigation
    '4': countermeasure_and_systems, # Countermeasures and Sensors
    '5': jester_management_commands # Jester Management
}


from sentence_transformers import SentenceTransformer
import torch
import os

favored_words = [
    # Ground Operations
    "Wheel", "Chocks", "Place", "Remove",
    "External", "Power", "Connect", "Disconnect",
    "Air", "Source", "Right", "Left", "Engine", "Start", "Stop", "Flow",
    "Load", "Cartridges", "Boarding", "Ladder", "Steps", "Extend", "Retract",

    # Radio Operations
    "Radio", "Frequency", "Manual", "Megahertz", "Kilohertz", "Decimal", "Dot",
    "Comm", "Aux",
    "ATC", "Tanker", "Carrier",
    "TR", "ADF", "G", "Plus", "CMD", "Command",

    # Radar Operations
    "Radar", "Active", "Standby", "Auto-focus", "IFF", "Scan", "Elevation", "Identify",
    "Friendly", "Hostile", "Wide", "Narrow", "Center",
    "Focus", "Target", "Lock", "Feet", "Mile", "Nautical",

    # Air to Ground Weapons
    "Weapons", "Pave", "Spike", "Video", "TV",

    # Navigation
    "Go", "Resume", "Next", "Turn", "Point", "Waypoint", "Hold", "Activate",
    "Primary", "Secondary", "Flight", "Plan", "Divert", "Lat", "Long",
    "Edit", "Delete", "Insert", "Map", "Markers", "Airfields",

    # TACAN
    "TACAN", "Select", "Off", "AAR", "AATR", "Alpha", "Receive",
    "Digit", "Band", "Ground", "Station", "X", "Y",

    # Countermeasures and Systems
    "Chaff", "Flare", "Quantity", "Jettison", "Jammer",
    "Transmit", "Recorder", "AVTR", "Record",

    # Jester Management
    "Jester", "Presence", "Auto", "Force", "Disable",
    "Talking", "Silence", "Ejection", "Selector", "WSO",
    "Countermeasures", "Dispensing"
]

ground_ops_command = [
    "Chief, place wheel chocks.",
    "Let's get those wheel chocks on.",
    "Take the wheel chocks off.",
    "Chocks in.",
    "Remove the chocks.",
    "Connect external power.",
    "Disconnect external power, chief.",
    "Kill the external power.",
    "Unplug the external power.",
    "Connect air source to right engine.",
    "Air source to left engine, now.",
    "Hook up air to right engine.",
    "Start airflow.",
    "Chief, airflow on.",
    "Cut the airflow.",
    "Disconnect air source.",
    "Air source disconnected.",
    "Chief, air source off.",
    "Unhook the air source.",
    "Load start cartridges.",
    "Start cartridges in.",
    "Take out the start cartridges.",
    "Place the ladder.",
    "Take the ladder away.",
    "Extend the steps."
]

nav_commands = [
    "Go to next turn point.",
    "Resume to waypoint.",
    "Go to waypoint.",
    "Resume navigation to the next turn point.",
    "Hold at current turn point.",
    "Activate hold at current turn point.",
    "Set hold for primary flight plan at turn point.",
    "Hold at turn point in the secondary flight plan.",
    "Deactivate planned hold at primary turn point.",
    "Cancel hold for secondary flight plan at turn point.",
    "Deactivate planned hold at primary flight plan turn point.",
    "Divert to coordinates.",
    "Switch to primary flight plan and navigate to turn point.",
    "Set navigation for map marker.",
    "Navigate to.",
    "Divert to.",
    "Edit primary flight plan at waypoint.",
    "Edit secondary flight plan at waypoint.",
    "Modify primary flight plan waypoint.",
    "Insert new turn point after waypoint in primary flight plan.",
    "Delete turn point at waypoint in primary flight plan.",
    "Edit secondary flight plan turn point at waypoint.",
    "Delete turn point at waypoint in secondary flight plan.",
    "Select TACAN mode.",
    "Set TACAN to channel."
]

radio_commands = [
    "Set UHF radio to manual frequency MHz KHz.",
    "Select UHF radio channel Comm.",
    "Switch to UHF radio auxiliary channel.",
    "Tune UHF radio to ATC at airfield.",
    "Tune UHF radio to.",
    "Select UHF radio mode.",
    "Set manual frequency MHz KHz.",
    "Adjust UHF frequency to MHz KHz.",
    "Set Comm channel on UHF radio.",
    "Select Comm channel on UHF.",
    "Tune to Comm channel.",
    "Switch to auxiliary channel on UHF.",
    "Select auxiliary channel.",
    "Set UHF auxiliary channel.",
    "Switch to auxiliary channel on UHF.",
    "Tune to ATC at airfield.",
    "Tune UHF to.",
    "Set UHF mode to.",
    "Select  mode on UHF.",
    "Set UHF to.",
    "Select guard mode on UHF.",
    "Adjust UHF to ADF mode.",
    "Select ADF mode on UHF.",
    "Set UHF to TR.",
    "Switch UHF to ADF."
]

radar_commands = [
    "Set radar to active mode.",
    "Switch radar to standby mode.",
    "Activate radar auto-focus.",
    "Deactivate radar auto-focus.",
    "Perform IFF check.",
    "Identify contact.",
    "Set radar scan elevation to angels to.",
    "Adjust radar scan elevation to to feet.",
    "Adjust radar scan elevation to angels to ",
    "Center radar scan elevation.",
    "Adjust radar scan elevation to minus feet.",
    "Adjust radar scan elevation to below feet.",
    "Increase radar scan elevation to  feet.",
    "Switch radar scan to nautical miles wide.",
    "Change radar scan to miles narrow.",
    "Set radar scan to miles wide.",
    "Adjust radar scan to nautical miles narrow.",
    "Focus radar on target.",
    "Lock radar on target.",
    "Radar to active mode.",
    "Set radar to standby.",
    "Turn on auto-focus.",
    "Turn off auto-focus.",
    "Perform radar IFF.",
    "Scan feet below us."
]

countermeasure_and_systems = [
    "Set chaff mode to .",
    "Set flare mode to .",
    "Report chaff/flare quantity.",
    "Jettison flares.",
    "Set jammer to .",
    "Turn AVTR recorder .",
    "Set AVTR recorder to .",
    "Activate chaff mode .",
    "Activate flare mode .",
    "Report current chaff flare quantity.",
    "Jettison all flares.",
    "Switch jammer to .",
    "Turn off AVTR recorder.",
    "Switch AVTR recorder to .",
    "Select chaff mode .",
    "Select flare mode .",
    "Set TV Video to Weapons.",
    "Switch TV Video to Pave Spike.",
    "Activate TV Video for Weapons.",
    "Activate TV Video for Pave Spike.",
    "Select TV Video mode Weapons.",
    "Select TV Video mode Pave Spike.",
    "TV Video set to Weapons.",
    "Switch TV mode to Weapons.",
    "Switch TV mode to Pave Spike."
]

# MAYBE FIX THIS SOON.
air_to_ground_commands = [
    "Set TV Video to Weapons.",
    "Switch TV Video to Pave Spike.",
    "Activate TV Video for Weapons.",
    "Activate TV Video for Pave Spike.",
    "Select TV Video mode Weapons.",
    "Select TV Video mode Pave Spike.",
    "TV Video set to Weapons.",
    "TV Video set to Pave Spike.",
    "Switch TV mode to Weapons.",
    "Switch TV mode to Pave Spike."
]

jester_management_commands = [
    "Set Jester presence to auto.",
    "Force Jester presence.",
    "Disable Jester presence.",
    "Jester, talk with me.",
    "Jester, silence please.",
    "Set ejection selector to WSO.",
    "I'll be dispensing the countermeasures.",
    "Jester, you dispense the countermeasures.",
    "Activate Jester presence auto.",
    "Stay with me Jester.",
    "Jester, go away.",
    "Jester, I need you to talk with me.",
    "Jester, please be silent.",
    "Switch ejection selector to WSO.",
    "Switch ejection selector to both.",
    "Change countermeasures dispensing to Jester.",
    "Turn Jester presence to auto.",
    "Turn Jester presence to force.",
    "Turn Jester presence to disable.",
    "Change Jester presence to force.",
    "Enable Jester presence disable.",
    "Jester, start talking with me.",
    "Jester, stop talking please.",
    "Set ejection selector mode to both.",
    "Change countermeasures to manual dispensing."
]

root_class = {
    '0': ground_ops_command, # Ground Operations
    '1': radio_commands, # Radio Operations
    '2': radar_commands, # Radar Operations
    '3': nav_commands, # Navigation
    '4': countermeasure_and_systems, # Countermeasures and Sensors
    '5': jester_management_commands # Jester Management
}

'''
embedding_dict = {}

for cat in root_class:
    embeddings = model.encode(root_class[cat], convert_to_tensor=True)
    embedding_dict[i] = embeddings

    i += 1

torch.save(embedding_dict, 'root.pt')
'''

commands = {
    "0": [
        "Chief, place wheel chocks.",
        "Wheel chocks in position.",
        "Chocks in.",
        "Set the chocks."
    ],
    "1": [
        "Chief, remove wheel chocks.",
        "Chocks out.",
        "Remove the chocks.",
        "Wheel chocks off."
    ],
    "2": [
        "Connect external power.",
        "Chief, external power on.",
        "Connect power.",
        "External power, please."
    ],
    "3": [
        "Disconnect external power.",
        "Chief, power off.",
        "Remove external power.",
        "External power disconnect."
    ],
    "4": [
        "Connect air source to the right engine.",
        "Air source to right engine.",
        "Right engine air, connect.",
        "Hook up air to right engine."
    ],
    "5": [
        "Connect air source to the left engine.",
        "Air source to left engine.",
        "Left engine air, connect.",
        "Hook up air to left engine."
    ],
    "6": [
        "Start airflow.",
        "Airflow on.",
        "Initiate airflow.",
        "Airflow, please."
    ],
    "7": [
        "Stop airflow.",
        "Airflow off.",
        "Cease airflow.",
        "Airflow, disconnect."
    ],
    "8": [
        "Disconnect airflow.",
        "Remove air supply.",
        "Remove ground air supply.",
        "Airflow off."
    ],
    "9": [
        "Load start cartridges.",
        "Insert start cartridges.",
        "Start cartridges in.",
        "Prepare start cartridges."
    ],
    "10": [
        "Remove start cartridges.",
        "Unload start cartridges.",
        "Start cartridges out.",
        "Remove cartridges, please."
    ],
    "11": [
        "Place ladder.",
        "Ladder in position.",
        "Set the ladder.",
        "Ladder ready."
    ],
    "12": [
        "Remove ladder.",
        "Ladder off.",
        "Clear the ladder.",
        "Take away the ladder."
    ],
    "13": [
        "Extend steps.",
        "Steps in.",
        "Deploy steps.",
        "Extend the ladder steps."
    ],
    "14": [
        "Retract steps.",
        "Steps out.",
        "Stow steps.",
        "Retract the ladder steps."
    ],
    "15": [
        "Set radio frequency to.",
        "Tune radio to.",
        "Radio frequency.",
        "Set comms to."
    ],
    "16": [
        "Select comm channel.",
        "Tune to comm channel.",
        "Switch to comm channel.",
        "Comm channel, please."
    ],
    "17": [
        "Select auxiliary channel.",
        "Auxiliary channel selected.",
        "Switch to auxiliary channel.",
        "Aux channel, please."
    ],
    "18": [
        "Tune to channel.",
        "Set comms to channel.",
        "Tune radio to channel.",
        "Channel, please."
    ],
    "19": [
        "Select radio mode to.",
        "Radio mode.",
        "Set radio to mode.",
        "Switch radio to mode."
    ],
    "20": [
        "Set radar to.",
        "Radar mode.",
        "Radar to setting.",
        "Switch radar to."
    ],
    "21": [
        "Auto-focus on.",
        "Enable auto-focus.",
        "Auto-focus engage.",
        "Turn on auto-focus."
    ],
    "22": [
        "Auto-focus off.",
        "Disable auto-focus.",
        "Auto-focus disengage.",
        "Turn off auto-focus."
    ],
    "23": [
        "Set scan elevation to.",
        "Scan elevation.",
        "Adjust scan elevation to.",
        "Set radar elevation to."
    ],
    "24": [
        "Set scan to.",
        "Scan setting.",
        "Adjust scan to.",
        "Radar scan."
    ],
    "25": [
        "Focus.",
        "Radar focus.",
        "Set focus to.",
        "Focus on."
    ],
    "26": [
        "Lock target.",
        "Lock on.",
        "Lock on target.",
        "Engage target."
    ],
    "27": [
        "Set TV to weapons.",
        "TV mode weapons.",
        "TV set to weapons.",
        "Switch TV to weapons."
    ],
    "28": [
        "Set TV to Pave Spike.",
        "TV mode Pave Spike.",
        "TV set to Pave Spike.",
        "Switch TV to Pave Spike."
    ],
    "29": [
        "Go to waypoint.",
        "Resume waypoint.",
        "Navigate to waypoint.",
        "Proceed to waypoint."
    ],
    "30": [
        "Hold at turn point.",
        "Hold position at turn point.",
        "Maintain hold at turn point.",
        "Stay at turn point."
    ],
    "31": [
        "Deactivate hold.",
        "Cancel hold.",
        "Release hold.",
        "Undo hold."
    ],
    "32": [
        "Divert to waypoint.",
        "Divert course to waypoint.",
        "Change route to waypoint.",
        "Head to alternate waypoint."
    ],
    "33": [
        "Edit turn point, primary flight plan.",
        "Modify turn point in primary flight plan.",
        "Adjust primary flight plan, turn point.",
        "Edit primary flight plan, turn point."
    ],
    "34": [
        "Insert new turn point after turn point, primary flight plan.",
        "Add turn point after in primary flight plan.",
        "Insert waypoint post in primary plan.",
        "New turn point after, primary plan."
    ],
    "35": [
        "Insert new turn point before turn point, primary flight plan.",
        "Add turn point before in primary flight plan.",
        "Insert waypoint pre in primary plan.",
        "New turn point before, primary plan."
    ],
    "36": [
        "Delete turn point, primary flight plan.",
        "Remove turn point from primary flight plan.",
        "Delete waypoint in primary plan.",
        "Clear turn point, primary plan."
    ],
    "37": [
        "Select TACAN mode.",
        "TACAN mode.",
        "Set TACAN to mode.",
        "Switch TACAN mode to."
    ],
    "38": [
        "Select TACAN channel.",
        "TACAN channel.",
        "Set TACAN to channel.",
        "Switch TACAN channel to."
    ],
    "39": [
        "Tune TACAN to.",
        "TACAN tuned to.",
        "Set TACAN frequency to.",
        "Adjust TACAN to."
    ],
    "40": [
        "Set chaff mode to.",
        "Chaff mode.",
        "Select chaff mode.",
        "Switch chaff to mode."
    ],
    "41": [
        "Set flare mode to.",
        "Flare mode.",
        "Select flare mode.",
        "Switch flare to mode."
    ],
    "42": [
        "Tell me chaff and flare quantity.",
        "Report chaff and flare status.",
        "Check chaff and flare count.",
        "Chaff and flare quantity check."
    ],
    "43": [
        "Jettison flares.",
        "Release flares.",
        "Dump flares.",
        "Get rid of all the flares."
    ],
    "44": [
        "Transmit jammer.",
        "Jammer on.",
        "Activate jammer.",
        "Engage jammer."
    ],
    "45": [
        "Jammer standby.",
        "Set jammer to standby.",
        "Turn the jammer off.",
        "Standby jammer."
    ],
    "46": [
        "Recorder on.",
        "Activate AVTR recorder.",
        "Turn on AVTR.",
        "AVTR recorder start."
    ],
    "47": [
        "AVTR recorder standby.",
        "Recorder off.",
        "Set AVTR to standby.",
        "Standby AVTR recorder."
    ],
    "48": [
        "Start recording.",
        "Begin AVTR recording.",
        "Let's film some action.",
        "Record with AVTR."
    ],
    "49": [
        "Jester presence auto.",
        "Set Jester presence to auto.",
        "Jester auto mode.",
        "Auto Jester presence."
    ],
    "50": [
        "Jester presence force.",
        "Force Jester presence.",
        "Jester presence forced.",
        "Set Jester to force mode."
    ],
    "51": [
        "Jester presence disable.",
        "Disable Jester presence.",
        "Turn off Jester presence.",
        "Jester presence off."
    ],
    "52": [
        "Jester, talk with me.",
        "Communicate, Jester.",
        "Jester, open comms.",
        "Let's talk, Jester."
    ],
    "53": [
        "Jester, silence please.",
        "Quiet, Jester.",
        "Jester, no comms.",
        "Silence, Jester."
    ],
    "54": [
        "Ejection selector WSO.",
        "Set ejection for WSO.",
        "WSO ejection selected.",
        "Ejection mode WSO."
    ],
    "55": [
        "Ejection selector both.",
        "Set ejection for both.",
        "Both ejection selected.",
        "Ejection mode both."
    ],
    "56": [
        "Countermeasures dispensing manual.",
        "Set countermeasures to manual.",
        "Manual countermeasures.",
        "Manual dispensing selected."
    ],
    "57": [
        "Countermeasures dispensing Jester.",
        "Set countermeasures to Jester.",
        "Jester countermeasures.",
        "Jester dispensing selected."
    ],
    "58": [
        "Perform IFF check.",
        "Identify contact.",
        "See if those are friendlies.",
        "Are those enemies?"
    ]
}

commands_index = {
    0: "Place wheel chocks",
    1: "Remove wheel chocks",
    2: "Connect external power",
    3: "Disconnect external power",
    4: "Connect air source to the right engine",
    5: "Connect air source to the left engine",
    6: "Start airflow",
    7: "Stop airflow",
    8: "Disconnect airflow",
    9: "Load start cartridges",
    10: "Remove start cartridges",
    11: "Place ladder",
    12: "Remove ladder",
    13: "Extend steps",
    14: "Retract steps",
    15: "Set radio frequency",
    16: "Select comm channel",
    17: "Select auxiliary channel",
    18: "Tune to #",
    19: "Select radio mode to #",
    20: "Set radar to #",
    21: "Auto-focus on",
    22: "Auto-focus off",
    23: "Set scan elevation to #",
    24: "Set scan to #",
    25: "Focus #",
    26: "Lock #",
    27: "Set TV to weapons",
    28: "Set TV to Pave Spike",
    29: "Go to/resume (turn point or waypoint)",
    30: "Hold (turn point)",
    31: "Deactivate hold",
    32: "Divert to #",
    33: "Edit turn point # (primary or secondary flight plan)",
    34: "Insert new turn point after turn point # (primary or secondary flight plan)",
    35: "Insert new turn point before turn point # (primary or secondary flight plan)",
    36: "Delete turn point # (primary or secondary flight plan)",
    37: "Select TACAN mode #",
    38: "Select TACAN channel #",
    39: "Tune TACAN to #",
    40: "Set chaff mode to #",
    41: "Set flare mode to #",
    42: "Tell me chaff and flare quantity",
    43: "Jettison flares",
    44: "Transmit jammer",
    45: "Jammer standby",
    46: "AVTR recorder on",
    47: "AVTR recorder standby",
    48: "AVTR recorder record",
    49: "Jester presence auto",
    50: "Jester presence force",
    51: "Jester presence disable",
    52: "Jester talk with me",
    53: "Jester silence please",
    54: "Ejection selector WSO",
    55: "Ejection selector both",
    56: "Countermeasures dispensing manual",
    57: "Countermeasures dispensing Jester",
    58: "Perform IFF check"
}
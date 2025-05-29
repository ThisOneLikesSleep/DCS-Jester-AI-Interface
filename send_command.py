import json
import re
import string
from thefuzz import process

command_defs = {
    0: 'select:wheel_chocks_place:',
    1: 'select:wheel_chocks_remove:',
    2: 'select:power_connect:',
    3: 'select:power_disconnect:',
    4: 'select:air_connect_right:',
    5: 'select:air_connect_left:',
    6: 'select:air_start_airflow:',
    7: 'select:air_stop_airflow:',
    8: 'select:air_disconnect:',
    9: 'select:engine_start_cartridges_load:',
    10: 'select:engine_start_cartridges_remove:',
    11: 'select:ladder_place:',
    12: 'select:ladder_remove:',
    13: 'select:steps_extend:',
    14: 'select:steps_retract:',
    # radio freqs
    15: 'select:radio_manual_freq_text:',
    16: 'select:radio_comm_chan:',
    17: 'select:radio_aux_chan:',
    18: 'select:radio_tune_atc:',
    19: 'select:radio_mode',  # off, tf_adf, tfg_adf, adfg_cmd, adf_g, g_adf
    20: 'select:radar_op:',  # active, standby
    21: 'select:radar_auto_focus:on',
    22: 'select:radar_auto_focus:off',
    23: 'select:radar_scan_zone:',  # 30;(5.250, 14,750, 24.750, 34.750, 44.750, 54.750);absolute
                                    # 30;(0, 3.5, 5, 7.5);relative
    24: 'select:radar_display_range:', # nm_25, nm_50;narrow,wide
    25: 'select:', # focus target
    26: 'select:', # lock target
    27: 'select:a2g_tv_feed:weapons',
    28: 'select:a2g_tv_feed:pave_spike',
    29: 'select:',  # go to waypoint
    30: 'select:',  # hold at turn point
    31: 'select:',  # deactivate hold
    32: 'select:divert_tgt1_lat_lon:',
    33: 'select:',  # edit turn point, primary flight plan
    34: 'select:',  # insert new point after turn point, primary flight plan
    35: 'select:',  # insert new point before turn point, primary flight plan
    36: 'select:',  # delete turn point, primary plan
    37: 'select:nav_tacan_mode:',  # off, r, tr, aar, aatr
    38: 'select:',  # nav_tacan_chan_tens (07), nav_tacan_chan_ones (075), nav_tacan_chan_band(075X)
    39: 'select:nav_tacan_tr:',
    40: 'select:systems_chaff:',  # off, single, multiple, program
    41: 'select:systems_flare:',  # off, single, program
    42: 'select:systems_countermeasures_quantity:',
    43: 'select:systems_flares_jettison:',
    44: 'select:systems_jammer:xmit',
    45: 'select:systems_jammer:standby',
    46: 'select:systems_avtr_recorder:off',  # fix this, trained to be on
    47: 'select:systems_avtr_recorder:standby',
    48: 'select:systems_avtr_recorder:record',
    49: 'select:crew_presence:auto',
    50: 'select:crew_presence:enabled',
    51: 'select:crew_presence:disabled',
    52: 'select:crew_talking:talk',
    53: 'select:crew_talking:silence',
    54: 'select:crew_ejection:wso',
    55: 'select:crew_ejection:both',
    56: 'select:crew_countermeasures:manual',
    57: 'select:crew_countermeasures:jester',
    58: 'select:radar_iff:'
}

dict_command_functions = {
    15: 'select radio manual frequency',
    16: 'select comm channel',
    17: 'select auxiliary channel',
    18: 'tune to asset / ATC',
    19: 'select radio mode',
    20: 'select radar mode active/off',
    23: 'select scan elevation',
    24: 'set scan range and azimuth',
    25: 'focus on target',
    26: 'lock on target',
    29: 'go to waypoint',
    30: 'hold at waypoint',
    31: 'deactivate current hold',
    32: 'divert to waypoint',
    33: 'Edit turn point, primary flight plan.',
    34: 'Insert new turn point after turn point, primary flight plan.',
    35: 'Insert new turn point before turn point, primary flight plan.',
    36: 'Delete turn point, primary flight plan.',
    37: 'Select TACAN mode.',
    38: 'Select TACAN channel.',
    39: 'Tune TACAN to.',
    40: 'Set chaff mode',
    41: 'Set flare mode'
}

dict_radio_letter = {
    'automatic': 'a',
    'direction': 'd',
    'finder': 'f',
    'guard': 'g',
    'transmit': 't',
    'receive': 'r',
    'command': 'cmd'
}

# yes off is not an acronym
radio_mode_acronyms = {'tf', 'tfg', 'adf', 'adfg', 'gadf', 'g', 'cmd', 'tfadf', 'tfgadf', 'adfgcmd', 'off'}

radio_modes = {
    'tfadf': 'tf_adf',
    'tfgadf': 'tfg_adf',
    'adfgcmd': 'adfg_cmd',
    'adfg': 'adf_g',
    'gadf': 'g_adf',
    'off': 'off'
}


def extract_last_acronym(keyword_str, acronyms):
    '''
    Given a list of keywords, returns the last occurance of the keyword
    '''

    # Define a pattern to match the specific acronym
    pattern = re.escape(acronyms)

    # Find all matches
    matches = re.findall(pattern, keyword_str)

    # Return the last match if there are any matches
    return matches[-1] if matches else ''

def radio_tune_freq(text, nlp):
    '''
    Manual frequency seelction.
    '''
    doc = nlp(text)
    cardinals = []
    for ent in doc.ents:
        if ent.label == 'CARDINAL':
            cardinals.append(ent.text)

    # return value if list is empty
    if len(cardinals) == 0:
        return None

    pattern = r'^(22[5-9]|2[3-9][0-9]|3[0-8][0-9]|39[0-8])(\.\d{1,3})?$|^399(\.9[0-5]?)?$'

    valid_freqs = []
    for c in cardinals:
        if re.match(pattern, c):
            valid_freqs.append(c)

    # only get the last entry
    freq = valid_freqs[-1]

    # return string of frequency in 6 digit numbers, without the decimal
    return str(int(freq * 1000)).zfill(6)

def radio_select_channel(text, nlp):
    '''
    Extracts valid channel selection for aux and comm channels
    '''
    doc = nlp(text)
    cardinals = []
    for ent in doc.ents:
        if ent.label == 'CARDINAL':
            cardinals.append(ent.text)

    # return value if cardinals is empty
    if len(cardinals) == 0:
        return None

    valid_choices = []
    for c in cardinals:
        if 0 <= c <= 20:
            valid_choices.append(c)

    # return value if no valid choices
    if len(valid_choices) == 0:
        return None

    # only get the last entry
    channel = valid_choices[-1]

    return channel

def radio_tune_asset(websocket_outbound, websocket_inbound, text):
    '''
    Selects a radio channel based on the text and return from the js script.
    '''
    # pattern for getting rid of frequency in item names
    pattern = r'^(.*?)\s*\('

    try:
        websocket_outbound.put('request_radio_menu')
    except Exception as e:
        print(f'Could not send message: {e}')
    try:
        radio_json = websocket_inbound.get(timeout=5)
        radio_data = json.loads(radio_json)['items']
    except Exception as e:
        print(f'Could not accept message: {e}')
        return None

    # Initialize lists to store item names and action values
    item_names = []
    action_values = []

    # iterate through the data
    for category in radio_data:
        for item in category['items']:
            item_name = item['name']
            if item_name == 'No nearby station' or item_name == 'Thinking...':
                continue
            # get rid of frequency in the item name
            match = re.search(pattern, item_name)
            if match:
                filtered_name = match.group(1)
                item_names.append(filtered_name)
                action_values.append(item.get('action_value'))
            else:
                print(f'Error with item {item_name}')

    # return this value if no options are available
    if len(item_names) == 0:
        return None

    pattern = r'\bto\s+(\w+)'
    found_phrases = re.findall(pattern, text)
    # only going to use last entry
    phrase = found_phrases[-1]
    match = process.extractOne(phrase, item_names)[0]

    index = item_names.index(match)
    freq = action_values[index]

    print(match, freq)
    return freq

def radio_mode(text, nlp):
    '''
    Extracts keywords and turn them into acronyms, which are then
    compared to return appropriate string for send_command.
    '''
    mode = ''

    text_lower = text.lower()
    text_no_punc = text_lower.translate(str.maketrans("", "", string.punctuation))
    words_list = re.split(r'\s+', text_no_punc)

    for word in words_list:
        if word in radio_mode_acronyms:
            mode += word
        elif word in dict_radio_letter:
            mode += dict_radio_letter[word]

    if mode == '':
        return None

    if mode in radio_modes:
        return radio_modes[mode]
    else:
        mode = extract_last_acronym(mode, radio_mode_acronyms)
        try:
            return radio_modes[mode]
        except:
            print(f'Error with parsing radio mode for string: {mode}')

def radar_on_off(text, nlp):
    '''
    Determines if sentence wants the radar to be on or off
    '''

    radar_on_ref = nlp('radar on')
    radar_off_ref = nlp('radar off')

    input_ref = nlp(text)
    on_similarity = input_ref.similarity(radar_on_ref)
    off_similarity = input_ref.similarity(radar_off_ref)

    if on_similarity > off_similarity:
        return 'active'
    else:
        return 'standby'

def radar_scan_elev(text, nlp):
    '''
    Classifies radar scan elevation command
    '''

    doc = nlp(text)
    cardinals = []
    for ent in doc.ents:
        if ent.label == 'CARDINAL':
            cardinals.append(ent.text)

    # 1 to 2 digit numbers
    angels = []
    for num in cardinals:
        num = int(num.translate(str.maketrans("", "", string.punctuation)))
        if 0 <= num <= 60:
            angels.append(num)
        elif 1000 <= num <= 60000:
            angels.append(num // 1000)

    # if only one number exists, return fixed elevation based on the number closest to it.
    if len(angels) == 1 and angels[0] <= 10:
        alt_options = [0, 3.5, 5, 7.5]
        closest_option = min(alt_options, key=lambda x: abs(num - x))
        return f'30;{closest_option};relative'

    else:
        last_pair = []
        for i in range(len(angels) - 1):
            for j in range(i + 1, len(angels)):
                if abs(angels[i] - angels[j]) < 11:
                    last_pair = [angels[i], angels[j]]
        if len(last_pair) == 0:
            return None

        last_pair_avg = float(last_pair[0] + last_pair[1]) / 2
        alt_options = [5.250, 14,750, 24.750, 34.750, 44.750, 54.750]
        closest_option = min(alt_options, key=lambda x: abs(last_pair_avg - x))
        return f'30;{closest_option};relative'

def radar_azimuth(text, nlp):
    '''
    Determines scan range and azimuth for the radar. May need some improvement later.
    '''

    doc = nlp(text)
    cardinals = []
    for ent in doc.ents:
        if ent.label == 'CARDINAL':
            cardinals.append(ent.text)

    valid_ranges = []
    for c in cardinals:
        if c == 25 or c == 50:
            valid_ranges.append(c)

    words = re.split(r'\W+', text.lower())
    valid_azimuth = []
    for word in words:
        if word == 'wide' or word == 'narrow':
            valid_azimuth.append(word)
    if len(cardinals) == 0 or len(valid_azimuth) == 0:
        return None

    range = valid_ranges[-1]
    azimuth = valid_azimuth[-1]

    range = lambda x: 'nm_25' if range == 25 else 'nm_50'
    return f'{range};{azimuth}'

def send_command(sock_j_outbound, sock_j_inbound, sock_d_inbound, index, nlp):
    '''
    Sends data through socket as defined by the index
    '''
    print('Send_command called')

    # place holder to reject unimplemented command
    if index in dict_command_functions:
        if 25 <= index <= 26:
            sock_j_outbound.put('request_radar_menu')
            try:
                print(sock_j_inbound.get(timeout=5))
            except Exception as e:
                print('Could not grab message from the server.')
        return

    base_message = command_defs[index]
    print(base_message)

    sock_j_outbound.put(base_message)

    try:
        msg = sock_j_inbound.get(timeout=5)
        print(msg)
    except Exception as e:
        print('Could not grab message from the server.')
    return
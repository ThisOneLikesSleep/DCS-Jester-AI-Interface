import tkinter as tk
import torch
import pygame
import multiprocessing
import sys
import tkinter.messagebox
import sounddevice as sd
import tkinter.ttk as ttk
from recording import *
from classify import *
from udp_server import *
from ws_server import *
from special_key_codes import special_keys_labels
from transformers import AutoTokenizer, AutoModel
from multiprocessing import Process, Queue
from tkinter import *
from pynput import keyboard

# global variables
button = multiprocessing.Value('i', -1)  # -1 indicates no button set
key_device = multiprocessing.Array('c', b' ' * 50)  # Shared array for the key_device string
recording_device = multiprocessing.Value('i', -1)  # currently selected device for recording
local_record_device_ind = -1  # local index for recording device
recording_queue = Queue()  # queue for holding recordings
current_record_process = None
current_stop_event = None
processes = []

def list_recording_devices():
    '''
    Returns a list including the device index and its name
    '''
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    input_devices = []
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            device_name = device['name']
            input_devices.append((idx, hostapis[device['hostapi']]['name'] , device_name))

    return input_devices

def set_keybind():
    global current_record_process, current_stop_event

    print("Press any button on the joystick or keyboard to set the keybind...")

    # wipe the keybind
    button.value = -1
    key_device.value = b' ' * 50

    def on_press(key):
        if hasattr(key, 'vk'):
            button.value = key.vk
        else:
            button.value = key.value.vk
        key_device.value = b'keyboard'
        print(f"Keybind set to keyboard key {key}")
        listener.stop()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    for joystick in joysticks:
        joystick.init()


    while button.value == -1:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                with button.get_lock():
                    button.value = event.button
                with key_device.get_lock():
                    key_device.value = f"joystick_{event.joy}".encode('utf-8')
                print(f"Keybind set to button {event.button} on {key_device.value.decode('utf-8')}")
                listener.stop()
                return
        time.sleep(0.01)

    pygame.quit()

def get_key_name(key_code):
    '''
    Function to get key name from key code
    '''
    if 32 <= key_code <= 126:
        return chr(key_code)
    return special_keys_labels.get(key_code, f"Unknown key ({key_code})")

def update_key():
    '''
    When called, calls set_keybind to update the global variable button
    '''
    global button, local_record_device_ind, recording_queue
    global current_record_process, current_stop_event
    global cuda_bool, root

    # terminate recording process if it's running
    if current_record_process is not None:
        current_stop_event.set()
        current_record_process.join()
        print("Old recording process terminated.")
        current_record_process = None
        current_stop_event = None

    set_keybind()

    if key_device.value == b'keyboard':
        key_label = get_key_name(button.value)
    else:
        key_label = 'button ' + str(button.value)

    recording_device.value = local_record_device_ind

    keybind_label.config(text=f"Keybind: {key_device.value.decode('utf-8')}, {key_label}")
    print(f"Keybind set to: {key_device.value.decode('utf-8')}, {key_label}")  # Debug print

    # start a new recording process
    current_stop_event = multiprocessing.Event()
    print(recording_device.value)
    current_record_process = multiprocessing.Process(target=record,
                                                     args=(button, key_device, recording_device,
                                                           current_stop_event, recording_queue))
    current_record_process.start()
    print("New recording process started.")
def window(button, key_device, recording_device):
    '''
    mainloop here
    '''
    global keybind_label, process_names, local_record_device_ind
    global cuda_bool, api_key_entry, root

    root = Tk()
    root.withdraw()

    cuda = None

    # function that updates device list
    def update_dropdown():
        devices = list_recording_devices()
        combo['values'] = devices
        combo.set(devices[0] if devices else '')

    # sets index number based on the audio device selected
    def on_select(event):
        global local_record_device_ind
        selected_index = combo.current()
        label_value = list_of_devices[selected_index]
        local_record_device_ind = label_value[0]
        print(local_record_device_ind, ' index set')

    def terminate_processes(event, processes):
        print("Terminating all processes...")
        event.set()
        for process, stop_event in processes:
            stop_event.set()
            process.join()
        print("All processes terminated.")
        root.destroy()


    # message shown when program boots up
    use_cuda_message = ('Would you like to use CUDA for this program?'
                        '\nPlease ensure you have enough VRAM for this application.')

    # check CUDA availability and ask user if they want to use it
    if torch.cuda.is_available():
        if (float(torch.version.cuda) - 12.1) < 1e-5:
            root.bell()
            use_cuda = tkinter.messagebox.askyesno('Use CUDA?', use_cuda_message)
            if use_cuda:
                cuda = True
            else:
                cuda = False
    else:
        cuda = False

    cuda_bool = multiprocessing.Value('b', cuda)
    sock_j_outbound = Queue()
    sock_j_inbound = Queue()
    sock_d_inbound = Queue()

    # show main window after use_cuda is set
    if cuda is not None:
        # recall hidden root window
        if cuda is not None:
            classify_process = Process(target=classify,
                                       args=(recording_queue, cuda_bool,
                                             multiprocessing.Event(),
                                             sock_j_outbound, sock_j_inbound, sock_d_inbound))
            tcp_process = Process(target=start_udp, args=(sock_d_inbound,))
            ws_process = Process(target=start_ws, args=(sock_j_outbound, sock_j_inbound))
            classify_process.start()
            tcp_process.start()
            ws_process.start()
            processes.append((classify_process, multiprocessing.Event()))

            root.deiconify()
            root.title("Audio Recording and Classification")
            root.geometry("400x400")
            main_frame = ttk.Frame(root, padding="10 10 10 10")
            main_frame.pack(fill=tk.BOTH, expand=True)
            main_greeting = ttk.Label(main_frame, text='P L A C E H O L D E R', font=("Helvetica", 14))
            main_greeting.pack(pady=10)
            combo = ttk.Combobox(main_frame)
            list_of_devices = list_recording_devices()
            combo['values'] = list_of_devices
            combo['width'] = 40
            combo.state(['readonly'])
            combo.pack(pady=10)
            combo.set(list_of_devices[0] if list_of_devices else '')
            combo.bind("<<ComboboxSelected>>", on_select)
            local_record_device_ind = list_of_devices[0][0]
            update_button = ttk.Button(main_frame, text="Update Devices", command=update_dropdown)
            update_button.pack(pady=10)
            keybind_label = ttk.Label(main_frame, text="Keybind: None", font=("Helvetica", 12))
            keybind_label.pack(pady=10)
            set_keybind_button = ttk.Button(main_frame, text='Set Keybind', command=update_key)
            set_keybind_button.pack(pady=10)
            api_key_label = ttk.Label(main_frame, text="API Key:", font=("Helvetica", 12))
            api_key_label.pack(pady=10)
            api_key_entry = ttk.Entry(main_frame, width=40)
            api_key_entry.pack(pady=10)
            root.protocol("WM_DELETE_WINDOW", lambda: terminate_processes(multiprocessing.Event(), processes))
            root.mainloop()

def main():
    window_process = multiprocessing.Process(target=window,
                                             args=(button, key_device, recording_device))

    window_process.start()
    processes.append((window_process, multiprocessing.Event()))


if __name__ == '__main__':
    main()

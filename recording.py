import pygame
import time
import sounddevice as sd
import numpy as np
import multiprocessing
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
from special_key_codes import special_key_mapping

is_recording = False
audio_data = []

def get_device_id(device_name):
    """Get the device ID for a given device name."""
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        print(f"Device {idx}: {device['name']}")  # Print available devices
        if device_name in device['name']:
            return idx
    raise ValueError(f"Recording device '{device_name}' not found.")

def record(button, key_device, recording_device, stop_event, recording_queue):
    global is_recording, audio_data

    button_value = button.value
    key_device_value = key_device.value.decode()
    device_id = recording_device.value

    print('Record process started')
    print(f"Recording Device ID: {device_id}")

    def on_press(key):
        global is_recording
        if hasattr(key, 'vk'):
            temp = key.vk
        else:
            temp = key.value.vk
        if temp == button_value and not is_recording:
            print('Started recording...')
            is_recording = True

    def on_release(key):
        global is_recording, audio_data
        nonlocal recording_queue
        if hasattr(key, 'vk'):
            temp = key.vk
        else:
            temp = key.value.vk
        if temp == button_value and is_recording:
            print('Stopped recording.')
            is_recording = False
            audio_to_queue(audio_data)
            audio_data = []


    def audio_callback(indata, frames, time, status):
        '''
        Callback function to collect audio data.
        '''

        global audio_data

        if is_recording:
            audio_data.append(indata.copy())

    def audio_to_queue(audio):
        '''
        Sends audio file to queue
        '''

        nonlocal recording_queue

        if audio:
            audio_data_array = np.concatenate(audio_data, axis=0).flatten()
            print(len(audio_data_array))
            if len(audio_data_array) > 8000:
                recording_queue.put(audio_data_array.astype(np.float32))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    if key_device_value == 'keyboard':
        listener.start()
        print('Listener started')
        try:
            with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000, device=device_id):
                print('Audio stream started')
                while not stop_event.is_set():
                    stop_event.wait(0.1)
        finally:
            listener.stop()
            print('Record process terminated.')
    else:
        pygame.init()
        joystick_index = int(key_device_value.split('_')[-1])
        joystick = pygame.joystick.Joystick(joystick_index)
        joystick.init()
        print(f"Initialized Joystick: {joystick.get_name()}")

        try:
            with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000, device=device_id):
                print('Audio stream started')
                while not stop_event.is_set():
                    for event in pygame.event.get():
                        if event.type == pygame.JOYBUTTONDOWN:
                            if event.button == button_value and not is_recording:
                                is_recording = True
                                print('Recording started')

                        if event.type == pygame.JOYBUTTONUP:
                            if event.button == button_value and is_recording:
                                is_recording = False
                                print('Recording stopped')

                                audio_to_queue(audio_data)
                                audio_data = []

        finally:
            try:
                listener.stop()
            except:
                print('Listener not running')
            print('Record process terminated.')
        pygame.quit()
    return

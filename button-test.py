#! /usr/bin/env python

from __future__ import print_function

import RPi.GPIO as GPIO

class TreeHatButtons:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        self._buttonnames=['big', 'medium', 'small']
        self._buttonids=[12, 25, 4]
        for buttonid in self._buttonids:
            GPIO.setup(buttonid, GPIO.IN)

    def get_states(self):
        return map(GPIO.input, self._buttonids)

    def print_states(self, states=None):
        if states is None: states=self.get_states()
        for i in range(len(self._buttonnames)):
            print("%s=%d " % (self._buttonnames[i], states[i]), end='')
        print()

if __name__ == '__main__':

    buttons=TreeHatButtons()
    
    while True:

        states=buttons.get_states()
        buttons.print_states(states)
        if sum(states)==len(states):
            print("Quit...")
            break

#! /usr/bin/env python

from __future__ import print_function

import RPi.GPIO as GPIO
import time

class TreeHatButtons:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        #self._buttonnames=['big', 'medium', 'small']
        #self._buttonids=[12, 25, 17]
        self._buttonnames=['medium', 'small']
        self._buttonids=[25, 17]
        for buttonid in self._buttonids:
            GPIO.setup(buttonid, GPIO.IN)

        self._prevstate=self.get_state_code()
        self._prevt=time.time()

    def get_state_code(self):
        result=0
        for i in range(len(self._buttonnames)):
            result+=GPIO.input(self._buttonids[i])*2**i
        return result

    def get_states(self):
        return self.code_to_states(self.get_state_code())

    def code_to_states(self, code):
        result=[]
        for i in range(len(self._buttonnames)):
            result.append((code & 2**i) > 0)
        return result

    def names_to_code(self, names):
        code=0
        for name in names:
            if not self._buttonnames.count(name):
                raise RuntimeError("No such button '%s'" % name)
            code+=2**self._buttonnames.index(name)
        return code

    def state_string(self, code=None, present_only=False):
        result=[]
        if code is None: code=self.get_state_code()
        for i in range(len(self._buttonnames)):
            if present_only:
                if code & 2**i:
                    result.append(self._buttonnames[i])
            else:
                result.append("%s=%d" % (self._buttonnames[i], (code & 2**i) > 0))
        return ', '.join(result)

    def all_on(self, code=None):
        if code is None: code=self.get_state_code()
        return sum(self.code_to_states(code))==len(self._buttonnames)

    def buttons_on(self, names,
                  code=None,
                  exclusive=True):
        if code is None: code=self.get_state_code()
        if exclusive:
            return code==self.names_to_code(names)
        return code&self.names_to_code(names)

    def check_state(self):
        state=self.get_state_code()
        t=time.time()
        if state!=self._prevstate:
            if state>self._prevstate:
                result={'type':"press", 'code':state, 'prevcode':self._prevstate, 'tsec':t-self._prevt}
            else:
                result={'type':"release", 'code':state, 'prevcode':self._prevstate, 'tsec':t-self._prevt}
            self._prevstate=state
            self._prevt=t
            return result
        else:
            return None

    def wait_for_event(self, sleepsec=0.1):
        while True:
            event=self.check_state()
            if event is not None: return event
            time.sleep(sleepsec)

    def wait_for_state(self, code, sleepsec=0.1):
        while self.get_state_code()!=code:
            time.sleep(sleepsec)

    def wait_for_buttons(self, names,
                         eventtype=None,
                         sleepsec=0.1,
                         exclusive=True,
                         mintsec=None,
                         maxtsec=None):
        code=self.names_to_code(names)
        while True:
            event=self.wait_for_event(sleepsec=sleepsec)
            if eventtype is not None and event['type']!=eventtype: continue
            if mintsec is not None and event['tsec']<mintsec: continue
            if maxtsec is not None and event['tsec']>maxtsec: continue
            if eventtype=='press':
                if exclusive and event['code']!=code: continue
                if not exclusive and not event['code']&code: continue
            if eventtype=='release':
                if exclusive and event['prevcode']!=code: continue
                if not exclusive and not event['prevcode']&code: continue
            return

if __name__ == '__main__':

    buttons=TreeHatButtons()

    while True:
        event=buttons.check_state()
        if event is not None:
            if event['type']=='release' and buttons.all_on(event['prevcode']):
                print("Quitting")
                exit()
            elif event['type']=='release':
                if event['tsec']>1:
                    print('long ', end='')
                print('press '+buttons.state_string(event['prevcode']-event['code'], present_only=True))

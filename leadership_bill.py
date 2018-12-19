# -*- coding:utf-8 -*-
import arg
from util import config, moniter, util
import logging
from copy import deepcopy


class Sign:
    def __init__(self, person=None, time=None, act=None):
        self.person = person
        self.time = time
        self.act = act

    def to_json(self):
        return {
            "person": self.person,
            "time": self.time,
            "act": self.act
        }


class LeaderRecord:
    def __init__(self):
        self.record_type = None
        self.gen = None
        self.user = None
        self.speaker = None
        self.listener = None
        self.meeting_type = None
        self.talking_type = None

    def to_json(self):
        if self.record_type is None:
            return {}
        elif self.record_type == 'm-info' or self.record_type == 'm-plan':
            assert isinstance(self.gen, Sign) and isinstance(self.user, Sign), "{},{},{}".format(self.record_type, self.gen, self.user)
            return {
                'record_type': self.record_type,
                'gen': self.gen.to_json(),
                'user': self.user.to_json()
            }
        elif self.record_type == 'talking':
            assert isinstance(self.speaker, Sign) and isinstance(self.listener, Sign)
            assert isinstance(self.meeting_type, str) and isinstance(self.talking_type, str)
            return {
                'record_type': self.record_type,
                'speaker': self.speaker.to_json(),
                'listener': self.listener.to_json(),
                'meeting': self.meeting_type,
                'talking_type': self.talking_type
            }
        else:
            raise ValueError


class LeaderBill:
    def __init__(self):
        self.records = []

    def add_record(self, **kargs):
        new_record = LeaderRecord()
        if kargs['record_type'] == 'm-info' or kargs['record_type'] == 'm-plan':
            new_record.record_type = kargs['record_type']
            new_record.gen = kargs['gen']
            new_record.user = kargs['user']
        elif kargs['record_type'] == 'talking':
            new_record.record_type = kargs['record_type']
            new_record.speaker = kargs['speaker']
            new_record.listener = kargs['listener']
            new_record.meeting_type = kargs['meeting']
            new_record.talking_type = kargs['talking_type']
        else:
            raise ValueError
        self.records.append(new_record)

    def to_json(self):
        return [rec.to_json() for rec in self.records]


leader_bill = LeaderBill()

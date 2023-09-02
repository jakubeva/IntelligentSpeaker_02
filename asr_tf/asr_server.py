#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2016-2099 Ailemon.net
#
# This file is part of ASRT Speech Recognition Tool.
#
# ASRT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# ASRT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASRT.  If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

from asr_tf.speech_model251bn import ModelSpeech
from asr_tf.language_model import ModelLanguage

datapath = '../asr_tf/'
modelpath = '../asr_tf/model_speech/'
ms = ModelSpeech(datapath)
ms.LoadModel(modelpath + 'm251bn/speech_model251bn_e_0_step_40000（预训练）.base.h5')

ml = ModelLanguage('../asr_tf/model_language')
ml.LoadModel()


def recognize(filename):
    r = ''
    try:
        r_speech = ms.RecognizeSpeech_FromFile(filename)
        print(r_speech)
        str_pinyin = r_speech
        r = ml.PinyinToText(str_pinyin)
        print("已完成拼音向文字的转换")
    except Exception as ex:
        r = ''
        print('[*Message] Server raise a bug. ', ex)
    return r

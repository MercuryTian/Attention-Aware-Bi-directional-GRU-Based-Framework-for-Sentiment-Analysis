# encoding: utf-8
# @author: Zhengxi Tian
# email: zhengxi.tian@hotmail.com

import os
import codecs
import re
import xml.sax

output_file = 'restaurant_train.txt'

class textHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ''
        self.text = ''
        self.aspectTerm = ''
        
    def startElement(self, tag, attributes):
        self.CurrentData = tag
    
    def endElement(self, tag):
        self.CurrentData = ''
        if tag == ''
            
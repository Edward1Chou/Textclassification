#! /bin/env python
# -*- coding: utf-8 -*-
"""
将结果展示成label + text 的形式，易于观察，
并且提供输出到文档的功能
"""
import os
import numpy as np


class LabelText(object):
    def __init__(self, label_list, ori_path):
        self.label_list = label_list
        self.ori_path = ori_path


    def arrangeLabelText(self, show=True, write=False):
        """
        label+text 未排序
        """
        abs_path = os.path.abspath(self.ori_path)
        if write == True:
            write_path = '/'.join(abs_path.split('/')[:-1]) + '/labelText.csv'
            print "new file saved in " + write_path
            w = open(write_path, 'w')
        with open(self.ori_path, 'r') as o:
            for l, s in zip(self.label_list, o.readlines()):
                try:
                    line = str(l) + "\t" + str(s.strip())
                    if show == True:
                        print line
                    if write == True:
                        w.write(line)
                        w.write('\n')
                except:
                    print "--------SOMETHING WRONG!-------"
                    continue
            if write == True:
                w.close()


    def sortByLabel(self, show=True, write=False):
        """
        label+text 排序
        """
        abs_path = os.path.abspath(self.ori_path)
        if write == True:
            write_path = '/'.join(abs_path.split('/')[:-1]) + '/sortedLabelText.csv'
            print "new file saved in " + write_path
            w = open(write_path, 'w')
        with open(self.ori_path, 'r') as o:
            index = np.argsort(self.label_list)
            ori_lines = o.readlines()
            # print "len ori " + str(len(ori_lines))
            for i in range(len(index)):
                try:
                    line = str(self.label_list[index[i]]) + '\t' + str(ori_lines[index[i]].strip())
                    # line = str(self.label_list[index[i]]) + '\t'
                    if show == True:
                        print line
                    if write == True:
                        w.write(line)
                        w.write('\n')
                except:
                    print "--------SOMETHING WRONG!-------"
                    continue
            if write == True:
                w.close()

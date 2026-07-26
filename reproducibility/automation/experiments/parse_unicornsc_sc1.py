"""Parse the Unicorn SC-1 CamFlow archives for the threaTrace experiment.

This is the upstream ``parse_unicornsc.py`` with only the documented SC-1
fragment-name correction: SC-1 archives extract ``camflow-benign.txt.*``,
whereas the upstream SC-2 parser expects ``camflow-normal.txt.*``.
"""

import csv
import os
import os.path as osp
import time

import numpy as np
import pandas as pd


def show(value):
	print(value + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


for i in range(3):
	os.system('tar -zxvf ../graphchi-cpp-master/graph_data/unicornsc/camflow-attack-' + str(i) + '.gz.tar')
for i in range(13):
	os.system('tar -zxvf ../graphchi-cpp-master/graph_data/unicornsc/camflow-benign-' + str(i) + '.gz.tar')

os.system('rm error.log')
os.system('rm parse-error-camflow-*')
show('Start processing.')
for i in range(25):
	show('Attack graph ' + str(i+125))
	f = open('camflow-attack.txt.'+str(i), 'r')
	fw = open('../graphchi-cpp-master/graph_data/unicornsc/'+str(i+125)+'.txt', 'w')
	for line in f:
			tempp = line.strip('\n').split('\t')
			temp = []
			temp.append(tempp[0])
			temp.append(tempp[2].split(':')[0])
			temp.append(tempp[1])
			temp.append(tempp[2].split(':')[1])
			temp.append(tempp[2].split(':')[2])
			temp.append(tempp[2].split(':')[3])
			fw.write(temp[0]+'\t'+temp[1]+'\t'+temp[2]+'\t'+temp[3]+'\t'+temp[4]+'\t'+temp[5]+'\n')
	f.close()
	fw.close()
	os.system('rm camflow-attack.txt.' + str(i))

for i in range(125):
	show('Benign graph ' + str(i))
	f = open('camflow-benign.txt.'+str(i), 'r')
	fw = open('../graphchi-cpp-master/graph_data/unicornsc/'+str(i)+'.txt', 'w')
	for line in f:
			tempp = line.strip('\n').split('\t')
			temp = []
			temp.append(tempp[0])
			temp.append(tempp[2].split(':')[0])
			temp.append(tempp[1])
			temp.append(tempp[2].split(':')[1])
			temp.append(tempp[2].split(':')[2])
			temp.append(tempp[2].split(':')[3])
			fw.write(temp[0]+'\t'+temp[1]+'\t'+temp[2]+'\t'+temp[3]+'\t'+temp[4]+'\t'+temp[5]+'\n')
	f.close()
	fw.close()
	os.system('rm camflow-benign.txt.' + str(i))
show('Done.')

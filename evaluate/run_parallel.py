import os
import sys
import datetime
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

command_file = sys.argv[1]
commands = [line.rstrip() for line in open(command_file)]

report_step = 32
pool = Pool(report_step)
for idx, return_code in enumerate(pool.imap(partial(call, shell=True), commands)):
  if idx % report_step == 0:
     print('[%s] command %d of %d' % (datetime.datetime.now().time(), idx, len(commands)))
  if return_code != 0:
     print('!! command %d of %d (\"%s\") failed' % (idx, len(commands), commands[idx]))

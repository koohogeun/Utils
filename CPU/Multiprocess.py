import gc; gc.enable()
from multiprocessing import Pool

def function(kwargs):
  # do something
  retrun None

data = [] # must list type
p = Pool(16, maxtasksperchild=1)
result = list(tqdm(p.imap(function, data), total=len(data)))
p.close()
p.join()

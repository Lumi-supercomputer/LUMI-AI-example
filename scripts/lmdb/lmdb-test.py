import lmdb
import pickle
import numpy

lmdb_path = 'data-formats/lmdb/val_images'
db = lmdb.open(lmdb_path, subdir=False,
               readonly=True, lock=False,
               readahead=False, meminit=False)

with db.begin(write=False) as txn:
    length = pickle.loads(txn.get(b'__len__'))
    keys = pickle.loads(txn.get(b'__keys__'))
    unpacked = pickle.loads(txn.get(keys[0]))

#print(unpacked)
#print(type(unpacked))  # (image, label)


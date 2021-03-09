import h5py
import subprocess as sp

inf_file = h5py.File("/overflow/dschridelab/10e4_test_infer_FINAL.hdf5")

for key in list(inf_file.keys())[1:]:
    sub_cmd = f"sbatch run_embedding_conversion.sh {key}"
    sp.Popen(sub_cmd.split())

import h5py
import subprocess as sp

inf_file = h5py.File(
    "/overflow/dschridelab/projects/GeneGraphs/embeddings/test1.hdf5", "r"
)

for key in list(inf_file.keys()):
    sub_cmd = f"sbatch run_embedding_conversion.sh {key}"
    sp.Popen(sub_cmd.split())

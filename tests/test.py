import numpy as np
from tqdm import tqdm

new_fp_features = np.load(
    "/data2/shreyas/mining_tomograms/working/tests/first_pass_features.npy"
)
s1_fp_features = np.load(
    "/data2/shreyas/mining_tomograms/s1_clean_results_picket_v2/tomotwin_test/old_features.npy"
)

print(new_fp_features.shape, s1_fp_features.shape)
print(np.allclose(new_fp_features, s1_fp_features))

idxs = []
for i in tqdm(range(len(new_fp_features))):
    if not np.allclose(new_fp_features[i], s1_fp_features[i]):
        idxs.append(i)
print(len(idxs))

import numpy as np
import pandas as pd
import ComputationalHypergraphDiscovery as chd
from load_data import *


Ns_max = 3000
file_name  = f"chd_pair_examples_sample_{Ns_max}"
file_name += "_normalized"


def log_message(message, log_file):
    with open(f"{log_file}.log", "a") as file:
        file.write(message + '\n')

##############################################
# Load data
df_meta = get_meta_data()
scalar_df_list = []
for i, d in zip(df_meta["file #"], df_meta["direction"]):
    file_path = f"data/pair{str(i).zfill(4)}.txt"
    df_i   = parse_file_to_dataframe(file_path)

    if len(df_i.columns) > 2:
        continue
    elif df_i.isna().any().any():
        df_cleaned = df.dropna()
        print(f"NaNs dropped in ", file_path)

    scalar_df_list.append((i, df_i, d))


##############################################
# Compute CHD
def get_edge_weights(df_i, normalize, verbose=False):
    graph_discovery=chd.GraphDiscovery.from_dataframe(df_i, normalize=normalize, verbose=verbose)
    graph_discovery.fit()
    #graph_discovery.plot_graph()

    G = graph_discovery.G
    edge_weights = (G.get_edge_data("Column_1", "Column_2")["signal"].item(),
                    G.get_edge_data("Column_2", "Column_1")["signal"].item())

    return edge_weights


set_id  = np.array([i[0] for i in scalar_df_list])
ew_data = np.zeros((set_id.size,3))
ew_data[:,0] = set_id

for i, (set_id, df_i, true_direction) in enumerate(scalar_df_list):

    # some data sets are much larger and take up to 1/2 hr to finish
    df_i_s = df_i.sample(n=min(df_i.shape[0],Ns_max), random_state=7)

    ew = (-1,-1)
    normalize = True
    try:
        ew = get_edge_weights(df_i_s, normalize)
        chd_dir = 1 if ew[0] > ew[1] else -1
        out = f"test #{str(i).zfill(3)}: chd_d={chd_dir}, true_d={true_direction}: ({ew[0]:1.5f}, {ew[1]:1.5f})"
    except Exception as e:

        try:
            normalize = False
            ew = get_edge_weights(df_i_s, normalize)
            chd_dir = 1 if ew[0] > ew[1] else -1
            out = f"test #{str(i).zfill(3)}: chd_d={chd_dir}, true_d={true_direction}: ({ew[0]:1.5f}, {ew[1]:1.5f})"
        except:
            out = f"test #{str(i).zfill(3)}: FAILED -> " + str(e)

    log_message(out, file_name)
    assert ew_data[i,0] == set_id, "check data_set mapping"
    ew_data[i,1:] = ew

##############################################
# Save results
np.save(f"{file_name}.npy", ew_data)
tmp = np.load(f"{file_name}.npy")
assert np.allclose(tmp, ew_data), "saving failed"


##############################################
# Post-process results
scores = []
for i, (set_id, df_i, true_direction) in enumerate(scalar_df_list):

    set_id0 = ew_data[i,0]
    ew      = ew_data[i,1:]
    chd_dir = 1 if ew[0] > ew[1] else -1

    scores.append(1 if chd_dir == true_direction else 0)

acc = 100*np.sum(scores)/len(scores)
log_message(f"\n\nCHD accuracy: {acc:2.2f}%", file_name)

scores = []
for i, (set_id, df_i, true_direction) in enumerate(scalar_df_list):

    set_id0 = ew_data[i,0]
    ew      = ew_data[i,1:]

    if abs(ew[0] - ew[1]) < 1e-3:
        continue

    chd_dir = 1 if ew[0] > ew[1] else -1

    scores.append(1 if chd_dir == true_direction else 0)

acc = 100*np.sum(scores)/len(scores)
log_message(f"CHD accuracy: {acc:2.2f}%, if only counting edge-weights with diff > 1e-3. Fraction of such weights {len(scores)}/{len(scalar_df_list)}", file_name)



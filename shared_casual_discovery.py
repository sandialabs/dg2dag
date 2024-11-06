import numpy as np
from scipy.linalg import null_space
import torch
import pandas as pd
import os
import torch.nn as nn
from torch.nn import ModuleList
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import copy
import random

## Helper functions
def find_edges(adj_matrix):
    return  np.stack(np.nonzero(adj_matrix), axis=1) # num_edges x 2 matrix; first col is parent, second col is child
def find_parents_children(adj_matrix):
    N = len(adj_matrix) #number of nodes
    parent_list = [np.nonzero(adj_matrix[:,j])[0] for j in range(N)] #list of arrays where array j gives the parents of node j
    children_list = [np.nonzero(adj_matrix[j])[0] for j in range(N)] # list of arrays where array j gives the children of node j
    return parent_list, children_list
def find_undirected_cycles(adj_matrix):
    edge_matrix = find_edges(adj_matrix)
    edges = []
    for j in edge_matrix:
        edges.append(tuple(j))
        edges.append(tuple([j[1], j[0]]))
    G = nx.DiGraph(edges)
    simple_cycles = nx.simple_cycles(G)
    cycles = [j for j in simple_cycles if len(j)>2]
    return cycles
def find_2cycles(edge_matrix):
    # edge_matrix is the list of 2-tuples of edges (output of find_edges)
    num_edges = len(edge_matrix)
    d1_2cycles = np.zeros((1,num_edges)) #return zero matrix if no two cycles found
    two_cycles= []
    two_cycle_edge_idxs = []
    for j in range(num_edges):
        reverse_edge = [edge_matrix[j][1], edge_matrix[j][0]]
        mask = np.all(edge_matrix == reverse_edge, axis=1)
        idx = np.nonzero(mask)[0]
        pot_2cycles = edge_matrix[np.all(edge_matrix == reverse_edge, axis=1)]
        if len(pot_2cycles) > 0:
            if idx > j:
                two_cycles.append(pot_2cycles)
                two_cycle_edge_idxs.append([j, idx[0]])
    if len(two_cycle_edge_idxs) > 0 : # 2-cycles were found
        num_2cycles = len(two_cycle_edge_idxs)
        d1_2cycles = np.zeros((num_2cycles, num_edges))
        for j in range(num_2cycles):
            d1_2cycles[j, two_cycle_edge_idxs[j][0]] = 1
            d1_2cycles[j, two_cycle_edge_idxs[j][1]] = 1
    return two_cycles, two_cycle_edge_idxs, d1_2cycles
def find_solo_2cycles(edge_matrix, parent_list, children_list):
    two_cycles, two_cycle_edge_idxs, d1_2cycles = find_2cycles(edge_matrix)
    solo_two_cycles = []
    for idx in range(len(two_cycles)):
        two_cycle = two_cycles[idx]
        n0 = two_cycle[0][0]
        n1 = two_cycle[0][1]
        s0 = set(list(parent_list[n0]) + list(children_list[n0]))
        s1 = set(list(parent_list[n1]) + list(children_list[n1]))
        common_adj = s0.intersection(s1)
        if len(common_adj) ==  0:
            solo_two_cycles.append(d1_2cycles[idx])
    if len(solo_two_cycles) == 0:
        return np.zeros((1, len(edge_matrix)))
    elif len(solo_two_cycles) == 1:
        return np.expand_dims(solo_two_cycles[0], axis=0)
    else:
        return np.stack(solo_two_cycles, axis=0)
def compute_C(adj_mat):
    cycles = find_undirected_cycles(adj_mat)
    n_cycles = len(cycles)
    edge_matrix = find_edges(adj_mat)
    n_edges = len(edge_matrix)
    dir_cycles = []
    for cycle_idx in range(n_cycles):
        cycle = cycles[cycle_idx]
        n = len(cycle)
        edge_lists = [[] for j in range(n)]
        edge_directions = [[] for j in range(n)]
        for i in range(n):
            forward = np.array([cycle[i],cycle[(i+1) % n]])
            forward_loc = np.where(np.all(edge_matrix == forward, axis=1))[0]
            backward = np.array([cycle[(i+1) % n], cycle[i]])
            backward_loc = np.where(np.all(edge_matrix == backward, axis=1))[0]
            if len(forward_loc) > 0:
                edge_lists[i].append(forward_loc[0])
                edge_directions[i].append(1)
            if len(backward_loc) > 0:
                edge_lists[i].append(-backward_loc[0])
                edge_directions[i].append(-1)
        dir_sub_cycles = [j for j in itertools.product(*edge_lists)]
        cycle_directions = [j for j in itertools.product(*edge_directions)]
        for cycle_idx, cycle in enumerate(dir_sub_cycles):
            n = len(cycle)
            temp_array = np.zeros((1,n_edges))
            for idx in range(n):
                edge_idx = np.abs(cycle[idx])
                edge_sgn = cycle_directions[cycle_idx][idx] #int(cycle[idx] / edge_idx)
                temp_array[0,edge_idx] = edge_sgn
            dir_cycles.append(temp_array)

    parent_list, children_list = find_parents_children(adj_mat)
    #dual_edges = find_solo_2cycles(edge_matrix, parent_list, children_list)
    _,_,dual_edges = find_2cycles(edge_matrix)
    if n_cycles > 0:
        cycle_matrix = np.concatenate(dir_cycles, axis=0)
        C = np.concatenate((cycle_matrix, dual_edges), axis=0)
    else:
        C = dual_edges
    return C
def plot_directed_graph(node_lbls, adj_matrix, title=False, DAG=False, DAG_edges=None):
    nodes = node_lbls
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes)
    edge_matrix = find_edges(adj_matrix)
    num_edges = len(edge_matrix)
    #_, edge_idxs, _ = find_2cycles()
    #edge_idxs = np.array(edge_idxs).flatten()
    edge_labels= {}
    for edge in range(num_edges):
        source= edge_matrix[edge][0]
        target = edge_matrix[edge][1]
        if not DAG:
            graph.add_edge(nodes[source], nodes[target], weight = adj_matrix[source, target])
            edge_labels[(nodes[source], nodes[target])] = '{0:.3f}'.format(adj_matrix[source, target])
        else: #DAG
            graph.add_edge(nodes[source], nodes[target], weight = DAG_edges[edge])
            edge_labels[(nodes[source], nodes[target])] = '{0:.3f}'.format(DAG_edges[edge])
    pos = nx.circular_layout(graph) #nx.circular_layout(graph)
    edges = graph.edges()
    weights = [abs(graph[u][v][0]['weight']) for u, v in edges]
    weights_n = [7*float(i) for i in weights]
    radius = 5
    plt.figure(figsize=[15,9])
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=nodes, node_size=4000)
    nx.draw_networkx_labels(graph, pos=pos, font_size=10, font_color='white', font_weight="bold")
    nx.draw_networkx_edges(graph, pos=pos, width=weights_n, node_size=4000, arrowsize=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=14, bbox=dict(alpha=0), label_pos=0.25, verticalalignment='top', rotate=False, horizontalalignment='left')
    plt.axis('off')
    if title:
        plt.title(title)
    #plt.savefig(os.path.join(fig_save_dir,save_name), bbox_inches = 'tight', transparent=True)
    plt.show()
    plt.close()


def plot_clean_graph(node_lbls, adj_matrix, title=False, DAG=False, DAG_edges=None, ax=None):
    nodes = node_lbls
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes)
    edge_matrix = find_edges(adj_matrix)
    num_edges = len(edge_matrix)
    edge_labels = {}

    for edge in range(num_edges):
        if not DAG:  # plot directed graph / CHD
            source = edge_matrix[edge][0]
            target = edge_matrix[edge][1]
            graph.add_edge(nodes[source], nodes[target], weight=adj_matrix[source, target])
            edge_labels[(nodes[source], nodes[target])] = '{0:.2f}'.format(adj_matrix[source, target])
        else:  # DAG
            if DAG_edges[edge] < 0:  # flip the edge
                source = edge_matrix[edge][1]
                target = edge_matrix[edge][0]
            else:
                source = edge_matrix[edge][0]
                target = edge_matrix[edge][1]
            graph.add_edge(nodes[source], nodes[target], weight=np.abs(DAG_edges[edge]))
            edge_labels[(nodes[source], nodes[target])] = '{0:.2f}'.format(np.abs(DAG_edges[edge]))

    # Positioning of nodes
    pos = nx.circular_layout(graph)  # nx.circular_layout(graph)
    edges = graph.edges()
    weights = [abs(graph[u][v][0]['weight']) for u, v in edges]
    weights_n = [7 * float(i) for i in weights]

    # Create figure only if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=[15, 9])
        show_plot = True
    else:
        show_plot = False

    nx.draw_networkx_nodes(graph, pos=pos, nodelist=nodes, node_size=4000, ax=ax)
    nx.draw_networkx_labels(graph, pos=pos, font_size=10, font_color='white', font_weight="bold", ax=ax)
    nx.draw_networkx_edges(graph, pos=pos, width=weights_n, node_size=4000, ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=14, bbox=dict(alpha=0), label_pos=0.65,
                                 verticalalignment='top', rotate=False, horizontalalignment='left', node_size=4000, ax=ax)
    ax.set_axis_off()

    if title:
        ax.set_title(title)

    if show_plot:
        plt.show()
        plt.close()


def solve_system(adj_matrix):
    #returns edge values and lambdas for the KKT solution to the minimization problem
    edge_values = adj_matrix.flatten()[adj_matrix.flatten() > 0]
    num_edges = len(edge_values)
    edge_matrix = find_edges(adj_matrix)
    #par_list, chil_list = find_parents_children(adj_matrix)
    d1 = compute_C(adj_matrix)
    #print("d1:", d1)
    num_cycles = len(d1)
    KKT_top = np.concatenate((np.eye(num_edges, dtype=adj_matrix.dtype), -d1.transpose()), axis=1)
    KKT_bottom = np.concatenate((-d1, np.zeros((num_cycles, num_cycles))), axis=1)
    KKT = np.concatenate((KKT_top, KKT_bottom), axis=0)
    KKT_inv = np.linalg.pinv(KKT)
    u = np.concatenate((edge_values, np.zeros(num_cycles, dtype=adj_matrix.dtype)))
    DAG_edges = KKT_inv @ u
    return DAG_edges[:num_edges], DAG_edges[num_edges:]


def get_activation_function(activation_function):
    if activation_function == 'ReLU':
        return nn.ReLU()
    elif activation_function == 'Tanh':
        return nn.Tanh()
    elif activation_function == 'LeakyReLU':
        return nn.LeakyReLU()
    elif activation_function == 'Sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}")

class MLP(nn.Module):
    def __init__(self, num_adj,
                 hidden_layer_width=100, hidden_layer_depth=3, activation_function='ReLU'):
        super(MLP, self).__init__()
        layers = []

        # Input layer
        input_dim = num_adj
        for _ in range(hidden_layer_depth):
            layers.append(nn.Linear(input_dim, hidden_layer_width))
            layers.append(get_activation_function(activation_function))
            input_dim = hidden_layer_width

        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, data):
        return self.network(data)


class nodal_MLP(nn.Module):
    def __init__(self, node_idx, num_adj, edge_matrix,
                 hidden_layer_width=100, hidden_layer_depth=3, activation_function='ReLU',
                 debug=False):
        super(nodal_MLP, self).__init__()
        self.node = node_idx
        self.num_edges = len(edge_matrix)
        edge_directions, data_idx, weight_idx = self.edge_dir(node_idx, edge_matrix)

        if debug:
            print("#Nodal MLP################################################")
            print(f"node={node_idx}: num_adj={num_adj}, num_edges={self.num_edges}")
            print("\tedge_directions=", edge_directions)
            print("\tdata_idx=", data_idx)
            print("\tweight_idx=", weight_idx)
            print("##########################################################")

        self.register_buffer('edge_directions', torch.tensor(edge_directions))
        self.register_buffer('data_idx', torch.tensor(data_idx))
        self.register_buffer('weight_idx', torch.tensor(weight_idx))
        self.num_adjs = num_adj
        self.MLP = MLP(self.num_adjs)

        self.MLP = MLP(
            num_adj=self.num_adjs,
            hidden_layer_width=hidden_layer_width,
            hidden_layer_depth=hidden_layer_depth,
            activation_function=activation_function
        )

    def forward(self, data, weights):
        data_by_edge = []
        for idx in range(self.num_adjs):
            weight = weights[self.weight_idx[idx]]
            direction = self.edge_directions[idx]
            data_idx = self.data_idx[idx]
            data_by_edge.append(data[:,data_idx]*F.relu(weight*direction))
        data_by_edge = torch.stack(data_by_edge, dim=-1)
        data_out = self.MLP(data_by_edge)
        return data_out

    def pretrain_forward(self, data, weights):
        data_by_edge = []
        for idx in range(self.num_adjs):
            weight = weights[self.weight_idx[idx]]
            direction = self.edge_directions[idx]
            data_idx = self.data_idx[idx]
            data_by_edge.append(data[:,data_idx]*abs(weight*direction))
        data_by_edge = torch.stack(data_by_edge, dim=-1)
        data_out = self.MLP(data_by_edge)
        return data_out

    def edge_dir(self, node_idx, edge_matrix):
        edge_orientations = []
        data_idx = []
        weight_idx = []
        for edge_idx in range(self.num_edges):
            if edge_matrix[edge_idx,0] == node_idx: #node is a parent
                edge_orientations.append(-1)
                data_idx.append(edge_matrix[edge_idx, 1])
                weight_idx.append(edge_idx)
            elif edge_matrix[edge_idx,1] == node_idx: #node is a child
                edge_orientations.append(1)
                data_idx.append(edge_matrix[edge_idx,0])
                weight_idx.append(edge_idx)
        return edge_orientations, data_idx, weight_idx


class Reconstruct(nn.Module):
    def __init__(self, adj_matrix, use_CHD_init=True, use_noise=False):
        super(Reconstruct, self).__init__()
        edge_matrix = find_edges(adj_matrix)
        parent_list, children_list = find_parents_children(adj_matrix)
        #module info
        self.num_edges = len(edge_matrix)
        self.num_nodes = len(parent_list)
        self.parent_list = parent_list
        self.children_list = children_list
        self.edge_matrix = edge_matrix
        self.num_adj = [len(list(parent_list[j]) + list(children_list[j])) for j in range(self.num_nodes)]

        #weight parameters
        self.use_noise = use_noise
        weight_init, _ = solve_system(adj_matrix) # nearest DAG init
        if use_CHD_init:
            weight_init = adj_matrix.flatten()[np.nonzero(adj_matrix.flatten())[0]] #CHD init
        self.weight_init = torch.tensor(weight_init).unsqueeze(-1)
        self.weights = nn.Parameter(self.weight_init, requires_grad=True)
        C = compute_C(adj_matrix)
        self.nullspace = null_space(C)
        self.d1 = torch.tensor(C)

        #MLPs
        MLPs = ModuleList() #for now, this assumes no isolated nodes
        for node_idx in range(self.num_nodes):
            MLPs.append(nodal_MLP(node_idx, self.num_adj[node_idx], edge_matrix))
        self.MLPs = MLPs
    def forward(self, data):
        has_parents = self.compute_current_has_parents()
        data_out = []
        for node_idx in range(self.num_nodes):
            if has_parents[node_idx]:
                if self.use_noise:
                    data_out.append(self.MLPs[node_idx](data, self.weights + self.compute_noise()))
                else:
                    data_out.append(self.MLPs[node_idx](data, self.weights))
        data_out = torch.cat(data_out, dim=1) #check this is the right dimension
        return data_out
    def pretrain_forward(self, data):
        #has_parents = self.compute_current_has_parents()
        data_out = []
        for node_idx in range(self.num_nodes):
            if self.num_adj[node_idx] > 0: #if there are adjacencies
                data_out.append(self.MLPs[node_idx].pretrain_forward(data, self.weights))
        data_out = torch.cat(data_out, dim=1) #check this is the right dimension
        return data_out
    def compute_current_has_parents(self):
        current_edge_matrix = copy.deepcopy(self.edge_matrix)
        for weight_idx in range(self.num_edges):
            if self.weights[weight_idx] < 0:
                edge = self.edge_matrix[weight_idx]
                flipped = np.array([edge[1], edge[0]])
                current_edge_matrix[weight_idx] = flipped
            elif self.weights[weight_idx] == 0:
                current_edge_matrix[weight_idx] = np.array([-1, -1]) #garbage array bc edge is deleted
        current_children = current_edge_matrix[:,1]
        has_parents = [len(np.where(current_children == j)[0]) > 0 for j in range(self.num_nodes)]
        return has_parents
    def compute_loss(self, data, data_out):
        has_parents = self.compute_current_has_parents()
        num_recon_nodes = data[:,has_parents].shape[-1]
        mse_loss = F.mse_loss(data[:, has_parents], data_out)
        DAG_loss = torch.sum((self.d1 @ self.weights) ** 2)
        return mse_loss, DAG_loss, num_recon_nodes
    def compute_pretrain_loss(self, data, data_out):
        has_parents = np.array(self.num_adj) > 0
        num_recon_nodes = data[:, has_parents].shape[-1]
        mse_loss = F.mse_loss(data[:,has_parents], data_out)
        return mse_loss, num_recon_nodes
    def compute_noise(self, noise_level=None):
        if noise_level is None:
            noise_level = np.std(np.array(self.weight_init))
        noise = torch.normal(0,noise_level,size=(1,self.num_edges))
        proj_noise = torch.sum((noise @ self.nullspace) * self.nullspace, dim=1, keepdim=True)
        return proj_noise


def pretrain(model, epochs, lrate, data_tensor):
    ave_loss = np.zeros(epochs)
    pretrain_optimizer = optim.Adam(model.MLPs.parameters(), lr=lrate)
    for epoch in range(epochs):
        pretrain_optimizer.zero_grad()
        model_out = model.pretrain_forward(data_tensor)
        mse_loss, num_recon_nodes = model.compute_pretrain_loss(data_tensor, model_out)
        loss = mse_loss / num_recon_nodes
        loss.backward()
        pretrain_optimizer.step()
        ave_loss[epoch]=loss.detach()
    return ave_loss


def find_direction(df,
                   M=None,
                   epochs=2000,
                   pretrain_epochs = 100, pre_lr = 1e-2,
                   d1_weight = 1.0, lr = 1e-1,
                   rng_seed=7,
                   plots=False,
                   save_fig=False,
                   use_noise=True,
                   use_lowest_loss=False,
                   ):

    ## step 1: get adjacency matrix (from CHD runs) ###############################
    attr      = list(df.columns)

    data_tensor = torch.tensor(df.to_numpy()).to(dtype=torch.float64)
    #standarize data
    d_mean = torch.mean(data_tensor, dim=0)
    d_std = torch.std(data_tensor, dim=0) + 1e-10
    data_tensor = (data_tensor - d_mean) / d_std

    ## step 2: build model ########################################################
    #Set seeds for reproducibility
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)

    model = Reconstruct(M, use_noise=use_noise).to(dtype=torch.float64)

    ## step 2b: pretrain the neural networks ######################################
    pre_losses = pretrain(model, pretrain_epochs, pre_lr, data_tensor)

    # plt.figure()
    # plt.plot(pre_losses)
    # plt.show()



    ## step 3: train from reconstruction ##########################################
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9)

    save_name     = "low_loss.param"
    curr_low_loss = 1e8

    model.train()
    mse_losses = np.zeros(epochs)
    DAG_losses = np.zeros(epochs)
    total_loss = np.zeros(epochs)
    model_weights_hist = np.zeros((epochs,model.weights.detach().flatten().numpy().size))
    num_nodes = np.zeros(epochs)
    for epoch in range(1,epochs+1):
        optimizer.zero_grad()
        model_out = model(data_tensor)
        mse_loss, DAG_loss, num_recon_nodes = model.compute_loss(data_tensor, model_out)
        num_nodes[epoch-1]=num_recon_nodes
        #loss =  mse_loss
        #loss = mse_loss + d1_weight*DAG_loss #this is the original loss
        #loss = (1/num_recon_nodes) * mse_loss + d1_weight*DAG_loss #this is the average mse original loss
        loss = (1/num_recon_nodes) * mse_loss + d1_weight * DAG_loss
        #loss = (1/num_recon_nodes) * mse_loss + (epoch/epochs * 2)*DAG_loss #dynamic DAG loss weight
        loss.backward()
        optimizer.step()
        mse_losses[epoch-1]=mse_loss.detach()
        DAG_losses[epoch-1]=DAG_loss.detach()
        model_weights_hist[epoch-1,:] = model.weights.detach().flatten().numpy()
        total_loss[epoch-1]=loss

        # saving every epoch takes a long time
        if epoch > 25 and epoch % 5 == 0 and curr_low_loss > loss.item():
            torch.save({'epoch': epoch, 'loss': loss, 'model_state_dict': model.state_dict()}, save_name)
            curr_low_loss = loss.item()

    if use_lowest_loss:
        saved_model_dict = torch.load(save_name)
        assert saved_model_dict['loss'] <= curr_low_loss, "lowest loss ewas not saved correctly"
        saved_model_parameters = saved_model_dict['model_state_dict']
        model.load_state_dict(saved_model_parameters)
        epoch = saved_model_dict['epoch']
    ## step 4: post-process #######################################################
    if plots:
        fig, axs = plt.subplots(1, 3, figsize=[15, 3], constrained_layout=True)
        axs[0].semilogy(mse_losses + d1_weight*DAG_losses, c='b', label="Total loss")
        axs[0].semilogy(mse_losses, c='g', label="Reconstruction loss")
        axs[0].semilogy(DAG_losses*d1_weight, c='r', label="DAG loss")
        axs[0].legend(fontsize='xx-large')
        axs[0].set_title("Loss vs epoch")
        #plot_directed_graph(attr, M, title="DAG learned via reconstruction", DAG=True, DAG_edges=model.weights.detach().flatten().numpy()) # reconstructed DAG

        # print("MSE loss ", mse_losses[-1])
        # print("num recon nodes", num_nodes[-1])
        # print("rng seed ", rng_seed)


        init, _ = solve_system(M)
        #plot_directed_graph(attr, M, title="CHD directed graph", DAG=False) # original directed graph from CHD
        #plot_directed_graph(attr, M, title="DAG initalization", DAG=True, DAG_edges=init)
        #plot_directed_graph(attr, M, title="DAG learned via reconstruction", DAG=True, DAG_edges=model.weights.detach().flatten().numpy()) # reconstructed DAG
        #print("final loss: ", mse_losses[-1])

        plot_clean_graph(attr, M, title="CHD directed graph", DAG=False, ax=axs[1])
        plot_clean_graph(attr, M, title="learned DAG", DAG=True,
                         DAG_edges=model.weights.detach().flatten().numpy(), ax=axs[2])
        fig.tight_layout()
        if save_fig:
            plt.savefig("vdos.pdf")
        #plt.show()
        plt.close()


    return mse_losses, model_weights_hist, epoch


if __name__ == "__main__":

    #############################################
    # Toy problem
    x = np.linspace(-1,1,75)
    y = np.linspace(-1,1,25)**2
    y = np.repeat(y, 3, 0)
    attr   = ['x', 'y']
    df_toy = pd.DataFrame({'x':x, 'y':y})

    M      = np.zeros((2,2))
    M[0,1] = 0.5
    M[1,0] = 0.5

    out = find_direction(df_toy,
                       M=M,
                       epochs=2000,
                       pretrain_epochs = 100, pre_lr = 1e-2,
                       d1_weight = 1.0, lr = 2e-2,
                       rng_seed=7,
                       plots=True,
                       save_fig=None
                      )
    loss, mw_hist, _ = out

    #############################################
    # VDoS Problem
    data_dir = "vdos_data/data"
    attr = ["comp_type", "disorder", "dpa", "max_peak", "peak_pos", "strain", "stress"]
    data = {att: np.load(os.path.join(data_dir, att+".npy")) for att in attr}
    data_df = pd.DataFrame.from_dict(data)


    chd_mat = np.array([[0.,   0.,   0.,   0.,   0.25, 0.,   0.  ],
                      [0.,   0.,   0.06, 0.41, 0.24, 0.,   0.  ],
                      [0.,   0.,   0.,   0.,   0.32, 0.,   0.  ],
                      [0.,   0.46, 0.,   0.,   0.27, 0.22, 0.  ],
                      [0.,   0.,   0.,   0.,   0.,   0.,   0.  ],
                      [0.,   0.,   0.,   0.,   0.24, 0.,   0.24],
                      [0.,   0.,   0.,   0.,   0.33, 0.,   0.  ]])

    out = find_direction(data_df,
                       M=chd_mat,
                       epochs=2000,
                       pretrain_epochs = 100, pre_lr = 1e-2,
                       d1_weight = 1.0, lr = 2e-2,
                       rng_seed=7,
                       plots=True,
                       save_fig=True
                      )
    loss, mw_hist, _ = out

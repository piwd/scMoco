import os
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph
import torch
from sklearn.preprocessing import LabelEncoder

def exists(v):
    return v is not None

def seed_all(seed, cuda_deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic: # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def create_adj_np(data, alpha=100):
    '''
    Constructing the gene co-occurrence matrix with the gene expression matrix.
    params:
        data: The input data of gene expression matrix.
            type: [sc.AnnData, sp.spmatrix, np.ndarray]
        return: co_occurence_matrix, stored as a csr_matrix. Shape: nGene * nGene.
    '''
    if isinstance(data, sp.spmatrix):
        X = data.copy()
        X = X.toarray()
    elif isinstance(data, sc.AnnData):
        if 'counts' in data.layers and data.layers['counts'] is not None:
            if isinstance(data.layers['counts'], sp.spmatrix):
                X = data.layers['counts'].copy()
                X = X.toarray()
            elif isinstance(data.layers['counts'], np.ndarray):
                X = data.layers['counts'].copy()
        else:
            if isinstance(data.X, sp.spmatrix):
                X = data.X.copy()
                X = X.toarray()
            elif isinstance(data.X, np.ndarray):
                X = data.X.copy()
    elif isinstance(data, np.ndarray):
        X = data.copy()
    else:
        raise ValueError("Input data should be AnnData, spmatrix, or np.ndarray")
    # Binarized the expression matrix.
    X[X.nonzero()] = 1
    co_occurrence_matrix = X.T @ X # Constructing the gene co-occurrence matrix by matrix multiplication.
    
    X = X.sum(axis = 0).reshape(-1, 1)
    div_mtx = X @ X.T

    with np.errstate(divide='ignore', invalid='ignore'):
       norm_mtx = np.where(div_mtx != 0, 1 / div_mtx, 0)
    adj = np.multiply(np.power(co_occurrence_matrix, 2) * alpha, norm_mtx)
    
    np.fill_diagonal(adj, 0) # Set the matrix diagonal to zero.
    return sp.csr_matrix(adj)

'''
def create_adj_sp(data):
    """
    Construct the gene co-occurrence matrix with the gene expression matrix.
    To avoid memory overflow, all operations are performed directly on sp.spmatrix objects. Slower, more memory-efficient.
    params:
        data: The input data of gene expression matrix.
            type: (sc.AnnData, sp.spmatrix, np.ndarray)
        return: A co-occurence matrix, stored as a csc_matrix. Shape: nGene * nGene.
    """
    if isinstance(data, sp.spmatrix):
        X = data.copy()
        X = X.tocsr()
    elif isinstance(data, sc.AnnData):
        if 'counts' in data.layers and data.layers['counts'] is not None:
            X = data.layers['counts'].copy()
            if isinstance(X, sp.spmatrix):
                X = X.tocsr()
            elif isinstance(X, np.ndarray):
                X = sp.csr_matrix(X)
        else:
            X = data.X.copy()
            if isinstance(X, sp.spmatrix):
                X = X.tocsr()
            elif isinstance(X, np.ndarray):
                X = sp.csr_matrix(X)
    elif isinstance(data, np.ndarray):
        X = sp.csr_matrix(data.copy())
    else:
        raise("Input data should be AnnData, spmatrix, or np.ndarray")
    # Binarized the expression matrix.
    X[X.nonzero()] = 1
    
    # Constructing the gene co-occurrence matrix by matrix multiplication.
    mtx = X.T * X 
    
    # Set the matrix diagonal to zero.
    n = min(mtx.shape)
    diag_indices = np.arange(n)
    mtx[diag_indices, diag_indices] = 0
    mtx.eliminate_zeros()
    return mtx
    '''

def normalize_adj(adj):
    '''
    Laplacian normalization to adjacency matrix.
    '''
    n = adj.shape[0]
    diags = adj.sum(axis = 1)
    D_sqrt = sp.spdiags(diags.flatten(), [0], n, n, format = 'csr').power(-0.5)
    A_sym = D_sqrt * adj * D_sqrt
    return A_sym

def binning_node_feat(data, n_bins, one_hot=False):
    '''
    Discretize the gene features by binning the gene occurrence number(degrees of node).
    params:
        data: The input data of gene expression matrix.
            type: (AnnData, spmatrix, np.ndarray)
        n_bins: The bins number to assign the gene features to. 
            type: int.
        one_hot: Return the one hot embedding or not. Shape: nGene * n_bins, else: 1 * nGene.
        return: A node feature array.
    '''
    if isinstance(data, sp.spmatrix):
        X = data.copy()
        X = X.tocsr()
    elif isinstance(data, sc.AnnData):
        if 'counts' in data.layers and data.layers['counts'] is not None:
            X = data.layers['counts'].copy()
            if isinstance(X, sp.spmatrix):
                X = X.tocsr()
            elif isinstance(X, np.ndarray):
                X = sp.csr_matrix(X)
        else:
            X = data.X.copy()
            if isinstance(X, sp.spmatrix):
                X = X.tocsr()
            elif isinstance(X, np.ndarray):
                X = sp.csr_matrix(X)
    elif isinstance(data, np.ndarray):
        X = sp.csr_matrix(data.copy())
    else:
        raise ValueError("Input data should be AnnData, spmatrix, or np.ndarray")
    # Binarized the expression matrix.
    X[X.nonzero()] = 1
    X = X.sum(axis = 0)
    X = np.asarray(X)[0]
    
    # Binning.
    log_X = np.log1p(X)
    log_bins = np.linspace(log_X.min(), log_X.max(), n_bins + 1)
    original_bins = np.exp(log_bins) - 1
    binned_X = np.digitize(X, original_bins, right=True) - 1
    binned_X = np.clip(binned_X, 0, n_bins - 1)
    return np.eye(n_bins)[binned_X] if one_hot else binned_X

def scipy_sparse_to_torch_sparse(adj):
    
    """Convert a scipy sparse matrix to a torch sparse tensor."""

    coo = adj.tocoo()

    indices = np.vstack((coo.row, coo.col))
    indices = torch.as_tensor(indices, dtype=torch.long)
    values = torch.as_tensor(coo.data)
    shape = coo.shape
    torch_sparse = torch.sparse_coo_tensor(indices, values, size=shape)

    return torch_sparse.coalesce()

def is_logged(adata):
    return (adata.X.max() < 30) and (adata.X.min() >= 0)

def prepare_adata_for_gene2vec(adata_raw, ref_gene=None, n_features=3000, min_cells=None, min_genes=None):
    '''ref_gene: list of reference gene symbols.'''
    adata = adata_raw.copy()
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_cells is not None:
        sc.pp.filter_genes(adata, min_cells=min_cells)

    sc.pp.normalize_total(adata)
    if not is_logged(adata):
        print('performing log-transform.')
        sc.pp.log1p(adata)

    n = min(n_features, adata.X.shape[1])
    print(f'finding top {n} highly variable genes.')
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=n,
        subset=True,
        inplace=True,
    )

    if ref_gene is None:
        ref_gene = np.load('Data/gene_ids.npy')
    gene_idx = {gene: idx for idx, gene in enumerate(ref_gene)}
    
    adata_genes = adata.var_names.to_list()
    matched_idx = []
    for g in adata_genes:
        if g in gene_idx:
            matched_idx.append(gene_idx[g])
        else:
            matched_idx.append(-1)

    valid_mask = np.array(matched_idx) != -1
    valid_idx = np.array(matched_idx)[valid_mask]
    print(f'{len(valid_idx)}/{len(adata_genes)} genes matched.')

    adata_matched = adata[:, valid_mask].copy()
    adata_matched.layers['counts'] = adata_matched.X.copy()

    return adata_matched, valid_idx

    
def prepare_adata(adata_raw, n_features=3000, min_cells=None, min_genes=None, Modality="scRNA-seq"):
    adata = adata_raw.copy()
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_cells is not None:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata)
    if not is_logged(adata):
        print(f'performing log-transform for {Modality} Data.')
        sc.pp.log1p(adata)
    n = min(n_features, adata.X.shape[1])
    print(f'finding top {n} highly variable features.')
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=n,
        subset=True,
        inplace=True,
    )
    return adata


def calculate_gene_feat(data, n_bins, batch_id=None, device='cpu'):
    if exists(batch_id):
        gene_feat_list = []
        batch_id = np.asarray(batch_id)
        for i in np.unique(batch_id):
            node_feat = binning_node_feat(data[batch_id == i], n_bins)
            node_feat = torch.from_numpy(node_feat).float()
            gene_feat_list.append(node_feat)
        gene_feat = torch.vstack(gene_feat_list).mean(dim=0).long().to(device)
    else:
        gene_feat = binning_node_feat(data, n_bins)
        gene_feat = torch.from_numpy(gene_feat).long().to(device)
    return gene_feat

def get_gene2vec(gene_idx, g2v_array=None, device='cpu'):
    if exists(g2v_array):
        g2v_subset = g2v_array[gene_idx]
    else:
        g2v_array = np.load('data/gene2vec_16906.npy')
        g2v_subset = g2v_array[gene_idx]
    return torch.from_numpy(g2v_subset).float().to(device)

def get_feature_id(adata, device='cpu'):
    LE = LabelEncoder()
    feat = LE.fit_transform(np.asarray(adata.var_names))
    return torch.from_numpy(feat).long().to(device)

def graph_positional_encoding(adj, k = 200, device='cpu'):
    adj = csgraph.laplacian(adj, normed=True, return_diag=False)
    eig_vals, eig_vecs = eigsh(adj, k = k+1, which='SM')
    lap_pe = eig_vecs[:, 1:]
    lap_pe = torch.from_numpy(lap_pe).float().to(device)
    return lap_pe
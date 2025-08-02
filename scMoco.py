import copy
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from accelerate import Accelerator
from model import Model
from utils import *

class Trainer:
    def __init__(self,
                data1,
                data2,
                *,
                n_features_1=3000,
                n_features_2=3000,
                n_bins=None,
                gene2vec=True,
                random_seed=2025,
                device='cpu',
                lr=0.0001,
                epochs=500,
                early_stopping_patience=0,
                early_stopping_delta=0.0,
                **kwargs):
        
        if isinstance(data1, sc.AnnData):
            self.adata1 = data1.copy()
        else:
            raise ValueError("Input data should be AnnData format.")

        if isinstance(data2, sc.AnnData):
            self.adata2 = data2.copy()
        else:
            raise ValueError("Input data should be AnnData format.")

        self.n_features_1 = n_features_1
        self.n_features_2 = n_features_2
        self.n_bins = n_bins
        self.gene2vec = gene2vec
        self.random_seed = random_seed
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.model_kwargs = kwargs
        
        # Initialize training control parameters
        self.best_val_loss = None
        self.early_stop_counter = 0
        self.early_stop = False
        self.current_epoch = 0
        self.best_model_state = None

        print('Preprocessing adata.\n---------------------------------')
        if gene2vec:
            self.adata1, matched_idx = prepare_adata_for_gene2vec(adata_raw=self.adata1, n_features=n_features_1)
        else:
            self.adata1 = prepare_adata(adata_raw=self.adata1, n_features=n_features_1, Modality="Modality_1")
        self.adata2 = prepare_adata(adata_raw=self.adata2, n_features=n_features_2, Modality="Modality_2")
        
        print('Constructing and normalizing graph.\n---------------------------------')
        if exists(kwargs['out_dim']):
            k = kwargs['out_dim']
        else:
            k = 200
        
        self.adj1 = create_adj_np(self.adata1)
        self.lap_pe1 = graph_positional_encoding(adj=self.adj1, k=k, device=device)
        self.adj1 = normalize_adj(self.adj1)
        self.adj1 = scipy_sparse_to_torch_sparse(self.adj1).float().to(self.device)

        self.adj2 = create_adj_np(self.adata2)
        self.lap_pe2 = graph_positional_encoding(adj=self.adj2, k=k, device=device)
        self.adj2 = normalize_adj(self.adj2)
        self.adj2 = scipy_sparse_to_torch_sparse(self.adj2).float().to(self.device)

        print('Generating gene features.\n---------------------------------')
        if gene2vec:
            self.gene_feat1 = self.get_gene2vec(matched_idx)
            if exists(n_bins):
                self.gene_feat2 = binning_node_feat(self.adata2, n_bins)
                self.gene_feat2 = torch.from_numpy(self.gene_feat2).long().to(self.device)
            else:
                self.gene_feat2 = get_feature_id(self.adata2, device=self.device)
        elif exists(n_bins):
            self.gene_feat1 = binning_node_feat(self.adata1, n_bins)
            self.gene_feat1 = torch.from_numpy(self.gene_feat1).long().to(self.device)
            self.gene_feat2 = binning_node_feat(self.adata2, n_bins)
            self.gene_feat2 = torch.from_numpy(self.gene_feat2).long().to(self.device)
        else:
            self.gene_feat1 = get_feature_id(self.adata1, device=self.device)
            self.gene_feat2 = get_feature_id(self.adata2, device=self.device)

        self.X1 = self.adata1.X.toarray() if sp.issparse(self.adata1.X) else self.adata1.X
        self.X2 = self.adata2.X.toarray() if sp.issparse(self.adata2.X) else self.adata2.X
        self.X1 = torch.from_numpy(self.X1).float().to(self.device)
        self.X2 = torch.from_numpy(self.X2).float().to(self.device)

        self.X1_dim = self.adata1.shape[-1]
        self.X2_dim = self.adata2.shape[-1]

        # Initialize model and optimizer placeholder
        self.model = None
        self.optimizer = None

    def get_graph(self):
        return {
            "adj1": self.adj1, 
            "gene_feat1": self.gene_feat2, 
            "adj2": self.adj2, 
            "gene_feat2": self.gene_feat2
            }
    
    def get_gene2vec(self, gene_idx, g2v_array=None):
        if exists(g2v_array):
            g2v_subset = g2v_array[gene_idx]
        else:
            g2v_array = np.load('Data/gene2vec_16906.npy')
            g2v_subset = g2v_array[gene_idx]
        return torch.from_numpy(g2v_subset).float().to(self.device)
    
    def train(self):
        accelerator = Accelerator()
        seed_all(self.random_seed)
        
        # Initialize model and optimizer
        self.model = Model(
            X1_dim= self.X1_dim,
            X2_dim=self.X2_dim,
            n_features_1=self.n_bins if exists(self.n_bins) else self.n_features_1,
            n_features_2=self.n_bins if exists(self.n_bins) else self.n_features_2,
            gene2vec=self.gene2vec,
            **self.model_kwargs
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model, self.optimizer = accelerator.prepare(
            self.model, self.optimizer
        )

        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            if self.early_stop:
                break
            
            self.current_epoch = epoch + 1
            
            # Training phase
            print(f"Epoch {self.current_epoch}\n---------------------------------")
            loss = self.model(
                X1=self.X1, 
                X2=self.X2,
                input_1=self.gene_feat1, 
                adj_1=self.adj1, 
                input_2=self.gene_feat2, 
                adj_2=self.adj2,
                lap_pe1=self.lap_pe1, 
                lap_pe2=self.lap_pe2
            )['loss']

            accelerator.backward(loss)
            self.optimizer.step()
            print(f"Train loss: {loss.item():>7f}")
            
            # Early stopping
            if self.early_stopping_patience > 0:
                if self.best_val_loss is None:
                    self.best_val_loss = loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                elif (self.best_val_loss - loss) > self.early_stopping_delta:
                    self.best_val_loss = loss
                    self.early_stop_counter = 0
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {self.current_epoch}.")
                        self.early_stop = True

            self.model.train()

        # Load best model weights if early stopping was enabled
        if self.early_stopping_patience > 0 and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model weights based on validation loss.")

        print("Model training finished!\n")
        with torch.no_grad():
            self.model.eval()
            results = self.model(
                X1=self.X1, 
                X2=self.X2,
                input_1=self.gene_feat1, 
                adj_1=self.adj1, 
                input_2=self.gene_feat2, 
                adj_2=self.adj2,
                lap_pe1=self.lap_pe1, 
                lap_pe2=self.lap_pe2
            )
            #embeddings = results['embeddings']
            embeddings = F.normalize(results['embeddings'], p=2, eps=1e-12, dim=1)
        output = {
            "embeddings": embeddings.detach().cpu().numpy(),
            "attn_weights": results['attn_weights'].detach().cpu().numpy()
        }
        return output


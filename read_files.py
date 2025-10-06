import numpy as np
import torch
from collections import defaultdict
import os


def read_content_file(content_path):
    """
    Read the .content file and extract paper features and labels.
    
    Args:
        content_path (str): Path to the .content file
        
    Returns:
        tuple: (paper_features, paper_labels, paper_id_to_idx, vocab_size)
            - paper_features: numpy array of shape (N, V) where N is number of papers, V is vocab size
            - paper_labels: list of class labels for each paper
            - paper_id_to_idx: dictionary mapping paper IDs to indices
            - vocab_size: size of vocabulary (V)
    """
    paper_features = []
    paper_labels = []
    paper_id_to_idx = {}
    
    with open(content_path, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            paper_id = parts[0]
            features = [int(x) for x in parts[1:-1]]  # All except first (ID) and last (label)
            label = parts[-1]
            
            paper_id_to_idx[paper_id] = idx
            paper_features.append(features)
            paper_labels.append(label)
    
    paper_features = np.array(paper_features, dtype=np.float32)
    vocab_size = paper_features.shape[1]
    
    return paper_features, paper_labels, paper_id_to_idx, vocab_size


def read_cites_file(cites_path, paper_id_to_idx):
    """
    Read the .cites file and build adjacency matrix.
    
    Args:
        cites_path (str): Path to the .cites file
        paper_id_to_idx (dict): Dictionary mapping paper IDs to indices
        
    Returns:
        numpy.ndarray: Adjacency matrix of shape (N, N) where N is number of papers
    """
    N = len(paper_id_to_idx)
    adjacency_matrix = np.zeros((N, N), dtype=np.float32)
    
    with open(cites_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                cited_paper = parts[0]
                citing_paper = parts[1]
                
                # Check if both papers exist in our dataset
                if cited_paper in paper_id_to_idx and citing_paper in paper_id_to_idx:
                    cited_idx = paper_id_to_idx[cited_paper]
                    citing_idx = paper_id_to_idx[citing_paper]
                    
                    # Create directed edge: citing_paper -> cited_paper
                    # According to README: "paper1 paper2" means "paper2->paper1"
                    adjacency_matrix[citing_idx, cited_idx] = 1.0
    
    return adjacency_matrix


def load_cora_dataset(data_dir="data/cora"):
    """
    Load the Cora dataset and return tensors in the required format.
    
    Args:
        data_dir (str): Directory containing the Cora dataset files
        
    Returns:
        tuple: (features_tensor, adjacency_matrix)
            - features_tensor: torch.Tensor of shape (1, N, V) - article representation tensor
            - adjacency_matrix: torch.Tensor of shape (N, N) - adjacency matrix representing citation relationships
            where N is the number of papers and V is the vocabulary size
    """
    content_path = os.path.join(data_dir, "cora.content")
    cites_path = os.path.join(data_dir, "cora.cites")
    
    # Check if files exist
    if not os.path.exists(content_path):
        raise FileNotFoundError(f"Content file not found: {content_path}")
    if not os.path.exists(cites_path):
        raise FileNotFoundError(f"Cites file not found: {cites_path}")
    
    # Read content file
    print("Reading content file...")
    paper_features, paper_labels, paper_id_to_idx, vocab_size = read_content_file(content_path)
    N = len(paper_features)
    print(f"Loaded {N} papers with vocabulary size {vocab_size}")
    
    # Read cites file
    print("Reading cites file...")
    adjacency_matrix = read_cites_file(cites_path, paper_id_to_idx)
    print(f"Built adjacency matrix with {np.sum(adjacency_matrix)} edges")
    
    # Convert to PyTorch tensors
    # Reshape features to (1, N, V) as required
    features_tensor = torch.from_numpy(paper_features).unsqueeze(0)  # Shape: (1, N, V)
    adjacency_tensor = torch.from_numpy(adjacency_matrix)  # Shape: (N, N)
    
    features_tensor = features_tensor.cuda().float()
    adjacency_tensor = adjacency_tensor.cuda().int()
    
    print(f"Features tensor shape: {features_tensor.shape}")
    print(f"Adjacency matrix shape: {adjacency_tensor.shape}")
    
    return features_tensor, adjacency_tensor


def load_citeseer_dataset(data_dir="data/citeseer"):
    """
    Load the CiteSeer dataset and return tensors in the required format.
    
    Args:
        data_dir (str): Directory containing the CiteSeer dataset files
        
    Returns:
        tuple: (features_tensor, adjacency_matrix)
            - features_tensor: torch.Tensor of shape (1, N, V) - article representation tensor
            - adjacency_matrix: torch.Tensor of shape (N, N) - adjacency matrix representing citation relationships
    """
    content_path = os.path.join(data_dir, "citeseer.content")
    cites_path = os.path.join(data_dir, "citeseer.cites")
    
    # Check if files exist
    if not os.path.exists(content_path):
        raise FileNotFoundError(f"Content file not found: {content_path}")
    if not os.path.exists(cites_path):
        raise FileNotFoundError(f"Cites file not found: {cites_path}")
    
    # Read content file
    print("Reading content file...")
    paper_features, paper_labels, paper_id_to_idx, vocab_size = read_content_file(content_path)
    N = len(paper_features)
    print(f"Loaded {N} papers with vocabulary size {vocab_size}")
    
    # Read cites file
    print("Reading cites file...")
    adjacency_matrix = read_cites_file(cites_path, paper_id_to_idx)
    print(f"Built adjacency matrix with {np.sum(adjacency_matrix)} edges")
    
    # Convert to PyTorch tensors
    features_tensor = torch.from_numpy(paper_features).unsqueeze(0)  # Shape: (1, N, V)
    adjacency_tensor = torch.from_numpy(adjacency_matrix)  # Shape: (N, N)
    
    features_tensor = features_tensor.cuda().float()
    adjacency_tensor = adjacency_tensor.cuda().int()
    
    print(f"Features tensor shape: {features_tensor.shape}")
    print(f"Adjacency matrix shape: {adjacency_tensor.shape}")
    
    return features_tensor, adjacency_tensor


def load_pubmed_dataset(data_dir="data/Pubmed-Diabetes"):
    """
    Load the PubMed dataset and return tensors in the required format.
    
    Args:
        data_dir (str): Directory containing the PubMed dataset files
        
    Returns:
        tuple: (features_tensor, adjacency_matrix)
            - features_tensor: torch.Tensor of shape (1, N, V) - article representation tensor
            - adjacency_matrix: torch.Tensor of shape (N, N) - adjacency matrix representing citation relationships
    """
    # PubMed has a different file structure
    cites_path = os.path.join(data_dir, "data", "Pubmed-Diabetes.DIRECTED.cites.tab")
    content_path = os.path.join(data_dir, "data", "Pubmed-Diabetes.NODE.paper.tab")
    
    # Check if files exist
    if not os.path.exists(content_path):
        raise FileNotFoundError(f"Content file not found: {content_path}")
    if not os.path.exists(cites_path):
        raise FileNotFoundError(f"Cites file not found: {cites_path}")
    
    # Read PubMed content file (different format)
    print("Reading PubMed content file...")
    paper_features, paper_labels, paper_id_to_idx, vocab_size = read_pubmed_content_file(content_path)
    N = len(paper_features)
    print(f"Loaded {N} papers with vocabulary size {vocab_size}")
    
    # Read PubMed cites file (different format)
    print("Reading PubMed cites file...")
    adjacency_matrix = read_pubmed_cites_file(cites_path, paper_id_to_idx)
    print(f"Built adjacency matrix with {np.sum(adjacency_matrix)} edges")
    
    # Convert to PyTorch tensors
    features_tensor = torch.from_numpy(paper_features).unsqueeze(0)  # Shape: (1, N, V)
    adjacency_tensor = torch.from_numpy(adjacency_matrix)  # Shape: (N, N)
    
    features_tensor = features_tensor.cuda().float()
    adjacency_tensor = adjacency_tensor.cuda().int()
    
    print(f"Features tensor shape: {features_tensor.shape}")
    print(f"Adjacency matrix shape: {adjacency_tensor.shape}")
    
    return features_tensor, adjacency_tensor


def read_pubmed_content_file(content_path):
    """
    Read the PubMed .NODE.paper.tab file and extract paper features.
    
    Args:
        content_path (str): Path to the .NODE.paper.tab file
        
    Returns:
        tuple: (paper_features, paper_labels, paper_id_to_idx, vocab_size)
    """
    paper_features = []
    paper_labels = []
    paper_id_to_idx = {}
    
    with open(content_path, 'r') as f:
        # Skip header line
        next(f)
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            paper_id = parts[0]
            features = [float(x) for x in parts[1:-1]]  # All except first (ID) and last (label)
            label = parts[-1]
            
            paper_id_to_idx[paper_id] = idx
            paper_features.append(features)
            paper_labels.append(label)
    
    paper_features = np.array(paper_features, dtype=np.float32)
    vocab_size = paper_features.shape[1]
    
    return paper_features, paper_labels, paper_id_to_idx, vocab_size


def read_pubmed_cites_file(cites_path, paper_id_to_idx):
    """
    Read the PubMed .DIRECTED.cites.tab file and build adjacency matrix.
    
    Args:
        cites_path (str): Path to the .DIRECTED.cites.tab file
        paper_id_to_idx (dict): Dictionary mapping paper IDs to indices
        
    Returns:
        numpy.ndarray: Adjacency matrix of shape (N, N)
    """
    N = len(paper_id_to_idx)
    adjacency_matrix = np.zeros((N, N), dtype=np.float32)
    
    with open(cites_path, 'r') as f:
        # Skip header line
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                cited_paper = parts[1]  # Second column is cited paper
                citing_paper = parts[0]  # First column is citing paper
                
                # Check if both papers exist in our dataset
                if cited_paper in paper_id_to_idx and citing_paper in paper_id_to_idx:
                    cited_idx = paper_id_to_idx[cited_paper]
                    citing_idx = paper_id_to_idx[citing_paper]
                    
                    # Create directed edge: citing_paper -> cited_paper
                    adjacency_matrix[citing_idx, cited_idx] = 1.0
    
    return adjacency_matrix


if __name__ == "__main__":
    # Example usage
    try:
        print("Loading Cora dataset...")
        features, adj = load_cora_dataset()
        print(f"Successfully loaded Cora dataset!")
        print(f"Features shape: {features.shape}")
        print(f"Adjacency matrix shape: {adj.shape}")
        print(f"Number of edges: {adj.sum().item()}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")

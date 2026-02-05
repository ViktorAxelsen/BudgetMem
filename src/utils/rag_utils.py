import os
from typing import List
import threading

import torch
import torch.nn.functional as F
from tqdm import tqdm


# Global lock for protecting SentenceTransformer in multi-threaded environments
_embedding_lock = threading.Lock()


try:
    import faiss
except ImportError:
    print("FAISS INITIALIZATION FAILED")
    faiss = None


def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    token_embeddings: [B, L, D]
    mask: [B, L]
    """
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None].clamp(min=1)

    return sentence_embeddings


def init_context_model(retriever: str):
    """Initialize context model, prefer local cache"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if retriever == 'dpr':
        from transformers import (
            DPRContextEncoder,
            DPRContextEncoderTokenizer,
        )
        model_name = "facebook/dpr-ctx_encoder-single-nq-base"
        try:
            context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name, local_files_only=True)
            context_model = DPRContextEncoder.from_pretrained(model_name, local_files_only=True).to(device)
            print(f"[init_context_model] ✓ Using locally cached model: {model_name}")
        except Exception as e:
            print(f"[init_context_model] Local model not found, starting download: {model_name}")
            context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
            context_model = DPRContextEncoder.from_pretrained(model_name).to(device)
        context_model.eval()
        return context_tokenizer, context_model
    elif retriever == 'contriever':
        from transformers import AutoTokenizer, AutoModel
        model_name = 'facebook/contriever'
        print(f"[init_context_model] Loading model: {model_name}")
        context_tokenizer = AutoTokenizer.from_pretrained(model_name)
        context_model = AutoModel.from_pretrained(model_name).to(device)
        print(f"[init_context_model] ✓ Model loaded: {model_name}")
        context_model.eval()
        return context_tokenizer, context_model
    elif retriever == 'dragon':
        from transformers import AutoTokenizer, AutoModel
        model_name = 'facebook/dragon-plus-context-encoder'
        try:
            context_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            context_model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
            print(f"[init_context_model] ✓ Using locally cached model: {model_name}")
        except Exception as e:
            print(f"[init_context_model] Local model not found, starting download: {model_name}")
            context_tokenizer = AutoTokenizer.from_pretrained(model_name)
            context_model = AutoModel.from_pretrained(model_name).to(device)
        context_model.eval()
        return context_tokenizer, context_model
    else:
        raise ValueError(f"Unknown retriever type: {retriever}")


def init_query_model(retriever: str):
    """Initialize query model, prefer local cache"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if retriever == 'dpr':
        from transformers import (
            DPRQuestionEncoder,
            DPRQuestionEncoderTokenizer,
        )
        model_name = "facebook/dpr-question_encoder-single-nq-base"
        try:
            question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name, local_files_only=True)
            question_model = DPRQuestionEncoder.from_pretrained(model_name, local_files_only=True).to(device)
            print(f"[init_query_model] ✓ Using locally cached model: {model_name}")
        except Exception as e:
            print(f"[init_query_model] Local model not found, starting download: {model_name}")
            question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
            question_model = DPRQuestionEncoder.from_pretrained(model_name).to(device)
        question_model.eval()
        return question_tokenizer, question_model
    elif retriever == 'contriever':
        from transformers import AutoTokenizer, AutoModel
        model_name = 'facebook/contriever'
        print(f"[init_query_model] Loading model: {model_name}")
        question_tokenizer = AutoTokenizer.from_pretrained(model_name)
        question_model = AutoModel.from_pretrained(model_name).to(device)
        print(f"[init_query_model] ✓ Model loaded: {model_name}")
        question_model.eval()
        return question_tokenizer, question_model
    elif retriever == 'dragon':
        from transformers import AutoTokenizer, AutoModel
        model_name = 'facebook/dragon-plus-query-encoder'
        try:
            question_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            question_model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
            print(f"[init_query_model] ✓ Using locally cached model: {model_name}")
        except Exception as e:
            print(f"[init_query_model] Local model not found, starting download: {model_name}")
            question_tokenizer = AutoTokenizer.from_pretrained(model_name)
            question_model = AutoModel.from_pretrained(model_name).to(device)
        question_model.eval()
        return question_tokenizer, question_model
    else:
        raise ValueError(f"Unknown retriever type: {retriever}")


class LongformerWrapper:
    """
    Wrapper for Longformer model to make it compatible with SentenceTransformer interface
    """
    def __init__(self, model_name: str, device: str, max_length: int = 4096):
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.max_length = max_length

        # Get embedding dimension from model config
        self._embedding_dim = self.model.config.hidden_size

    def get_sentence_embedding_dimension(self):
        """Return embedding dimension (compatible with SentenceTransformer)"""
        return self._embedding_dim

    def encode(self, sentences: List[str], batch_size: int = 32,
               show_progress_bar: bool = False, convert_to_numpy: bool = True,
               normalize_embeddings: bool = True):
        """
        Encode sentences to embeddings (compatible with SentenceTransformer)

        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array
            normalize_embeddings: Whether to normalize embeddings (L2)

        Returns:
            Embeddings as numpy array or torch tensor
        """
        import numpy as np
        from tqdm import tqdm

        all_embeddings = []

        # Process in batches
        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding")

        with torch.no_grad():
            for i in iterator:
                batch_sentences = sentences[i:i+batch_size]

                # Tokenize with Longformer's extended context
                inputs = self.tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)

                # Get model outputs
                outputs = self.model(**inputs)

                # Use mean pooling over token embeddings
                # Mask padding tokens
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state

                # Mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

                # Normalize if requested
                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())

        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)

        if convert_to_numpy:
            return all_embeddings.numpy()
        return all_embeddings


def init_data_embedding_model(model_name: str = 'all-MiniLM-L6-v2'):
    """
    Initialize embedding model using Sentence Transformer or Longformer
    Prefer locally cached models to avoid network timeout issues

    Args:
        model_name: Model name
                   Sentence Transformer models:
                   - 'all-MiniLM-L6-v2': Fast, 384-dim (default)
                   - 'all-mpnet-base-v2': High quality, 768-dim
                   - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual support
                   Longformer models:
                   - 'allenai/longformer-base-4096': Long text support, 768-dim, max 4096 tokens

    Returns:
        (tokenizer, encoder): Model tokenizer and encoder
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init_data_embedding_model] Loading {model_name} on {device}")

    # Check if it's a Longformer model
    if 'longformer' in model_name.lower():
        print(f"[init_data_embedding_model] Using Longformer with max_length=4096")
        try:
            model = LongformerWrapper(model_name, device, max_length=4096)
            # Return model as both tokenizer and encoder for consistency
            return model, model
        except ImportError as e:
            raise ImportError(
                f"transformers library not installed. "
                f"Install via: pip install transformers\n{e}"
            )
    else:
        # Use Sentence Transformer
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install via: pip install sentence-transformers"
            )

        try:
            model = SentenceTransformer(model_name, device=device, local_files_only=True)
            print(f"[init_data_embedding_model] ✓ Using locally cached model: {model_name}")
        except Exception as e:
            print(f"[init_data_embedding_model] Local model not found, starting download: {model_name}")
            print(f"[init_data_embedding_model] Tip: If download is slow, you can run a pre-download script first")
            os.environ['HF_HUB_TIMEOUT'] = '1800'
            os.environ['HF_HUB_NUM_RETRIES'] = '20'
            model = SentenceTransformer(model_name, device=device)
        
        model.eval()

        return model, model

def get_data_embeddings(
    encoder,
    inputs: List[str],
    batch_size: int = 32,
    show_progress: bool = False
):
    """
    Compute embeddings using Sentence Transformer model

    Args:
        encoder: Sentence Transformer model instance
        inputs: Input text list
        batch_size: Batch size
        show_progress: Whether to show progress bar

    Returns:
        numpy.ndarray: Embeddings array with shape [N, D]
    """
    if len(inputs) == 0:
        return torch.empty(0, encoder.get_sentence_embedding_dimension()).numpy()

    with _embedding_lock:
        embeddings = encoder.encode(
            inputs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    return embeddings




def get_embeddings_with_model(retriever: str,
                              inputs: List[str],
                              tokenizer,
                              encoder,
                              batch_size: int = 512) -> torch.Tensor:
    """Compute embeddings using initialized model and tokenizer (avoid reloading model)"""
    all_embeddings = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="GET EMBEDDINGS", disable=True):
            batch_texts = inputs[i:(i + batch_size)]

            if retriever == 'dpr':
                # DPR: Use pooler_output and L2 normalize
                enc_inputs = tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                outputs = encoder(**enc_inputs)
                embeddings = outputs.pooler_output.detach()
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                all_embeddings.append(embeddings)
            elif retriever == 'contriever':
                # mean pooling + L2 normalize
                enc_inputs = tokenizer(
                    batch_texts, padding=True, truncation=True, return_tensors='pt'
                ).to(device)
                outputs = encoder(**enc_inputs)
                embeddings = mean_pooling(outputs[0], enc_inputs['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                all_embeddings.append(embeddings)
            elif retriever == 'dragon':
                # dragon: Use CLS token representation; optionally normalize
                enc_inputs = tokenizer(
                    batch_texts, padding=True, truncation=True, return_tensors='pt'
                ).to(device)
                outputs = encoder(**enc_inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings)
            else:
                raise ValueError(f"Unknown retriever type: {retriever}")

    return torch.cat(all_embeddings, dim=0).cpu().numpy()




def build_faiss_index(
    embeddings: torch.Tensor,
    metric: str = 'ip'
):
    """
    Build vector index using Faiss.
    embeddings: torch.Tensor or numpy.ndarray, shape [N, D]
    metric: 'ip' or 'l2'
    Returns: faiss.Index
    """
    if faiss is None:
        raise ImportError("faiss is not installed, please `pip install faiss-gpu` or `faiss-cpu`.")

    if isinstance(embeddings, torch.Tensor):
        xb = embeddings.detach().cpu().numpy()
    else:
        xb = embeddings

    xb = xb.astype('float32')
    dim = xb.shape[1]

    if metric == 'l2':
        index = faiss.IndexFlatL2(dim)
    elif metric == 'ip':
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(xb)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    index.add(xb)
    return index


def faiss_knn_search(
    index,
    query_embeddings: torch.Tensor,
    top_k: int = 8,
    metric: str = 'ip'
):
    """
    Perform KNN retrieval using Faiss.
    Returns: (distances, indices)
    """
    if isinstance(query_embeddings, torch.Tensor):
        xq = query_embeddings.detach().cpu().numpy()
    else:
        xq = query_embeddings

    xq = xq.astype('float32')
    if metric == 'ip':
        faiss.normalize_L2(xq)

    D, I = index.search(xq, top_k)
    return D, I




if __name__ == '__main__':
    pass

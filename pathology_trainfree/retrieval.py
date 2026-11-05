import torch
import torch.nn.functional as F


class SimilarityRetriever:
    def __init__(self, memory, cfg, device):
        self.ids = [int(x) for x in cfg["memory"]["class_ids"]]
        self.k = int(cfg["retrieval"]["top_k"])
        self.chunk_size = int(cfg["retrieval"]["chunk_size"])
        self.banks = {}
        for cls in self.ids:
            bank = torch.from_numpy(memory.vectors[cls].astype("float32")).to(device)
            if len(bank) < self.k:
                raise ValueError(f"Top-K={self.k} exceeds class {cls} memory size {len(bank)}")
            self.banks[cls] = F.normalize(bank, dim=1)
        self.device = device

    @torch.inference_mode()
    def scores(self, tokens):
        tokens = torch.as_tensor(tokens, dtype=torch.float32, device=self.device).reshape(-1, next(iter(self.banks.values())).shape[1])
        tokens = F.normalize(tokens, dim=1)
        chunks = []
        for chunk in tokens.split(self.chunk_size):
            chunks.append(torch.stack([(chunk @ self.banks[cls].T).topk(self.k, dim=1).values.mean(1)
                                       for cls in self.ids], dim=1))
        return torch.cat(chunks).cpu()


def classify(scores, class_ids, threshold=0.0):
    ids = [int(x) for x in class_ids]
    if len(ids) == 2 and ids == [0, 1]:
        margin = scores[..., 1] - scores[..., 0]
        return (margin > threshold).to(torch.uint8), margin
    index = scores.argmax(dim=-1)
    mapping = torch.tensor(ids, device=scores.device)
    return mapping[index], None

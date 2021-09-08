import argparse
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer
import logging
from ..models.multnat_model import MultNatModel


def print_nn(pattern: np.array, nbrs: NearestNeighbors,
             tokenizer: RobertaTokenizer):
    distances, indices = nbrs.kneighbors(X=pattern)
    lines = [[] for _ in range(indices.shape[1])]
    for dists, neighbors in zip(distances, indices):
        for i, (d, n) in enumerate(zip(dists, neighbors)):
            sim = 1 - d*d/2
            if n < len(tokenizer):
                lines[i].append("{:25s} ({:.2f})".format(
                    tokenizer.convert_ids_to_tokens([n])[0], sim))
            else:
                lines[i].append("C")

    for line in lines:
        print("\t".join(line))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('num_patterns', type=int)
    parser.add_argument('num_tokens_per_pattern', type=int)
    parser.add_argument('--num-nn', default=5, type=int)
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    logger.info('Loading model')
    model = MultNatModel.load_from_checkpoint(args.checkpoint)
    logger.info('Done loading model')

    logger.info('Converting embeddings to numpy')
    emb = model.model.get_input_embeddings()
    contokens = emb.weight[-args.num_patterns *
                           args.num_tokens_per_pattern:].detach().numpy()
    patterns = [
        normalize(contokens[s:s+args.num_tokens_per_pattern])
        for s in range(0, contokens.shape[0],
                       args.num_tokens_per_pattern)
    ]

    if len(patterns) != args.num_patterns:
        print("ERROR: Found {} patterns but expected {}".format(
            len(patterns), args.num_patterns))
        print("contokens:", contokens.shape)
        print("patterns[0]:", patterns[0].shape)
        # print(patterns)
        exit(1)

    logger.info('Constructing ball tree of embeddings')
    nbrs = NearestNeighbors(
        n_neighbors=args.num_nn, algorithm='ball_tree'
    ).fit(normalize(emb.weight.detach().numpy()))

    logger.info('Start nearest neighbors search')
    for pat in patterns:
        print_nn(pat, nbrs, model.tokenizer)
        print('-----')

    similarities = cosine_similarity(contokens)
    with np.printoptions(precision=2, suppress=True):
        print(similarities)

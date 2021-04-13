import sys, argparse
import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('--query_embed', help='query embedding file')
parser.add_argument('--target_embed', help='target embedding file')
parser.add_argument('--behavior_file', help='behavior file')
args = parser.parse_args()

query_embed, target_embed = {}, {}
sys.stderr.write(f'load query emnbeddings from {args.query_embed}\n')
with open(args.query_embed, 'r') as f:
    for line in f:
        entity, embed = line.rstrip('\n').split('\t')
        query_embed[entity] = np.array(embed.split(' '), dtype=float)

sys.stderr.write(f'load target emnbeddings from {args.target_embed}\n')
with open(args.target_embed, 'r') as f:
    for line in f:
        entity, embed = line.rstrip('\n').split('\t')
        target_embed[entity] = np.array(embed.split(' '), dtype=float)

sys.stderr.write(f'make predictions on {args.behavior_file}\n')
cache = defaultdict(float)
observed_uids = []
with open(args.behavior_file, 'r') as f:
    for line in tqdm(f):
        impression, uid, time, history, targets = line.rstrip('\n').split('\t')
        targets = targets.split(' ')
        scores = np.zeros(len(targets), dtype=float)
        # user -> item scoring
        if uid in query_embed:
            observed_uids.append(uid)
            for e, tid in enumerate(targets):
                tid = tid.split('-')[0]
                scores[e] += np.dot(query_embed[uid], target_embed[tid])
        # item -> item scoring
        elif history:
            for qid in history.split(' '):
                _scores = np.zeros(len(targets), dtype=float)
                for e, tid in enumerate(targets):
                    tid = tid.split('-')[0]
                    key = f"{qid} {tid}"
                    if key not in cache:
                        if qid in query_embed and tid in target_embed:
                            cache[key] = np.dot(query_embed[qid], target_embed[tid])
                        else:
                            cache[key] = 0.
                    _scores[e] = cache[key]
                # get ranks
                _ranks = ['N']*len(_scores)
                sorted_indexes = np.argsort(_scores)
                for r, i in enumerate(sorted_indexes, 1):
                    scores[i] += r
        # random user -> item scoring
        else:
            uid = random.choice(observed_uids)
            for e, tid in enumerate(targets):
                tid = tid.split('-')[0]
                scores[e] += np.dot(query_embed[uid], target_embed[tid])

        # display the final ranking
        ranks = ['N']*len(scores)
        sorted_indexes = np.argsort(-scores)
        for r, i in enumerate(sorted_indexes, 1):
            ranks[i] = str(r)
        sep = ','
        ranks = f"[{sep.join(ranks)}]"
        print(f"{impression} {ranks}")


"""設計データ・評価データの全系列を CachedSequence で読む。"""
from magpdr.data import sequences as P
from magpdr.heading.kalman.cached_sequence import CachedSequence

SPLIT_DIR = {"design": "seen_subjects_test_set", "test": "unseen_subjects_test_set"}


def load_seqs(split):
    """split（'design' か 'test'）の全系列を CachedSequence のリストで返す。"""
    sd = SPLIT_DIR[split]
    return [CachedSequence(sd, n) for n in P.seq_list(sd)]

import os
from argparse import ArgumentParser

from gensim.models import Word2Vec

from data.stream.iters import PlaylistIterator

from .callbacks import LogEpochLossCallback


def main():
    parser = ArgumentParser("Train song2vec model")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=10, help="context window size")
    parser.add_argument("--mincount", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--debug", type=int, default=-1)
    parser.add_argument("--data", type=str, required=True, help="path to train data")
    parser.add_argument("--output", type=str, required=True, help="path to model to be saved")
    parser.add_argument("--savename", type=str, default="model")
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    assert os.path.isdir(args.data)
    assert os.path.isdir(args.output)

    limit = args.debug if args.debug > 0 else None
    files = [os.path.join(args.data, fname) for fname in os.listdir(args.data)]

    loader = PlaylistIterator(files, limit)
    logger = LogEpochLossCallback()

    # a playlist is a sentence, a song is a word
    sentences = []
    for pl in loader:
        songs = [track["track_uri"] for track in pl["tracks"]]
        if len(songs) > 1:
            sentences.append(songs)

    model = Word2Vec(
        sentences=sentences,
        vector_size=args.dim,
        window=args.window,
        min_count=args.mincount,
        sg=1,
        epochs=args.epochs,
        workers=args.workers,
        compute_loss=True,
        callbacks=[logger],
    )

    print(f"Saving model to {args.output}/{args.savename}")
    model.save(os.path.join(args.output, args.savename))


if __name__ == "__main__":
    main()

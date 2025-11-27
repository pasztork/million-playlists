import json
import os
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("--in-folder", type=str, required=True, help="playlist folder")
    parser.add_argument("--out-folder", type=str, required=True, help="output file folder")
    args = parser.parse_args()

    mappings = {}

    for filename in tqdm(os.listdir(args.in_folder)):
        if filename.endswith(".json"):
            with open(os.path.join(args.in_folder, filename), "r") as f:
                data = json.load(f)
                for playlist in data["playlists"]:
                    for track in playlist["tracks"]:
                        uri = track["track_uri"]
                        mappings[uri] = {
                            "name": track.get("track_name", ""),
                            "artist": track.get("artist_name", ""),
                            "album": track.get("album_name", ""),
                        }

    # convert mappings to a DataFrame and save
    df = pd.DataFrame.from_dict(mappings, orient="index").rename_axis("uri").reset_index()
    df = df[["uri", "name", "artist", "album"]]
    df.to_csv(os.path.join(args.out_folder, "track_uri_mappings.csv"), index=False)


if __name__ == "__main__":
    main()

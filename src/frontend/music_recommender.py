from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

_HERE = Path(__file__).parent.parent.parent
TRACKS_CSV = _HERE / "data" / "track_uri_mappings.csv"
MODEL_FILE = _HERE / "models" / "w2v_v2"
df = pd.read_csv(TRACKS_CSV)
df = df.astype({"uri": "string", "name": "string", "artist": "string", "album": "string"})
model = Word2Vec.load(str(MODEL_FILE))


def search_items(query):
    matches = df[df["name"].str.contains(query, case=False) | df["artist"].str.contains(query, case=False)]
    choices = (matches["name"] + " (" + matches["artist"] + ")").tolist()[:5]
    return gr.Dropdown(choices=choices, value=None)


def on_select(selected_label, current_df):
    if not selected_label:
        return current_df

    title, artist = selected_label.rsplit(" (", 1)
    artist = artist.rstrip(")")

    sel_df = current_df[(current_df["Title"] == title) & (current_df["Artist"] == artist)]
    if not sel_df.empty:
        return current_df

    df = pd.concat([current_df, pd.DataFrame([(title, artist)], columns=["Title", "Artist"])], ignore_index=True)
    df = df.astype({"Title": "string", "Artist": "string"})
    return df


def on_recommend(pl_df):
    if pl_df is None or pl_df.empty:
        return pd.DataFrame([], columns=["Title", "Artist", "Album"])

    playlist = []

    for _, row in pl_df.iterrows():
        title = str(row.get("Title", "") or row.get(0, "")).strip()
        artist = str(row.get("Artist", "") or row.get(1, "")).strip()
        if not title or not artist:
            continue

        mask = (df["name"].str.lower().str.strip().eq(title.lower())) & (
            df["artist"].str.lower().str.strip().eq(artist.lower())
        )
        matches = df.loc[mask, ["uri"]]
        for uri in matches["uri"].to_list():
            if uri in model.wv:
                playlist.append(uri)

    pl_df = pd.DataFrame(playlist, columns=["uri"])
    pl_df = pl_df.merge(df, on="uri", how="left")

    vectors = [model.wv[track] for track in playlist]
    centroid = np.mean(vectors, axis=0)

    recommendations = model.wv.similar_by_vector(centroid, topn=20)
    recommendations = [r for r, _ in recommendations if r not in playlist]

    rec_df = pd.DataFrame(recommendations, columns=["uri"])
    rec_df = rec_df.merge(df, on="uri", how="left")
    rec_df = rec_df[["name", "artist", "album"]]
    rec_df = rec_df.rename(columns={"name": "Title", "artist": "Artist", "album": "Album"})
    return rec_df


def on_clear():
    return None, None, None, None


with gr.Blocks() as demo:
    gr.Markdown("# 🎶 Playlist Recommender")

    search_box = gr.Textbox(label="Search keyword")

    result_dropdown = gr.Dropdown(label="Results", choices=[])

    search_box.submit(
        search_items,
        inputs=[search_box],
        outputs=[result_dropdown],
    )

    playlist = gr.Dataframe(
        headers=["Title", "Artist"],
        datatype=["str", "str"],
        label="Playlist",
        interactive=False,
    )

    result_dropdown.change(
        on_select,
        inputs=[result_dropdown, playlist],
        outputs=[playlist],
    )

    rec_btn = gr.Button("Recommend")

    out = gr.Dataframe(
        headers=["Title", "Artist", "Album"],
        datatype=["str", "str", "str"],
        label="Recommendations",
        interactive=False,
        value=[],
    )

    rec_btn.click(on_recommend, inputs=[playlist], outputs=[out])

    clear_btn = gr.Button("Clear")

    clear_btn.click(
        on_clear,
        inputs=[],
        outputs=[search_box, result_dropdown, playlist, out],
    )

demo.launch(server_name="0.0.0.0", server_port=8000)

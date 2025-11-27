from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

_HERE = Path(__file__).parent.parent.parent
TRACKS_CSV = _HERE / "data" / "track_uri_mappings.csv"
MODEL_FILE = _HERE / "models" / "w2v_v4"

# Load heavy resources once at import time, but expose them to the UI via gr.State
_TRACKS_DF = pd.read_csv(TRACKS_CSV)
_TRACKS_DF = _TRACKS_DF.astype({"uri": "string", "name": "string", "artist": "string", "album": "string"})
_MODEL = Word2Vec.load(str(MODEL_FILE))


def search_items(query, tracks_df):
    if not query:
        return [], {}

    matches = tracks_df[
        tracks_df["name"].str.contains(query, case=False) | tracks_df["artist"].str.contains(query, case=False)
    ]
    rows = matches.head(10)

    choices = []
    uri_map = {}
    for _, row in rows.iterrows():
        title = str(row["name"]).strip()
        artist = str(row["artist"]).strip()
        label = f"{title} ({artist})"
        choices.append(label)
        uri_map[label] = (title, artist, row.get("album", ""))

    return gr.Dropdown(choices=choices, value=None), uri_map


def on_select(selected_label, current_df, uri_map):
    if not selected_label:
        return current_df

    title, artist, album = uri_map[selected_label]
    sel_df = current_df[(current_df["Title"] == title) & (current_df["Artist"] == artist)]
    if not sel_df.empty:
        return current_df

    new_df = pd.concat(
        [current_df, pd.DataFrame([(title, artist, album)], columns=["Title", "Artist", "Album"])], ignore_index=True
    )
    new_df = new_df.astype({"Title": "string", "Artist": "string", "Album": "string"})
    return new_df


def on_recommend(pl_df, tracks_df, model):
    if pl_df is None or pl_df.empty:
        return pd.DataFrame([], columns=["Title", "Artist", "Album"])

    playlist = []

    for _, row in pl_df.iterrows():
        title = str(row.get("Title", "") or row.get(0, "")).strip()
        artist = str(row.get("Artist", "") or row.get(1, "")).strip()
        if not title or not artist:
            continue

        mask = (tracks_df["name"].str.lower().str.strip().eq(title.lower())) & (
            tracks_df["artist"].str.lower().str.strip().eq(artist.lower())
        )
        matches = tracks_df.loc[mask, ["uri"]]
        for uri in matches["uri"].to_list():
            if uri in model.wv:
                playlist.append(uri)

    if not playlist:
        return pd.DataFrame([], columns=["Title", "Artist", "Album"])

    pl_df_uri = pd.DataFrame(playlist, columns=["uri"])
    pl_df_uri = pl_df_uri.merge(tracks_df, on="uri", how="left")

    vectors = [model.wv[track] for track in playlist]
    centroid = np.mean(vectors, axis=0)

    recommendations = model.wv.similar_by_vector(centroid, topn=20)
    recommendations = [r for r, _ in recommendations if r not in playlist]

    rec_df = pd.DataFrame(recommendations, columns=["uri"]).merge(tracks_df, on="uri", how="left")
    rec_df = rec_df[["name", "artist", "album"]]
    rec_df = rec_df.rename(columns={"name": "Title", "artist": "Artist", "album": "Album"})
    return rec_df


def on_clear():
    return None, None, None, None, None


with gr.Blocks() as demo:
    gr.Markdown("# 🎧 Music recommender")

    # States: store mapping for last search, tracks dataframe and the loaded model
    dropdown_state = gr.State({})
    tracks_state = gr.State(_TRACKS_DF)
    model_state = gr.State(_MODEL)

    with gr.Row():
        search_box = gr.Textbox(label="Search keyword")

        result_dropdown = gr.Dropdown(label="Results", choices=[])

        search_box.submit(
            search_items,
            inputs=[search_box, tracks_state],
            outputs=[result_dropdown, dropdown_state],
        )

    playlist = gr.Dataframe(
        headers=["Title", "Artist", "Album"],
        datatype=["str", "str", "str"],
        label="Playlist",
        interactive=False,
        value=pd.DataFrame([], columns=["Title", "Artist", "Album"]).astype(
            {"Title": "string", "Artist": "string", "Album": "string"}
        ),
    )

    result_dropdown.change(
        on_select,
        inputs=[result_dropdown, playlist, dropdown_state],
        outputs=[playlist],
    )

    rec_btn = gr.Button("Recommend")

    out = gr.Dataframe(
        headers=["Title", "Artist", "Album"],
        datatype=["str", "str", "str"],
        label="Recommendations",
        interactive=False,
        value=pd.DataFrame([], columns=["Title", "Artist", "Album"]).astype(
            {"Title": "string", "Artist": "string", "Album": "string"}
        ),
    )

    rec_btn.click(on_recommend, inputs=[playlist, tracks_state, model_state], outputs=[out])

    clear_btn = gr.Button("Clear")

    clear_btn.click(
        on_clear,
        inputs=[],
        outputs=[search_box, result_dropdown, playlist, out, dropdown_state],
    )

theme = gr.themes.Soft(primary_hue=gr.themes.colors.purple, secondary_hue=gr.themes.colors.fuchsia)
demo.launch(server_name="0.0.0.0", server_port=8000, theme=theme)

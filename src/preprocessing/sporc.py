"""SPoRC -> data/interim/sporc/{train,val}.jsonl (Track NC, tier 1).

Input: data/raw/sporc/turns/text/*.parquet (185M diarised turns) joined to
data/raw/sporc/episodes/*.parquet for language and metadata.

Selection (decisions recorded in DATASHEET/notes, measured 2026-09-02):
- English only (`language` starts with 'en'; 99%+ of episodes).
- Conversational: 2 to 7 distinct speaker labels per episode, counted from the
  turns. This drops 227,574 monologue episodes (612M tokens) and the long tail
  of 8+ labels, which is mostly diarisation over-segmentation. 2-7 is also the
  speaker range the target distribution spans. SPoRC's own `num_main_speakers`
  is 0 for 751k episodes and is not used.
- At least 20 turns.
- Boilerplate: turns of >= 15 tokens whose lowercased text recurs in >= 3
  episodes of the same podcast are dropped (measured: 10,032 turn types, 3.0M
  tokens, 0.07% of the corpus; advertising is rarely verbatim-identical after
  ASR, so deeper ad detection is not worth building yet).

Rendering: one turn per line as `SPEAKER_NN: text`, SPoRC's labels kept as-is
(they are already the target's convention). Overlapping turns carry a list of
labels; the first is used. Val split: 2% of episodes by hash.

Usage: uv run python -m src.preprocessing.sporc [--limit N]
"""

import argparse

import duckdb

from src.preprocessing.common import RAW, Document, Writer, render_turns

TURNS = str(RAW / "sporc" / "turns" / "text" / "*.parquet")
EPISODES = str(RAW / "sporc" / "episodes" / "*.parquet")
ASR_SYSTEM = "SPoRC pipeline: Whisper transcription + diarisation (as distributed)"

QUERY = f"""
WITH boiler AS (
    SELECT podcast_id, hash(lower(turn_text)) AS h
    FROM '{TURNS}' WHERE token_count >= 15
    GROUP BY 1, 2 HAVING count(DISTINCT episode_id) >= 3
),
ep_stats AS (
    SELECT episode_id, count(DISTINCT speaker[1]) AS n_spk, count(*) AS n_turns
    FROM '{TURNS}' GROUP BY episode_id
),
keep AS (
    SELECT s.episode_id, s.n_spk, e.podcast_id, e.pod_title, e.category1, e.duration_seconds
    FROM ep_stats s JOIN '{EPISODES}' e USING (episode_id)
    WHERE s.n_spk BETWEEN 2 AND 7 AND s.n_turns >= 20 AND lower(e.language) LIKE 'en%'
)
SELECT t.episode_id, k.podcast_id, k.pod_title, k.category1, k.duration_seconds, k.n_spk,
       t.speaker[1] AS spk, t.turn_text, t.token_count
FROM '{TURNS}' t
JOIN keep k USING (episode_id)
ANTI JOIN boiler b ON b.podcast_id = t.podcast_id AND b.h = hash(lower(t.turn_text)) AND t.token_count >= 15
ORDER BY t.episode_id, t.start_time
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="stop after N episodes (0 = all)")
    args = parser.parse_args()

    con = duckdb.connect()
    con.execute("SET memory_limit='24GB'")
    con.execute("SET temp_directory='data/interim/.duckdb_tmp'")
    con.execute("SET threads=8")
    cur = con.execute(QUERY)

    writer = Writer("sporc")
    n_eps = 0
    cur_ep = None
    turns: list[tuple[str, str]] = []
    meta: dict = {}
    src_tokens = 0

    def flush() -> None:
        nonlocal n_eps
        if not turns:
            return
        text = render_turns(turns)
        doc = Document(
            source="sporc", doc_id=cur_ep, track="track-nc", tier=1, asr_system=ASR_SYSTEM,
            has_speakers=True, n_turns=len(turns), n_words=sum(len(t.split()) for _, t in turns),
            text=text, meta={**meta, "source_token_count": src_tokens},
        )
        writer.add(doc, cur_ep)
        n_eps += 1
        if n_eps % 10000 == 0:
            print(f"{n_eps} episodes", flush=True)

    while batch := cur.fetchmany(50_000):
        for ep, pod, title, cat, dur, n_spk, spk, text, tok in batch:
            if ep != cur_ep:
                flush()
                if args.limit and n_eps >= args.limit:
                    cur_ep = None
                    turns = []
                    break
                cur_ep, turns, src_tokens = ep, [], 0
                meta = {"podcast_id": pod, "pod_title": title, "category": cat,
                        "duration_s": dur, "n_speakers": n_spk}
            if text:
                turns.append((spk or "SPEAKER_00", text.strip()))
                src_tokens += tok or 0
        else:
            continue
        break
    flush()
    writer.close()


if __name__ == "__main__":
    main()

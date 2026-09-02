"""MeetingBank -> data/interim/meetingbank/{train,val}.jsonl (Track NC, tier 1).

Input: data/raw/meetingbank/{train,validation,test}.json (JSONL despite the
extension), one record per agenda item with `uid` = <council>_<date>_<item>.

The HF distribution is per agenda item and flat (no speaker labels; those ship
with the audio release). Whole meetings are reconstructed by grouping items on
<council>_<date>, preserving the record order within the source files, which
is how the ~28k-token documents come about. has_speakers=False. The human
summaries are kept in meta (summary supervision for the encoder-decoder arm).
Val split: 2% of meetings by hash; the source's own split is not used because
it splits by item, which would leak meetings across train and val.

Usage: uv run python -m src.preprocessing.meetingbank
"""

import json
from collections import OrderedDict

from src.preprocessing.common import RAW, Document, Writer

ASR_SYSTEM = "Speechmatics (as distributed by MeetingBank)"


def main() -> None:
    meetings: "OrderedDict[str, list[dict]]" = OrderedDict()
    for split in ("train", "validation", "test"):
        for line in (RAW / "meetingbank" / f"{split}.json").open():
            r = json.loads(line)
            council, date, _item = r["uid"].split("_", 2)
            meetings.setdefault(f"{council}_{date}", []).append(r)

    writer = Writer("meetingbank")
    for mid, items in meetings.items():
        text = "\n\n".join(i["transcript"].strip() for i in items if i.get("transcript"))
        if not text:
            continue
        council, date = mid.split("_", 1)
        writer.add(
            Document(
                source="meetingbank", doc_id=mid, track="track-nc", tier=1, asr_system=ASR_SYSTEM,
                has_speakers=False, n_turns=0, n_words=len(text.split()), text=text,
                meta={"council": council, "date": date, "n_items": len(items),
                      "summaries": " ||| ".join(i.get("summary", "") for i in items)},
            ),
            mid,
        )
    print(f"meetings reconstructed: {len(meetings)}")
    writer.close()


if __name__ == "__main__":
    main()

"""The ASR channel model: what a recogniser does to speech.

This is the enabling technology for the project's core science, not a side
component. All the permissively-licensed real ASR text that exists totals under
10M tokens, and Arm E needs 3B per corpus, so the only route to a controlled
written-vs-spoken ablation is to take clean text and put it through a channel
fitted from real paired data.

## What is fitted

AMI is the one Track P corpus that transcribes the same speech twice: a human
verbatim layer and real recogniser output from 2007. Aligning those word streams
gives, per word, whether the recogniser got it right, replaced it, dropped it, or
invented one. That is the channel.

Three transformations are separated deliberately, because they have different
statistics and different fixes:

1. **Deterministic surface effects** — casing collapse and punctuation removal.
   A recogniser of this era emits neither. These are near-certain, so modelling
   them probabilistically would add noise without adding realism.
2. **Lexical errors** — substitutions, deletions and insertions, estimated as
   distributions from the alignment.
3. **Severity** — a scalar that scales the error rates, because the reference
   recogniser is from **February 2007** and its word error rate is far above a
   modern system's. Fitting it raw and applying it unscaled would produce text
   noisier than anything this project will actually see.

## Why alignment is time-chunked

The two layers segment turns completely differently: ASR produces roughly 1.75x
as many turns as the human transcriber on the same meetings, so turn-to-turn
comparison would align text that is not parallel. Word-level edit distance over a
whole speaker stream is the right comparison but is O(n*m) and the streams reach
tens of thousands of words.

Both layers carry word timings, so the streams are cut into short time windows
first and aligned within each. That bounds the cost and, more importantly, stops
a single bad match early on from propagating misalignment through the rest of a
meeting.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from tt.data.loaders.ami import TimedWord

_PUNCT_ONLY = re.compile(r"^[^\w]+$", re.UNICODE)
_STRIP_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


class Op(StrEnum):
    """One alignment outcome for one reference or hypothesis word."""

    MATCH = "match"
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    """In the reference, absent from the hypothesis. The recogniser dropped it."""

    INSERTION = "insertion"
    """In the hypothesis only. The recogniser invented it, often from cross-talk."""


@dataclass(frozen=True)
class Edit:
    """One aligned position."""

    op: Op
    ref: str | None
    hyp: str | None


def is_punctuation(token: str) -> bool:
    """True for tokens that are punctuation alone.

    AMI's manual layer emits punctuation as its own word element. Those are not
    spoken words and must be removed before alignment, or every one of them
    aligns as a deletion and the deletion rate is inflated by the transcriber's
    comma usage rather than the recogniser's behaviour.
    """
    return bool(_PUNCT_ONLY.match(token))


def normalise(token: str) -> str:
    """Fold a token to the form on which reference and hypothesis are comparable.

    Casing and edge punctuation are stripped because the recogniser removes them
    deterministically. Leaving them in would score every correctly recognised
    word as a substitution.
    """
    return _STRIP_EDGE_PUNCT.sub("", token).lower()


def align_words(reference: list[str], hypothesis: list[str]) -> list[Edit]:
    """Levenshtein alignment with backtrace, on already-normalised tokens.

    Costs are the standard 1 per edit. Ties prefer substitution over an
    insertion/deletion pair, which keeps a mis-recognised word reported as one
    error rather than two.
    """
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return [Edit(Op.INSERTION, None, h) for h in hypothesis]
    if m == 0:
        return [Edit(Op.DELETION, r, None) for r in reference]

    # dp[i][j] = cost of aligning reference[:i] with hypothesis[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
            dp[i][j] = min(sub, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    edits: list[Edit] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = reference[i - 1] == hypothesis[j - 1]
            if dp[i][j] == dp[i - 1][j - 1] + (not same):
                edits.append(
                    Edit(
                        Op.MATCH if same else Op.SUBSTITUTION,
                        reference[i - 1],
                        hypothesis[j - 1],
                    )
                )
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            edits.append(Edit(Op.DELETION, reference[i - 1], None))
            i -= 1
            continue
        edits.append(Edit(Op.INSERTION, None, hypothesis[j - 1]))
        j -= 1
    edits.reverse()
    return edits


def align_streams(
    reference: list[TimedWord], hypothesis: list[TimedWord], *, window_s: float = 20.0
) -> list[Edit]:
    """Align two timed word streams, chunked by time.

    Punctuation-only reference tokens are dropped first: they are the
    transcriber's, not the speaker's, and counting them as deletions would
    measure comma usage rather than recogniser behaviour.
    """
    ref = [w for w in reference if not is_punctuation(w.text)]
    hyp = [w for w in hypothesis if not is_punctuation(w.text)]
    if not ref and not hyp:
        return []

    latest = max([w.start for w in ref + hyp], default=0.0)
    n_windows = int(latest // window_s) + 1

    ref_by: list[list[str]] = [[] for _ in range(n_windows)]
    hyp_by: list[list[str]] = [[] for _ in range(n_windows)]
    for w in ref:
        ref_by[min(int(w.start // window_s), n_windows - 1)].append(normalise(w.text))
    for w in hyp:
        hyp_by[min(int(w.start // window_s), n_windows - 1)].append(normalise(w.text))

    edits: list[Edit] = []
    for r, h in zip(ref_by, hyp_by, strict=True):
        r = [x for x in r if x]
        h = [x for x in h if x]
        if r or h:
            edits.extend(align_words(r, h))
    return edits


@dataclass
class ChannelStats:
    """Fitted channel parameters.

    Counts rather than rates, so that fits over different amounts of data can be
    summed, and so the evidence behind any rate is inspectable.
    """

    substitutions: Counter[tuple[str, str]] = field(default_factory=Counter)
    """(reference word, what the recogniser produced instead) -> count."""

    deletions: Counter[str] = field(default_factory=Counter)
    insertions: Counter[str] = field(default_factory=Counter)
    reference_counts: Counter[str] = field(default_factory=Counter)
    """How often each reference word appeared at all. The denominator for per-word rates."""

    n_reference_words: int = 0
    n_hypothesis_words: int = 0
    n_matches: int = 0
    source: str = ""
    asr_system: str = ""

    @property
    def n_substitutions(self) -> int:
        return sum(self.substitutions.values())

    @property
    def n_deletions(self) -> int:
        return sum(self.deletions.values())

    @property
    def n_insertions(self) -> int:
        return sum(self.insertions.values())

    @property
    def wer(self) -> float:
        """Word error rate, the standard (S + D + I) / N_reference."""
        if not self.n_reference_words:
            return 0.0
        return (
            self.n_substitutions + self.n_deletions + self.n_insertions
        ) / self.n_reference_words

    @property
    def substitution_rate(self) -> float:
        return self.n_substitutions / self.n_reference_words if self.n_reference_words else 0.0

    @property
    def deletion_rate(self) -> float:
        return self.n_deletions / self.n_reference_words if self.n_reference_words else 0.0

    @property
    def insertion_rate(self) -> float:
        return self.n_insertions / self.n_reference_words if self.n_reference_words else 0.0

    def update(self, edits: list[Edit]) -> None:
        for edit in edits:
            if edit.op is Op.MATCH and edit.ref is not None:
                self.n_matches += 1
                self.reference_counts[edit.ref] += 1
                self.n_reference_words += 1
                self.n_hypothesis_words += 1
            elif edit.op is Op.SUBSTITUTION and edit.ref is not None and edit.hyp is not None:
                self.substitutions[(edit.ref, edit.hyp)] += 1
                self.reference_counts[edit.ref] += 1
                self.n_reference_words += 1
                self.n_hypothesis_words += 1
            elif edit.op is Op.DELETION and edit.ref is not None:
                self.deletions[edit.ref] += 1
                self.reference_counts[edit.ref] += 1
                self.n_reference_words += 1
            elif edit.op is Op.INSERTION and edit.hyp is not None:
                self.insertions[edit.hyp] += 1
                self.n_hypothesis_words += 1

    def summary(self) -> dict[str, float | int | str]:
        return {
            "source": self.source,
            "asr_system": self.asr_system,
            "n_reference_words": self.n_reference_words,
            "n_hypothesis_words": self.n_hypothesis_words,
            "n_matches": self.n_matches,
            "n_substitutions": self.n_substitutions,
            "n_deletions": self.n_deletions,
            "n_insertions": self.n_insertions,
            "wer": round(self.wer, 4),
            "substitution_rate": round(self.substitution_rate, 4),
            "deletion_rate": round(self.deletion_rate, 4),
            "insertion_rate": round(self.insertion_rate, 4),
            "distinct_substitution_pairs": len(self.substitutions),
            "distinct_deleted_words": len(self.deletions),
            "distinct_inserted_words": len(self.insertions),
        }

    def save(self, path: Path) -> None:
        """Serialise for versioning as a wandb artifact.

        Every generated corpus records which channel version produced it, since
        comparing models trained under different channels is a silent confound.
        """
        path.write_text(
            json.dumps(
                {
                    "summary": self.summary(),
                    "substitutions": [[r, h, c] for (r, h), c in self.substitutions.most_common()],
                    "deletions": self.deletions.most_common(),
                    "insertions": self.insertions.most_common(),
                    "reference_counts": self.reference_counts.most_common(),
                },
                indent=1,
            )
        )

    @classmethod
    def load(cls, path: Path) -> ChannelStats:
        raw = json.loads(path.read_text())
        stats = cls(
            source=str(raw["summary"].get("source", "")),
            asr_system=str(raw["summary"].get("asr_system", "")),
            n_reference_words=int(raw["summary"]["n_reference_words"]),
            n_hypothesis_words=int(raw["summary"]["n_hypothesis_words"]),
            n_matches=int(raw["summary"]["n_matches"]),
        )
        stats.substitutions = Counter({(r, h): c for r, h, c in raw["substitutions"]})
        stats.deletions = Counter(dict(raw["deletions"]))
        stats.insertions = Counter(dict(raw["insertions"]))
        stats.reference_counts = Counter(dict(raw["reference_counts"]))
        return stats

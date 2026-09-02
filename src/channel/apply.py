"""Apply a fitted ASR channel to clean text (SYNTHSHEET route A, word-edit layer).

v2.1 span model. Given an artifact from src/channel/fit.py, turns a normalised
token stream into an ASR-style hypothesis by sampling fitted *edit spans*:

- At each position, the longest reference n-gram (n <= 3) with a fitted edit
  distribution is considered first. With probability edits(span)/count(span)
  (smoothed towards pooled rates for rare spans) the span is replaced by a
  sampled hypothesis phrase, which may be empty (deletion), a single word
  (substitution), or several words (`gonna` -> `going to`). This is what makes
  multi-token recogniser behaviour come out as phrases rather than the isolated
  halves the word-independent v2 produced.
- Words never seen in the reference back off to pooled deletion/substitution
  rates, with the substitute drawn as the nearest hypothesis-vocabulary word by
  string similarity (proper nouns decoding as near-homophone common words).
- Insertion runs are sampled as whole phrases (`thank you`, `okay okay`) at the
  fitted per-token cluster rate.
- Burstiness: each document draws a severity multiplier from the fitted per-file
  WER distribution, so some documents come out much noisier than others, as
  real files do. Disable with burst=False for a flat channel.

`severity` scales every edit probability on top (1.0 = as fitted). Everything is
driven by an explicit numpy Generator for reproducibility.

Word-edit layer only: output is lowercase, unpunctuated tokens in the channel's
normalised form. Punctuation/casing, marker tags and turn effects are the
surface layer (SYNTHSHEET section 4, not yet implemented).

Usage (programmatic): Channel.load(path).noise(tokens, rng)
CLI demo:            uv run python -m src.channel.apply "Hello, (um) I'm calling about my pension."
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from src.channel.fit import MAX_REF_SPAN, normalise

SMOOTH_K = 5.0
MIN_SPAN_EDITS = 2  # multi-word spans need this many observed edits to be used
P_CAP = 0.995  # near-deterministic normalisations (ohh->oh) must be allowed to fire
NEIGHBOUR_CUTOFF = 60  # fuzz.ratio; below this an unseen word is kept rather than mangled


class Channel:
    def __init__(
        self, artifact: dict, severity: float = 1.0, burst: bool = True, calibrated: bool = True
    ):
        self.version: str = artifact["version"]
        self.severity = severity * (artifact.get("calibration", 1.0) if calibrated else 1.0)
        self.burst = burst
        pooled = artifact["pooled"]
        self.pooled_wer: float = pooled["wer"]
        self.p_del_global = pooled["del_rate"]
        self.p_sub_global = pooled["sub_rate"]
        self.span_counts: dict[str, int] = pooled["span_counts"]
        self.ref_counts: dict[str, int] = pooled["reference_counts"]
        # A word's chance of being edited counts every cluster it took part in,
        # not only the clusters keyed by the word alone.
        self.involvement: dict[str, int] = pooled.get("word_involvement", {})

        grouped: dict[str, dict[str, int]] = {}
        for key, c in pooled["span_edits"].items():
            r, h = key.split("\t")
            grouped.setdefault(r, {})[h] = c
        self.spans: dict[str, tuple[list[str], np.ndarray, int]] = {}
        for r, hs in grouped.items():
            phrases = list(hs)
            counts = np.array([hs[p] for p in phrases], dtype=float)
            self.spans[r] = (phrases, counts / counts.sum(), int(counts.sum()))

        ins = pooled["insertion_phrases"]
        self.ins_phrases = list(ins)
        ins_counts = np.array([ins[p] for p in self.ins_phrases], dtype=float)
        self.ins_p = ins_counts / ins_counts.sum()
        self.p_ins_cluster = pooled["n_insertion_clusters"] / pooled["n_ref"]

        self.file_wers = np.array(pooled["file_wers"] or [self.pooled_wer])
        self.hyp_vocab = [w for w, c in pooled["hypothesis_counts"].items() if c >= 3]
        self._neighbour_cache: dict[str, str | None] = {}

    @classmethod
    def load(cls, path: str | Path, severity: float = 1.0, burst: bool = True) -> "Channel":
        return cls(json.loads(Path(path).read_text()), severity, burst)

    # The 1-gram outcome distribution is the word's own span mapping; when a word
    # was only ever edited inside larger clusters it has none, so _edit falls
    # back to the pooled delete/substitute split.

    def _p_edit(self, span: str, n: int, sev: float) -> float:
        """Probability that this reference span is edited (any hyp phrase)."""
        edits = self.spans[span][2] if span in self.spans else 0
        if n > 1:
            if edits < MIN_SPAN_EDITS:
                return 0.0
            count = max(self.span_counts.get(span, edits), edits)
            return min(sev * edits / count, P_CAP)
        c = self.ref_counts.get(span, 0)
        lam = c / (c + SMOOTH_K)
        p_cond = min(self.involvement.get(span, edits) / c, 1.0) if c else 0.0
        p = lam * p_cond + (1 - lam) * (self.p_del_global + self.p_sub_global)
        return min(sev * p, P_CAP)

    def _neighbour(self, w: str) -> str | None:
        if w not in self._neighbour_cache:
            hit = process.extractOne(w, self.hyp_vocab, scorer=fuzz.ratio, score_cutoff=NEIGHBOUR_CUTOFF)
            self._neighbour_cache[w] = hit[0] if hit and hit[0] != w else None
        return self._neighbour_cache[w]

    def _edit(self, span: str, rng: np.random.Generator) -> list[str]:
        if span in self.spans:
            phrases, p, _ = self.spans[span]
            out = phrases[rng.choice(len(phrases), p=p)]
            return out.split() if out else []
        # unseen single word: pooled del vs sub split, neighbour as substitute
        if rng.random() < self.p_del_global / (self.p_del_global + self.p_sub_global):
            return []
        nb = self._neighbour(span)
        return [nb] if nb else [span]

    @staticmethod
    def _emit(out: list[str], words: list[str]) -> None:
        """Append generated words, dropping one that would duplicate the previous
        output token. Real recogniser decoding suppresses accidental `the the`;
        genuine repeats survive because they come through the un-edited path."""
        for w in words:
            if out and out[-1] == w:
                continue
            out.append(w)

    def noise(self, tokens: list[str], rng: np.random.Generator) -> list[str]:
        sev = self.severity
        if self.burst:
            sev *= float(rng.choice(self.file_wers)) / self.pooled_wer
        p_ins = min(self.p_ins_cluster * sev, P_CAP)
        out: list[str] = []
        i = 0
        while i < len(tokens):
            edited = False
            for n in range(min(MAX_REF_SPAN, len(tokens) - i), 0, -1):
                span = " ".join(tokens[i:i + n])
                if n > 1 and span not in self.spans:
                    continue
                if rng.random() < self._p_edit(span, n, sev):
                    self._emit(out, self._edit(span, rng))
                    i += n
                    edited = True
                    break
            if not edited:
                out.append(tokens[i])
                i += 1
            if rng.random() < p_ins:
                phrase = self.ins_phrases[rng.choice(len(self.ins_phrases), p=self.ins_p)]
                self._emit(out, phrase.split())
        return out

    def noise_text(self, text: str, seed: int) -> str:
        rng = np.random.default_rng(seed)
        return " ".join(self.noise(normalise(text), rng))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="+")
    parser.add_argument("--artifact", default="data/channel/channel_v2_degraded.json")
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ch = Channel.load(args.artifact, args.severity)
    text = " ".join(args.text)
    print("clean :", " ".join(normalise(text)))
    for s in range(3):
        print(f"noised: {ch.noise_text(text, args.seed + s)}")


if __name__ == "__main__":
    main()

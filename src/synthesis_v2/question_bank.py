"""The v2 question bank: questions written for long podcast episodes. This module is the source of truth;
data/labelled_data_v2/questions.jsonl (approved) and questions_provisional.jsonl are derived from it.

    uv run python -m src.synthesis_v2.question_bank      # regenerates questions.jsonl (and questions_provisional.jsonl if any)

Nothing here is shared with the v1 bank (src/synthesis/question_bank.py): different family (`podcast`), different ids
(`v2-<family>-<nn>-<slug>`), different output directory. v1 questions the analysts found worth carrying (`spk-24`,
`spk-29`, `hst-09`) appear only as rewritten v2 questions and are not imported.

Design (2026-10-30, D66; three independent analyses then three independent clarity audits, two Opus and one Fable in
each round, under the session's review8 and review9 briefs). Every question had to earn its place against six
requirements: answerable from the transcript alone by applying the stated criteria; dependent on the whole episode (so
the bank pairs two places in the episode, announced against delivered, asked against answered, raised against resolved,
or asks a global negative); resistant to lexical shortcuts; an informative answer distribution; consistent for a
labeller at 10 to 16k tokens (no counting beyond small bands, no quality judgements, every term of degree defined once);
and grouped into lineages so a third of the candidates are proposed as held out (the owner's target for the final cell is at least 10) of training as the zero-shot cell. Questions the pool itself
showed to be decidable from the opening or closing 300 words (host named, guest introduced or thanked, call to action,
next-episode teaser, sponsor present, after-episode commitments) are deliberately absent, and so is any reference to a
line range (owner's rule): position is expressed only through events in the conversation.

Style (the audits' guide): one question sentence with one subject; a two-part question reads "Does [event], and
[outcome]?" with the outcome options first and `none` last, defined positively ("No speaker ..."); where an episode has
several instances the good outcome needs "every" and the bad one "at least one", so mixed episodes map to exactly one
option; criteria are one sentence of observable verbs; the "never happens" value is always `none`; speaker answers use
host / non_host / none (host_only / non_host_only / both / none where both can apply); counts use none / one / two_plus. The shared definitions (VOCAB) are stored once and shown with every question.
Co-hosted shows are common (SPoRC names two or more hosts for 9% of pool episodes and no host for 79%), so hosts are
rendered `host_1:`, `host_2:` and guests `guest_1:`, `guest_2:` in order of first appearance, and the questions say "a
host" and "a non-host speaker".

No question takes a per-episode argument (owner's decision, 2026-11-01, D78): every question is asked of every episode
in the same words. `BANK` is the approved bank (passed the pilots; `questions.jsonl`): since round four (D80) it is every
question that passed on the verbatim rendering, 32 trained and 12 held out. `PROVISIONAL` holds candidates not yet
piloted (`questions_provisional.jsonl`, written only while there are any; empty now).
"""

import json
from pathlib import Path

from src import prompts
from src.synthesis.schema import Question
from src.synthesis_v2 import OUT_DIR

DATASET = "sporc"
FAMILY = "podcast"

# The shared definitions: one block for the whole task, stored once (definitions.txt), not in each question. Every prompt
# that shows a model the bank's questions includes them: the lean labeller prints them once before the questions, and
# every one-question prompt (pilot route, subagent packages, training) opens the question's context with them, through
# `with_definitions` (until 2026-11-02 they were stored inside every description; the rendered prompts are unchanged).
VOCAB = prompts.text(prompts.V2 / "definitions.txt")


def q(id: str, text: str, options: list[tuple[str, str]], description: str = "") -> Question:
    return Question(id=id, source="bank", family=FAMILY, text=text, description=description.strip(),
                    options=[{"value": v, "criteria": c} for v, c in options], tags=[], dataset_allow_list=[DATASET])


def with_definitions(question: Question) -> Question:
    """The question as a model sees it on its own: its context opens with the shared definitions. A v2 bank question
    only (family podcast, id `v2-...`), and idempotent: records written before 2026-11-02 already carry the definitions
    in their description."""
    if question.family != FAMILY or not question.id.startswith("v2-") or question.description.startswith(VOCAB):
        return question
    return question.model_copy(update={"description": f"{VOCAB} {question.description}".strip()})


def without_definitions(question: Question) -> Question:
    """The question's own description, the shared definitions removed if a record carries them."""
    if not question.description.startswith(VOCAB):
        return question
    return question.model_copy(update={"description": question.description[len(VOCAB):].strip()})


# One definition of "a specific number", shared by num-01 and num-02 (and the removed top-09; the round-three audits asked for it).
NUMBER = ("A specific number is a count, amount of money, percentage, score or measurement stated as a fact; approximate figures ('about 30 million') count, vague quantities ('a lot', 'hundreds of') do not. Dates, years, durations, ages, times of day, episode numbers, positions in a ranking or order ('the 25th pick'), ratings given as opinions ('an eight out of ten'), phone numbers and addresses do not count, and neither do numbers in adverts or sponsor reads, in questions, or in imagined or hypothetical examples.")
# The mechanical recurrence rule fam-01 (and the removed top-08) uses instead of the definitions' "another subject in between" (a low bar in
# conversation, the failure that cut arc-03): three or more consecutive lines of four or more words.
LINES3 = ('Lines of fewer than four words, and lines that carry on the same discussion only through a pronoun ("it", "that", '
          '"this", "he", "she", "they"), are skipped when counting the lines between: they neither separate nor join, and a '
          'line of fewer than four words is not a mention either.')

# The questions that passed the first three pilots on the merged rendering. The approved BANK (end of module) is built from
# these and the later rounds, less every id in REMOVED: the round-four re-pilot on the verbatim rendering cut seven of them.
ROUNDS_1_TO_3: list[Question] = [
    # --- arc: what the episode set out to do against what it did ---
    q("v2-arc-05-new-subject-after-wrap-up",
      "After a host starts to wrap up the episode, does any speaker raise a subject that was not discussed earlier?",
      [("new_subject", ("A host starts to wrap up, and after that a speaker raises a subject that no line before the wrap-up "
                       "discussed.")),
       ("no_new_subject", ("A host starts to wrap up, and everything said after that is thanks, sign-off, requests to "
                          "listeners or subjects already discussed.")),
       ("none", "No host starts to wrap up the episode before the transcript ends.")],
      "A host starts to wrap up by thanking a non-host speaker, saying the episode is ending ('before we go', 'that's all for "
      "today') or signing off."),
    # --- cmt: things promised within the episode, and whether they were kept ---
    q("v2-cmt-02-promise-to-listener",
      "Does a speaker promise listeners specific content later in this episode, and is it delivered?",
      [("delivered", "Every such promise is fulfilled in a later line."),
       ("not_delivered", "At least one such promise is never fulfilled in this episode."),
       ("none", "No speaker promises listeners specific content for later in this episode.")],
      "A promise sounds like 'I'll give you the three steps' or 'stay with us for the answer'. The list of what the "
      "episode will cover, given before the main conversation starts, is not a promise here. Show notes and future "
      "episodes do not count."),
    # --- qa: questions posed against questions answered ---
    q("v2-qa-01-host-question-unanswered",
      "Does a host ask a non-host speaker a direct question that is never answered?",
      [("all_answered", "Every direct question a host asks a non-host speaker is responded to, in the next lines or later."),
       ("some_unanswered", ("At least one direct question a host asks a non-host speaker is never responded to in this "
                           "episode.")),
       ("none", "No host asks a non-host speaker a direct question.")],
      "A question is responded to when the speaker asked says something that addresses what it asks; 'I don't know' is a "
      "response, and a question with several parts is responded to if any part is. Greetings, questions the host says need "
      "no answer, and questions the host rephrases before anyone replies do not count."),
    q("v2-qa-03-unknown-answer-resolved",
      "Does a speaker say they do not know the answer to a question raised in the conversation, and does anyone answer "
      "it later?",
      [("answered_later", "Every question a speaker said they could not answer is answered by some speaker in a later line."),
       ("left_open", "At least one question a speaker said they could not answer is never answered in this episode."),
       ("none", "No speaker says they do not know the answer to a question raised in the conversation.")],
      "A guess offered as a guess still counts as not knowing. A speaker failing to recall a detail of their own story does "
      "not count."),
    # --- dis: disagreement and position change between speakers ---
    q("v2-dis-02-pushback-on-host",
      "Does a non-host speaker say a host is wrong about a fact or an opinion, and does the host concede?",
      [("accepted", ("Every time a non-host speaker says a host is wrong, the host concedes or does not restate the "
                    "position.")),
       ("rejected", "At least once, the host restates the original position after a non-host speaker says it is wrong."),
       ("none", "No non-host speaker says a host is wrong about a fact or an opinion.")],
      "Correcting a name, a date or a slip of the tongue does not count."),
    # --- clm: facts, predictions and sources across the episode ---
    q("v2-clm-01-fact-corrected",
      "Is a statement of fact made in this episode, such as a number, a name or a date, later said to be wrong?",
      [("self_corrected", ("The speaker who made the statement says in a later line that it was wrong or gives a different "
                          "version.")),
       ("corrected_by_other", ("Another speaker says in a later line that the statement is wrong or gives a different "
                              "version, and the original speaker does not correct it themselves.")),
       ("none", "No statement of fact is later said to be wrong.")],
      "Facts only; opinions and their clarifications do not count. A correction within the same line does not count."),
    q("v2-clm-03-prediction",
      "Does a speaker predict how a future event outside their own control will turn out, and can the prediction be "
      "checked?",
      [("specific", ("At least one prediction names a checkable outcome: a date, a number, a winner, or whether a named thing "
                    "happens.")),
       ("general", "Speakers predict only directions or tendencies, such as 'this will grow'."),
       ("none", "No speaker predicts a future event outside their own control.")],
      "Hopes, wishes and plans for the speakers' own actions do not count."),
    q("v2-clm-04-sources-cited",
      "How many different named sources do speakers cite as where a statement of fact comes from?",
      [("none", "No speaker names a source for a statement of fact."),
       ("one", "Exactly one named source is cited."),
       ("two_plus", "Two or more different named sources are cited.")],
      "A named source is a study, report, article, book, dataset, organisation or expert identified by title, publisher or "
      "name, cited as where a fact comes from ('according to', 'a study by'). Quotations used for inspiration, books "
      "mentioned as reading and scripture read aloud do not count."),
    # --- adv: recommendations and advice ---
    q("v2-adv-01-alternatives-chosen",
      "Do the speakers compare two or more alternatives a listener could choose between, and does a speaker say which "
      "they would pick?",
      [("chosen", "Alternatives are compared and a speaker later says which one they choose or recommend."),
       ("open", "Alternatives are compared and no speaker says which one they would pick."),
       ("none", "The speakers compare no alternatives a listener could choose between.")],
      "Alternatives are products, methods, places or courses of action. A choice narrated as part of a personal story "
      "does not count."),
    q("v2-adv-02-advice-conditioned",
      "Does a speaker give listeners advice, and does any speaker attach a condition or exception to it?",
      [("conditioned", ("At least one piece of advice is given an explicit condition or exception, before or after it, that "
                       "names when it does not apply.")),
       ("unconditioned", "Advice is given and none of it carries a condition or exception."),
       ("none", "No speaker gives listeners advice.")],
      "Advice is a speaker telling listeners to do or avoid something in their own lives. A general disclaimer read at the "
      "start or end of the episode, and requests to support the show, do not count."),
    q("v2-adv-03-items-recommended",
      "How many different items do speakers explicitly tell listeners to watch, read, buy, use, visit or follow?",
      [("none", "No speaker tells listeners to watch, read, buy, use, visit or follow anything."),
       ("one", "Exactly one item is recommended."),
       ("two_plus", "Two or more different items are recommended.")],
      "An item is a book, show, product, tool, place, account or person. Saying they like something is not a "
      "recommendation. This podcast, its sponsors, and the speakers' own work and accounts do not count."),
    q("v2-adv-04-advice-attribution",
      "Who gives listeners advice?",
      [("host_only", "Only hosts give listeners advice."),
       ("non_host_only", "Only non-host speakers give listeners advice."),
       ("both", "At least one host and at least one non-host speaker give listeners advice."),
       ("none", "No speaker gives listeners advice.")],
      "Advice is a speaker telling listeners to do or avoid something in their own lives; a speaker restating another's "
      "advice counts. Sign-offs ('stay safe') and requests to support the show do not count."),
    # --- seg: the shape of the recording ---
    q("v2-seg-01-structure",
      "How is the episode structured?",
      [("single", ("The same set of speakers converses from start to finish; played clips, adverts, readings and music "
                  "inside the conversation do not break it.")),
       ("segmented", ("The set of speakers changes between two or more parts of the episode, or a host introduces a part as "
                     "a separate segment with its own format.")),
       ("monologue", ("One speaker says nearly every line; other speakers appear only in a jingle, an advert, a clip or as "
                     "single-word reactions."))],
      "If both monologue and segmented apply, answer monologue."),
    # --- adv, continued ---
    q("v2-adv-05-advice-repeated",
      "Does a speaker give listeners advice, and is the same advice given again after another subject has been "
      "discussed in between?",
      [("repeated", ("At least one piece of advice is given again, by any speaker in any words, after another subject has "
                    "been discussed in between.")),
       ("not_repeated", ("Advice is given, and no piece of it is given again after another subject has been discussed in "
                        "between.")),
       ("none", "No speaker gives listeners advice.")],
      "Advice is a speaker telling listeners to do or avoid something in their own lives. Advice is the same when it tells "
      "listeners to do or avoid the same action; restating the episode's general theme does not count."),
    # --- round two (2026-10-31): common-trigger relational questions, after the pilot showed rare-trigger ones fail ---
    q("v2-per-03-story-answered",
      "Does a speaker tell a story from their own life, and does a different speaker then tell one on the same subject?",
      [("answered_with_story", ("After at least one such story, a different speaker tells a story from their own life on the "
                               "same subject before another subject has been discussed in between.")),
       ("no_story_reply", ("Stories from speakers' own lives are told, and none is followed by a different speaker's story on "
                          "the same subject before another subject has been discussed in between.")),
       ("none", "No speaker tells a story from their own life.")],
      "A story is an account of one particular occasion in the speaker's own past, saying what happened, in at least three "
      "sentences. Habits ('I usually'), general background ('I was a lazy kid'), and hypothetical or imagined examples are "
      "not stories. A story told in answer to a question counts."),
    q("v2-self-01-previous-episode",
      "Does a speaker refer to a particular previous episode of this podcast, and does a speaker say what it contained?",
      [("content_restated", ("At least one particular previous episode is referred to, and a speaker says in a sentence what "
                            "was discussed in it or who appeared in it.")),
       ("mentioned_only", "Particular previous episodes are referred to, and no speaker says what any of them contained."),
       ("none", "No speaker refers to a particular previous episode of this podcast.")],
      "Referring sounds like 'last week we', 'in episode twelve', 'our old episode on' or 'when we had X on'. Greetings such "
      "as 'welcome back to another episode', this episode's own number, future episodes and the speakers' other podcasts "
      "are not references to a previous episode."),
    q("v2-adv-06-advice-asked",
      "Does a speaker ask another speaker for advice, and does the answer tell someone what to do?",
      [("advised", ("Every such request is answered with at least one sentence telling someone what they should or should not "
                   "do.")),
       ("not_advised", ("At least one such request gets no answer, or an answer with no sentence telling someone what they "
                       "should or should not do.")),
       ("none", "No speaker asks another speaker for advice.")],
      "Asking for advice sounds like 'what would you tell someone who', 'what should people do' or 'any tips'; asking what "
      "someone thinks of a thing, or what a company or government should do, is not asking for advice. Requests inside a "
      "story or an imagined example do not count. 'You should', 'I'd tell them to' and 'the best thing you can do is' tell "
      "someone what to do; 'what helped me was' does not."),
    q("v2-num-01-figure-sourced",
      "Does a speaker give a specific number as a fact, and does any speaker say where it comes from?",
      [("sourced", ("At least one specific number is said by a speaker to come from a study, report, article, organisation, "
                   "person or measurement.")),
       ("unsourced", "Specific numbers are given as facts, and no speaker says where any of them comes from."),
       ("none", "No speaker gives a specific number as a fact.")],
      NUMBER + " Saying where it comes from sounds like 'according to', 'a study found', 'I read that' or 'they measured'; the "
      "speaker's own experience is not a source."),
    q("v2-hst-01-answer-restated",
      "Does a host restate in their own words an answer a non-host speaker has just given?",
      [("restated", ("At least once, the host's next line after a non-host speaker's answer sums up or paraphrases that answer "
                    "in a sentence.")),
       ("never_restated", ("Non-host speakers answer hosts' questions, and no host's next line ever sums up or paraphrases an "
                          "answer.")),
       ("none", "No non-host speaker answers a question from a host.")],
      "Restating sounds like 'so what you're saying is', 'so basically you' or a summary of the answer in the host's words; "
      "it may lead into a question or a comment. Repeating a few of the speaker's words, thanking, and adding the host's own "
      "view without summing up the answer are not restating."),
    # --- round three (2026-10-31): drafted against the first pilot's lessons, audited by two Opus and one Fable auditor ---
    q("v2-rep-01-reported-speech-named",
      "Does a speaker report words that a particular person said to them on one occasion, and is that person named?",
      [("named", ("At least one person whose words, said to the speaker on one occasion, a speaker reports is called by a "
                 "proper name in the sentence that reports the words or in the sentence before it.")),
       ("unnamed_only", ("Speakers report words that particular people said to them on one occasion, and no person whose words, "
                        "said to the speaker on one occasion, a speaker reports is called by a proper name in the sentence that "
                        "reports the words or in the sentence before it.")),
       ("none", "No speaker reports words that a particular person said to them on one occasion.")],
      "Reporting sounds like 'he told me', 'she said to me', 'my coach goes', 'I asked him and he said' or 'he texted me "
      "saying', followed by what was said, in the person's words or the speaker's own ('he told me to quit'). Said to them "
      "includes said to a group the speaker was part of, and messages written to the speaker. The person is not a speaker in "
      "this episode, and a speaker reporting their own words ('I told him', 'I was like') is not reporting. Only saying that "
      "someone talked about a subject ('he told us about his tour') does not report words. Habits ('she always says', 'he "
      "would be like') are not one occasion and are not reporting. A particular small group ('the two guys at the shop') counts "
      "as one person, named if any member of it is named. A relation or role ('my wife', 'a customer', 'my boss') is not a "
      "proper name; a first name alone is. Quotations from a book, study, broadcast or public statement, sayings, what people "
      "say in general, and citing a person only as the source of a fact or figure ('my doctor says forty per cent of patients "
      "...') do not count."),
    q("v2-plc-01-places-been",
      "How many different named places do speakers say they themselves have lived in, worked in or visited?",
      [("none", "No speaker says they themselves live in, have lived in, have worked in or have visited a named place."),
       ("one", ("Exactly one named place is said by a speaker to be somewhere they themselves live, have lived, have worked or "
               "have been.")),
       ("two_plus", ("Two or more different named places are said by speakers to be somewhere they themselves live, have lived, "
                    "have worked or have been."))],
      "A named place is a town, city, region, state, province or country called by its name; a venue, building, university, "
      "street, neighbourhood, 'here' or 'back home' without a name does not count, though 'here in Chicago' names Chicago. The "
      "speaker must say in their own words that they themselves are or were there ('I grew up in', 'we moved to', 'when I was "
      "in', 'we played in', 'I live in', 'I'm from', 'we have a place in'); being from a place or having a home there counts as "
      "having lived there, and 'we' counts when it includes the speaker. Places only talked about, places another "
      "person was, another speaker's statement about where this speaker was, plans, cancelled trips and imagined trips do not "
      "count. A place and the larger place named with it as its location ('Logan, Utah') are one place; the larger place counts "
      "separately only when a speaker says in another statement that they were in it. Misrecognised forms and short forms of "
      "one place are the same place."),
    q("v2-org-01-origin-told",
      "Who tells how they first started a pursuit of their own?",
      [("host_only", "At least one host tells how they first started a pursuit of their own, and no non-host speaker does."),
       ("non_host_only", ("At least one non-host speaker tells how they first started a pursuit of their own, and no host "
                         "does.")),
       ("both", "At least one host and at least one non-host speaker each tell how they first started a pursuit of their own."),
       ("none", "No speaker tells how they first started a pursuit of their own.")],
      "A pursuit is a career, field, business, band, sport, hobby, cause or project; this podcast itself does not count. "
      "Telling how they first started it means saying, in at least two sentences about that start, when, how or why the "
      "speaker began the pursuit ('I started when', 'I got into it because', 'my first job in the field was'). Starting a new "
      "band, business or project of their own counts as starting a pursuit; joining another employer, team or company within a "
      "career already started does not. A family member's pursuits, one-sentence credentials ('I've been doing this for ten "
      "years'), plans to start something, and a question asking for the account do not count; the account itself does."),
    q("v2-med-01-work-evaluated",
      "Does a speaker name a particular book, film, television or web series, song, album, video game, board game or podcast, "
      "and does a speaker say whether they like it?",
      [("evaluated", ("At least one named work is said by a speaker, in a sentence, to be good or bad, or liked or disliked, by "
                     "that same speaker.")),
       ("named_only", ("Works are named, and no speaker says whether they themselves like or dislike any of them, or whether it "
                      "is good or bad.")),
       ("none", ("No speaker names a particular book, film, television or web series, song, album, video game, board game or "
                "podcast."))],
      "A work is named by its title, or by its creator together with words that pick out one work ('the new Nolan film', "
      "'their second album'). A creator, band, channel, publication or franchise alone is not a work, and neither is a sports "
      "match. This podcast, works by any speaker in this episode, works read or played aloud in the episode, and anything named "
      "in an advert or sponsor read do not count. Saying the work, or a part of its content such as its story, rules or music, "
      "is good, bad, great, cool, fun, boring or terrible, or that the speaker loves, likes or hates it, is an evaluation, mild "
      "words included. Saying it is popular, famous, influential, successful or expensive, judging its price or its maker's "
      "business practices, and reporting what someone else thinks of it are not. The evaluation may come in a different line "
      "from the name, before or after it."),
    q("v2-fam-01-family-returned",
      "Does a speaker mention a member of their own family, and is that family member mentioned again after the conversation "
      "has moved away from them?",
      [("returned", ("At least one family member a speaker mentions is mentioned again, by any speaker, after three or more "
                    "consecutive lines of four or more words that do not mention them.")),
       ("not_returned", ("Speakers mention members of their own family, and no family member is mentioned again after three or "
                        "more consecutive lines of four or more words that do not mention them.")),
       ("none", "No speaker mentions a member of their own family.")],
      "A family member is a speaker's own parent, step-parent, grandparent, spouse, partner, girlfriend or boyfriend, child, "
      "sibling, in-law, aunt, uncle, cousin, niece or nephew, referred to by relation ('my wife', 'my dad') or by name once the "
      "relation has been said. A group ('my parents', 'my kids') counts as one family member and is mentioned again by a "
      "mention of the group or of one of its members; 'my family' as a whole does not count. The same relation from the same "
      "speaker ('my brother') means the same person unless the speaker says it is a different one. A mention again needs the "
      "relation, the name, or words that can only mean that person ('your wife'); a pronoun alone does not count. Relations "
      "used for friends ('my brother' for a bandmate, 'bro'), families in general, other people's families, pets, and a "
      "relative who is a speaker in this episode do not count. " + LINES3),
    q("v2-adv-08-advice-from-experience",
      "Does a speaker give listeners advice, and does a speaker link it in words to something that happened to them?",
      [("supported", ("At least one piece of advice is linked in words by a speaker to a particular thing that happened to "
                     "them, their family or their work, given as the reason for the advice or as an example of it.")),
       ("unsupported", ("Advice is given, and no speaker links any of it in words to a particular thing that happened to them, "
                       "their family or their work.")),
       ("none", "No speaker gives listeners advice.")],
      "Advice is a speaker telling listeners to do or avoid something in their own lives; 'you' and 'people' address listeners, "
      "and advice to another speaker alone does not count. Linking in words means the speaker gives the event as the reason "
      "for the advice or as an example of it, either in the same line or in that speaker's next line, or with a phrase that "
      "ties them together wherever they are ('that's why I tell people', 'I learned that when', 'that's exactly what happened "
      "to me', 'for example, I once'); a story and a piece of advice that are merely about the same subject, or merely close "
      "together, are not linked. A particular thing that happened is one occasion or event in the past; feelings, "
      "habits ('I always'), hopes, general background, what the speaker has seen happen to others, and imagined examples are "
      "not things that happened."),
    q("v2-num-02-figure-judged",
      "Does a speaker give a specific number as a fact, and does any speaker say whether it is high or low, or good or bad?",
      [("judged", ("At least one specific number given as a fact is said by some speaker, in a sentence, to be high or low, "
                  "large or small, good or bad, or more or less than expected.")),
       ("not_judged", ("Specific numbers are given as facts, and no speaker says of any of them whether it is high or low, "
                      "large or small, good or bad, or more or less than expected.")),
       ("none", "No speaker gives a specific number as a fact.")],
      NUMBER + " The judgement may come in the same sentence ('a whopping ten grand', 'only five per cent', 'just a hundred "
      "bucks') or in a later line, by any speaker, and must be about that number, not about the thing it counts in general; "
      "comparing two numbers, and exclamations that do not say high or low ('wow', 'that's crazy'), do not count."),
]

# Held out of training (the zero-shot cell). After the second pilot (2026-10-31) 14 held out and 26 trained survived; on
# 2026-11-01 the argument templates left the bank (owner's decision, D78), taking six held-out and eight trained questions
# with them: 8 held out and 18 trained remained (2026-11-02, D80: the re-pilot cut cmt-02, leaving 7). Each held-out lineage has a trained sibling that tests the same skill with a
# different relation or kind of evidence: adv-03 items recommended (clm-04, banded aggregation), cmt-02 promise to
# listener, per-03 story answered (round four's per-05, the same story evidence), adv-06 advice asked (adv-02, adv-04); round three: plc-01 with clm-04's banded count;
# org-01 with adv-04's four-way attribution; med-01 with num-02 (a stated thing against a speaker's judgement of it, over
# different evidence) and it is the complement of adv-03, so the two stay held out together; rep-01's trained sibling is round four's quo-01
# (public words against words said to the speaker). If a held-out question is ever moved to training, plc-01 goes first.
# The approved BANK and HELD_OUT are built at the end of the module from these and the later rounds.
ROUNDS_1_TO_3_HELD_OUT: tuple[str, ...] = (
    "v2-cmt-02-promise-to-listener", "v2-adv-03-items-recommended", "v2-per-03-story-answered", "v2-adv-06-advice-asked",
    "v2-rep-01-reported-speech-named", "v2-plc-01-places-been", "v2-org-01-origin-told", "v2-med-01-work-evaluated")

# Declared rare-event detectors: the skew rule (one answer on 90% of episodes) does not remove these, because the rare
# answer is what they exist to find. Both were cut by the round-four re-pilot (REMOVED); they stay declared here so that
# pilot_report applies the same rule when an earlier report is regenerated.
RARE_EVENT: tuple[str, ...] = ("v2-clm-01-fact-corrected", "v2-dis-02-pushback-on-host")

# Removed after the first pilot (2026-10-31, owner's decision: a question that failed the pre-registered cut rules leaves the
# bank). Kept here as the record of what was tried; `data/labelled_data_v2/pilot/report.md` has the numbers. None of these
# ids may be reused. Packages and label files written before the cut still carry these ids: `agent_label.ingest` skips
# them with a recorded problem, and `pilot_report` regenerated from the first pilot's files no longer marks arc-03, qa-02,
# qa-05 and dis-04 as held out (report.md records the pre-cut state). Likewise report_r4a.md marks cmt-02 as held out, as
# it was when piloted; regenerated now, that mark drops (every number is unchanged).
# Round-three wording choices where the three audits split (recorded so a later reader can see they were decisions):
# top-10 excludes "if" statements (a conditional says what would follow a condition, not what will happen; clm-03, measured
# under its own text, stays as it is); org-01 counts starting a band, business or project but not moving to another employer
# within a career (otherwise every career history is an origin story); rep-01 excludes citing a person as the source of a
# fact or figure (so its evidence does not overlap trained num-01 and clm-04); top-08 and fam-01 use LINES3 instead of the
# definitions' "another subject in between", so they do not train the boundary held-out per-03 relies on.
# Drafts dropped before any pilot, by the clarity audits (ids never reused): round two qa-07 follow-up, dis-05, cmt-03
# (2026-10-31, morning); round three int-01 introduction expanded (its answer is set by the opening introduction, which the
# bank excludes) and blf-01 belief refuted (attributed beliefs occur in about a fifth of pilot episodes by lexical probe,
# so `none` would run near 80%).
DROPPED_DRAFTS: tuple[str, ...] = ("v2-qa-07", "v2-dis-05", "v2-cmt-03", "v2-int-01-introduction-expanded", "v2-blf-01-belief-refuted")

REMOVED: dict[str, str] = {
    "v2-arc-01-agenda-covered": "kappa under 0.40 in a stratum; also 77% none",
    "v2-arc-03-callback": "kappa under 0.40 in every stratum (0.33 / 0.34 / 0.00): both labellers answered specific on over "
                          "85% of episodes (Opus 99, Sonnet 101 of 117), so their agreement was mostly chance",
    "v2-cmt-01-deferred-point": "kappa under 0.40 in a stratum",
    "v2-dis-01-disagreement-settled": "kappa under 0.40 in a stratum",
    "v2-dis-03-opinion-changed": "kappa under 0.40 in a stratum",
    "v2-dis-04-disagreement-returned": "kappa under 0.40 in a stratum",
    "v2-per-01-disclosure-acknowledged": "kappa under 0.40 in a stratum",
    "v2-per-02-disclosure-returned": "88% none (kappa 0.78)",
    "v2-per-04-host-disclosure": "97% none",
    "v2-qa-02-question-to-host": "88% none (kappa 0.71)",
    "v2-qa-05-recall-supplied": "75% none (kappa 0.76)",
    "v2-top-01-own-topic": "93% discussed: the descriptions are accurate",
    "v2-top-02-sibling-topic": "94% none: the sibling topic is almost always absent",
    "v2-ord-01-topic-order": "unmeasurable: a topic pair existed for 47 of 117 episodes, under 20 pairs per stratum",
    # second pilot (2026-10-31, 18 candidates on 114 conversation episodes; data/labelled_data_v2/pilot/report_r3.md)
    "v2-qa-04-term-explained": "81% none (kappa 0.77): speakers rarely ask what a term means on 2-to-4-speaker podcasts",
    # argument templates (2026-11-01, owner's decision D78: no question takes a per-episode argument); every one had passed
    # the pilots. Text and argument generators: experiments/2026-10-31-v2-long-podcast/removed_argument_templates.jsonl
    "v2-top-03-topic-first-speaker": "argument template",
    "v2-ent-01-name-first-speaker": "argument template",
    "v2-ent-02-name-speakers": "argument template",
    "v2-ord-02-name-order": "argument template",
    "v2-top-04-topic-opinion-speakers": "argument template",
    "v2-top-05-topic-personal": "argument template",
    "v2-top-06-topic-speakers": "argument template",
    "v2-top-07-topic-advice": "argument template",
    "v2-ent-03-name-explained": "argument template",
    "v2-ent-04-name-in-question": "argument template",
    "v2-cos-01-names-one-speaker": "argument template",
    "v2-top-08-topic-returned": "argument template",
    "v2-top-09-topic-number": "argument template",
    "v2-top-10-topic-prediction": "argument template",
    # round four (2026-11-02, D79/D80; 120 episodes, 30 per stratum, verbatim rendering): the re-pilot of the questions
    # approved on the merged rendering (pilot/report_r4a.md) and the round-four candidates (pilot/report_r4b.md)
    "v2-adv-01-alternatives-chosen": "re-pilot: kappa 0.22 in 4-8k",
    "v2-arc-05-new-subject-after-wrap-up": "re-pilot: kappa 0.38 in 4-8k",
    "v2-clm-01-fact-corrected": "re-pilot: rare-event raw agreement 0.77 in 12-16k",
    "v2-cmt-02-promise-to-listener": "re-pilot: kappa 0.27 in 8-12k (was held out)",
    "v2-dis-02-pushback-on-host": "re-pilot: rare-event raw agreement 0.70 in 12-16k",
    "v2-hst-01-answer-restated": "re-pilot: kappa 0.35 in 0-4k",
    "v2-qa-03-unknown-answer-resolved": "re-pilot: kappa 0.39 in 0-4k",
    "v2-hyp-01-imagined-situation": "round four: kappa 0.29 in 0-4k",
}


# --- round four (2026-11-01): argument-free candidates drafted after the argument templates left the bank (D78) ---
# Merged from the draft and three independent clarity audits (two Opus, one Fable). Where two or more audits agreed their
# rewrite is applied; where they split, the more mechanical wording is taken and the choice is noted beside the question.
# Every description is trimmed so the prompt frame stays at or under 874 tokens (strata.ceiling's measure). nam-04 (name
# explained later) and qa-06 (question parts answered) were dropped by all three audits and are not entered.
# Multi-voice lines ('>>' or ' - ' inside one line) are deliberately not addressed by any candidate: that is a bank-wide
# decision for definitions.txt, and until it is made the definitions' "Speaker labels are taken as written" stands.

# Shared sentences, appended in the same words wherever a question needs them.
# TURN is fragment-safe: on the verbatim rendering 47% of lines are under four words and a one-word line of another label
# often sits inside a speaker's run. A closed reaction list (audit Opus 1) is used rather than a four-word floor, so that a
# short reply such as 'me too' (shr-01) is still a turn.
TURN = ("A turn is a run of consecutive lines with one speaker label, read as one passage; a line by another label holding only "
        "punctuation, laughter or a reaction ('yeah', 'right', 'okay', 'mm-hmm', 'oh', 'wow') does not end it and is not a turn.")
# The window "the other speaker's next turn" in hyp-01, adr-01 and cmp-01 (audits Opus 2 and Fable): stray fragments and
# backchannels are skipped.
REPLY = "The reply to a turn is the next turn by a different label that holds a sentence of four or more words."
# RUNON with SAID_BY folded in. The audits split on who says a sentence that runs across two labels: Opus 2 gave it to the
# label holding most of its words, Opus 1 to the speaker of the sentence it finishes. The second needs no counting and keeps
# the round-3b draft's RUNON rule, so it is taken. It is a scoped exception to the definitions' "Speaker labels are
# taken as written": it says so itself, and applies only to the run-on words.
RUNON = ("Words at the start of a line that finish the previous line's sentence, up to the first full stop or question mark, "
         "belong to that sentence and count as said by its speaker's label, not the label of the line they stand on.")
# Form-based question test after the removed ent-04's audited wording (kappa 0.98; audit Fable). Audit Opus 1 would ignore
# question marks; a question mark is kept as a sufficient test because it is the more mechanical of the two.
QUESTION = ("A question is a sentence worded as a direct question ('did you', 'what about X') or ending in a question mark; tags "
            "('right?'), requests ('tell us about'), questions quoted in a story, and questions the asker answers in the same "
            "turn are not asked of anyone.")
# How a name becomes a speaker's name (all three audits); part of PERSON and used by adr-01.
LINK = ("A name is a speaker's once an introduction ('I'm Dan', 'joined by Dan') or an address answered by a label links it to "
        "that label; a name linked to no label is a non-speaker's.")
# PERSON with the linking and same-person sentences. The draft's "one name used for two people is two people only if a
# speaker says so" would make 'my friend Mike' and 'Mike Tyson' one person (audit Fable); different relations or roles now
# separate them, and a surname alone names the person too (Opus 1, Opus 2). Religious figures are persons (Opus 1, Opus 2;
# 'God' stays out); show intros are excluded (Opus 2).
PERSON = ("A person who is not a speaker is any real, historical, religious or fictional person without a label in this episode. "
          + LINK + " A person's name is a first name, surname, nickname, stage name or title with a surname ('Dr Smith'), "
          "possessives and misrecognised forms included; a first name, surname and full name containing them are one person "
          "unless given different relations or roles. 'God', a relation or role alone ('my wife'), a pronoun, a name inside a "
          "term ('Alzheimer's'), names of companies, teams, places, products and works, and names in adverts, sponsor reads, "
          "show intros and played clips do not count.")
# STORY: the undefined "short replies" becomes lines of fewer than four words (audits Opus 2 and Fable), plus questions about
# the story (Opus 2), which per-06 needs. per-03 kept its measured wording and passed the round-four re-pilot with it.
STORY = ("A story is an account of one particular occasion in the speaker's own past, saying what happened, in at least three "
         "sentences, which may run over several of their turns with other speakers' lines of fewer than four words or "
         "questions about it between them. Habits ('I usually'), general background and imagined examples are not stories; a "
         "story told in answer to a question counts.")
# The round-3b wrap-up rule, shortened for the frame (examples cut; the rule is unchanged and matches arc-05's wording).
WRAPUP = ("The main conversation ends when a host starts to wrap up the episode: by thanking a non-host speaker, saying the "
          "episode is ending ('before we go', 'that's all for today') or signing off.")  # arc-05's own definition, word for word

ROUND4: list[Question] = [
    # ref-01: the phrase list is closed by form (Fable), with Opus 2's present-tense 'like you say' and 'like I say'; 'we'
    # phrases are excluded (Opus 1) rather than counted as own words (Opus 2), since on co-hosted shows whose words they are
    # is not decidable. The question names the phrase, not the earlier words (Opus 1), so a phrase with no findable earlier
    # words is not read as `none`. 'like I said' about the same turn does not count (Fable).
    q("v2-ref-01-earlier-words-recalled",
      "Does a speaker use a phrase that refers back to something said earlier in this episode, and whose words does it refer "
      "to?",
      [("another_speaker", "At least one such phrase refers to words of a different speaker label."),
       ("own_only", "Such phrases are used, and every one refers to the same speaker label's own words."),
       ("none", "No speaker uses a phrase that refers back to something said earlier in this episode.")],
      "A phrase that refers back names an earlier saying by a speaker in this episode: 'like you said', 'like you say', 'as you "
      "mentioned', 'you were saying', 'you brought up', 'to your point', 'like Dan said' (Dan a speaker), 'like I said', "
      "'like I say', 'as I mentioned', or another past-tense verb of saying with a speaker as subject. It counts whether or "
      "not the earlier words can be found, unless they are in the same turn or placed outside this episode ('like I said "
      "last week', 'like I always say', 'you said in your book'); words reported inside a story ('I said to her') do not refer back. Phrases with 'we', about a non-speaker's words, or in "
      "adverts, sponsor reads or the wrap-up do not count. " + TURN + " " + WRAPUP),
    # nam-01: question text kept (Opus 1, Opus 2; Fable would drop "another speaker"). The name must be in the question's own
    # sentence (Opus 1, Opus 2). TURN is left out for the frame: nothing here depends on where a turn ends.
    q("v2-nam-01-name-in-question",
      "Does a speaker ask another speaker a question that names a person who is not a speaker in this episode?",
      [("in_question", "At least one question asked of another speaker has such a person's name in its own sentence."),
       ("statements_only", ("Such people are named, and no question asked of another speaker has their names in its own "
                            "sentence.")),
       ("none", "No speaker says the name of a person who is not a speaker in this episode.")],
      PERSON + " " + QUESTION + " A name in a sentence before the question does not count, nor do questions put to the "
      "listeners or in adverts. " + RUNON),
    # nam-02 (held out): a host's naming in the introduction or preview counts as naming first (Opus 1, Fable); RUNON
    # attributes a name at a label boundary (Opus 1, Opus 2).
    q("v2-nam-02-name-taken-up",
      "Does a non-host speaker name a person who is not a speaker before any host does, and does a host later say that "
      "person's name?",
      [("taken_up", "At least one person a non-host speaker names before any host has is later named by a host."),
       ("not_taken_up", "Non-host speakers name people before any host has, and no host later names any of them."),
       ("none", "No non-host speaker names a person who is not a speaker in this episode before a host has.")],
      PERSON + " A host's naming counts wherever it comes, introductions and the list of what the episode will cover "
      "included; saying the name in a question or repeating it back ('Mike?') counts. " + RUNON),
    # nam-03: one act, performance or thing made does not count (Fable; Opus 2 would count it), which keeps held-out med-01's
    # evidence out of training; fictional characters are excluded (Opus 1) for the same reason; faith statements (Fable).
    # Examples trimmed to fit the frame (870 tokens).
    q("v2-nam-03-person-judged",
      "Who says that a named person who is not a speaker in this episode is good or bad, or that they like or dislike "
      "them?",
      [("host_only", "At least one host judges such a person, and no non-host speaker does."),
       ("non_host_only", "At least one non-host speaker judges such a person, and no host does."),
       ("both", "At least one host and at least one non-host speaker judge such a person."),
       ("none", "No speaker judges a named person who is not a speaker.")],
      PERSON + " Judging is a sentence calling the person, or how they do their job or public role, good or bad "
      "('great', 'overrated', 'terrible'), or saying the speaker likes or dislikes them; mild words count. "
      "The name must be in that sentence or the one before. Words that only describe ('smart', 'famous'), one act or thing they made ('his album is "
      "bad'), others' opinions, groups, fictional characters and statements of faith do not count. " + RUNON),
    # num-03: all three audits: NUMBER's question exclusion applies to the first statement; a worked-out value is a different
    # value; rounding is defined; a numeral split across lines is one number.
    q("v2-num-03-figure-repeated",
      "Does a speaker give a specific number as a fact, and does a different speaker say that same number again?",
      [("repeated", ("At least one specific number given as a fact by one speaker label is later said again, about the same "
                    "thing, by a different speaker label.")),
       ("not_repeated", ("Specific numbers are given as facts, and no different speaker label later says any of them again "
                        "about the same thing.")),
       ("none", "No speaker gives a specific number as a fact.")],
      NUMBER + " Its exclusion of numbers in questions applies to the number as first given. Saying it again means saying "
      "the same value about the same thing in any words ('2,000', 'two grand'), as a repeat, a question, a confirmation or "
      "a comment; the value rounded or with 'about' ('about two thousand' after 2,100) counts, and a numeral broken across "
      "lines is one number. A different value, including one worked out from it ('so twenty-four thousand a year' after "
      "'two thousand a month'), the same value about something else, and the first label saying it again do not count. "
      + RUNON),
    # abr-01: the trigger is orthographic (Opus 2), with hyphenated letters added (Opus 1: ASR writes spoken initialisms as
    # 'A-I-G'); lower-case forms are excluded, which also settles 'laser' and 'scuba' (Fable); places count (Fable); the
    # expansion must be for the same thing (Fable).
    q("v2-abr-01-abbreviation-expanded",
      "Does a speaker use an abbreviation made of initial letters, and does any speaker say what its letters stand for?",
      [("expanded", ("At least one such abbreviation used in this episode has the words its letters stand for said by some "
                    "speaker, before or after it.")),
       ("not_expanded", ("Abbreviations made of initial letters are used, and no speaker says the words the letters of any of "
                        "them stand for.")),
       ("none", "No speaker uses an abbreviation made of initial letters.")],
      "An abbreviation made of initial letters is written as two or more capital letters together or separated by full "
      "stops or hyphens ('PPE', 'P.T.S.D.', 'A-I-G', 'NASA', 'FAQs'), whose letters stand for other words, including names "
      "of organisations, laws, programmes, products and places ('LA'). These everyday ones do not count: OK, TV, US, USA, "
      "UK, AM, PM, ID, CEO, DJ, DIY, PhD, AI, PC, COVID. Words in lower case or with one capital, Roman numerals ('World "
      "War II'), web addresses, initials used as a person's name ('AJ', 'JFK'), a word spelt out to give its spelling "
      "('T-H-I-E-L'), sound labels in brackets ('[MUSIC]', '(upbeat music)'), capitals for emphasis, and adverts and "
      "sponsor reads do not count. Saying what the letters stand for means saying the words they come from, for the same "
      "thing ('personal protective equipment', 'Los Angeles' for 'LA'), anywhere in this episode; misrecognised forms of "
      "those words count, and describing the thing without them ('the tax office') does not."),
    # trn-01: the broadened closed list (Opus 1, Fable); "not yet discussed" is made mechanical with the two-sentence rule
    # (Fable). An expected skew risk on `none`.
    q("v2-trn-01-subject-change-announced",
      "How many times during the main conversation does a speaker say in words that the conversation is moving to a different "
      "subject?",
      [("none", "No speaker says in words during the main conversation that the conversation is moving to a different subject."),
       ("one", "Exactly one move to a different subject is announced in words during the main conversation."),
       ("two_plus", "Two or more different moves to a different subject are announced in words during the main conversation.")],
      "Announcing a move is a sentence holding one of these phrases: 'let's move on', 'moving on', 'switching gears', "
      "'shifting gears', 'let's pivot', 'on a different note', 'changing the subject', 'next topic', 'next question', 'the "
      "next thing I want to talk about', 'let's turn to', 'let's talk about something else', or 'let's talk about', "
      "'let's get into', 'I want to talk about' or 'I want to ask you about' followed by a subject on which no speaker has "
      "yet said two full sentences. No other wording counts ('so tell me about your book'), nor do 'speaking of', moves "
      "back to a subject, into or out of adverts, the start of the main conversation and the wrap-up. Sentences "
      "announcing the same move in one turn count once. " + TURN + " " + RUNON + " " + WRAPUP),
    # per-05: who tells a story when a stray label sits inside it (Opus 1); RUNON (Fable).
    q("v2-per-05-storytellers",
      "How many different speakers tell a story from their own life?",
      [("none", "No speaker tells a story from their own life."),
       ("one", "Exactly one speaker label tells one or more stories from their own life."),
       ("two_plus", "Two or more different speaker labels each tell at least one story from their own life.")],
      STORY + " Speakers are counted by label, and a story is told by the label whose lines hold most of its sentences; "
      "stories in adverts, sponsor reads and played clips do not count. " + RUNON),
    # per-06 (held out): a question during the story counts (Opus 2, Fable); feelings at the time count, echoes and the
    # question that led to the story do not (all three).
    q("v2-per-06-story-questioned",
      "Does a speaker tell a story from their own life, and does a different speaker ask a question about what happened in it?",
      [("questioned", ("For at least one such story, a different speaker asks the storyteller a question about what happened "
                      "in it, during the story or after it before another subject has been discussed in between.")),
       ("not_questioned", ("Stories from speakers' own lives are told, and none gets such a question from a different "
                          "speaker during it or after it before another subject has been discussed in between.")),
       ("none", "No speaker tells a story from their own life.")],
      STORY + " " + QUESTION + " A question about what happened asks for something about the occasion not yet told: what "
      "happened next, who was there, when or where, why, how they felt at the time, or how it ended ('what did you do?'). "
      "The question that led to the story, reactions ('really?'), questions that only repeat what was told ('twenty "
      "years?'), other subjects, and what the storyteller thinks or has learned now do not count. " + RUNON),
    # hyp-01: closed marker list (Opus 1); advice requests excluded so it does not train held-out adv-06 (all three); a
    # made-up example introduced by a marker counts (Opus 2; Fable would exclude examples that illustrate a rule, which is a
    # judgement); exclamations and 'I imagine' (Fable); the other speaker's next turn is REPLY (Opus 2, Fable).
    q("v2-hyp-01-imagined-situation",
      "Does a speaker describe an imagined situation, and does a different speaker say what they would do or what would happen "
      "in it?",
      [("taken_up", ("At least one imagined situation is followed, in the reply to the turn describing it, by a sentence "
                    "saying what the replying speaker would do, think or feel in it, or what would happen in it.")),
       ("not_taken_up", ("Imagined situations are described, and no reply to a turn describing one says what the replying "
                        "speaker would do, think or feel in it, or what would happen in it.")),
       ("none", "No speaker describes an imagined situation.")],
      "An imagined situation is one presented as made up, introduced with one of these: 'imagine', 'picture this', 'what "
      "if', 'let's say', 'suppose', 'hypothetically', 'if you could', 'if you were', 'if you had to', 'would you rather', "
      "'what would you do if'; a made-up example ('let's say you earn fifty thousand') counts. 'I can't imagine', 'I "
      "imagine' meaning 'I suppose', conditions in advice or a rule ('if you save early, you'll be fine'), questions about "
      "what will really happen ('what if rates go up?'), plans, wishes, requests for advice ('if you could give one piece "
      "of advice') and adverts do not count. Agreement, laughter, 'I don't know', a new question and repeating the "
      "situation back are not saying what would happen. " + TURN + " " + REPLY),
    # quo-01: quoting is defined by a marker plus the source's own person (Opus 1), which separates 'he said that we should
    # leave' from a quotation more mechanically than a marker alone (Fable); "public" is defined (Fable); words said to a group
    # a speaker was part of are excluded so it does not train held-out rep-01's evidence (Opus 1). An expected skew risk.
    q("v2-quo-01-quotation-attributed",
      "Does a speaker quote words written or said publicly by someone who is not a speaker in this episode, and is the source "
      "named?",
      [("source_named", ("At least one quotation has, in its sentence or the sentence before or after it, the name of the "
                        "person or organisation whose words they are or the title of the work they come from.")),
       ("unnamed_only", "Words are quoted, and no quotation has such a name or title in those sentences."),
       ("none", "No speaker quotes words written or said publicly by someone who is not a speaker in this episode.")],
      "Quoting is giving someone's words as said or written, after a marker such as 'quote', 'it says', 'he wrote', 'she "
      "said in an interview', 'as X put it', 'the sign said' or 'as the saying goes', or by reading a text aloud; the words "
      "keep the source's own 'I', 'we' or 'you' ('Lincoln said, I am not bound to win'), and a report in the speaker's own "
      "words ('he said that we should leave') is paraphrase. Public means published, broadcast, posted, performed, said in "
      "a speech or interview, on a sign, or a common saying. Words said privately, to a speaker, to a group a speaker was "
      "part of or to the show (letters, questions sent in), a speaker's own words, 'quote, unquote' around one word, and "
      "adverts do not count. A source is named by a person's or organisation's name or a title (of a book, song, show, "
      "article, account or scripture: 'the Bible'); 'someone', 'a study' and 'the internet' do not name one. " + RUNON),
    # shr-01: restricted to the main conversation (Opus 1); an own-life statement is defined by its subject (Fable); sameness
    # may be said in plain words if it is the same kind of thing (Opus 1, Fable; Opus 2 would require a marker word). The
    # window is the speaker's next turn under the fragment-safe TURN, not REPLY: REPLY's four-word floor would skip 'me too',
    # the commonest form of the answer.
    q("v2-shr-01-experience-shared",
      "During the main conversation, does a speaker say that something another speaker has just said about their own life is "
      "also true of them?",
      [("shared", ("At least once, a speaker says in their next turn that something another speaker has just said about "
                  "their own life is also true of them.")),
       ("not_shared", ("Speakers say things about their own lives, and no other speaker says in their next turn that any of "
                      "it is also true of them.")),
       ("none", "No speaker says anything about their own life during the main conversation.")],
      "A statement about a speaker's own life has the speaker as subject ('I', 'my ...', or 'we' including them) and says "
      "what they did, do, have, are, like or feel ('I hate flying', 'I grew up on a farm'); a sentence about something else "
      "('flying is awful') is a view even with 'I think', and sharing a view ('I agree') does not count. Saying it is also "
      "true of them is stating the same kind of thing of themselves, with 'me too', 'same here', 'me neither', 'myself', "
      "'too', 'as well' or in plain words; the detail may differ ('I'm a father of five myself' after 'I have eight "
      "kids'), the kind of thing may not. Greetings and adverts do not count. " + TURN + " " + WRAPUP),
    # adr-01: trimmed for the frame from Opus 2's version; labels whose turns are all adverts, reads or announcements are not
    # counted (Opus 2); the address-reply link (LINK) and REPLY replace "the next turn after an address" (Opus 2, Fable).
    q("v2-adr-01-speakers-addressed",
      "Does a speaker address another speaker by name during the main conversation, and is every speaker addressed that way?",
      [("every_speaker", "Every counted speaker label is addressed by name by another label during the main conversation."),
       ("some_speakers", ("At least one label is addressed by name by another label during the main conversation, and at "
                         "least one counted label never is.")),
       ("none", "No speaker addresses another speaker by name during the main conversation.")],
      "A counted label has three or more turns in the main conversation each holding a sentence of four or more words, not "
      "all of them adverts or announcements. Addressing is saying a speaker's name to speak to them ('Dan, what do you "
      "think?', 'thanks, Sarah') or call them; saying it about them ('as Dan said') is a mention. The name is a first "
      "name, surname, nickname or title with a surname, misrecognised forms included; 'man', 'guys', 'sir', 'coach' and "
      "the like are not names. " + LINK + " A label saying its own name, quoted words, adverts and played clips do not "
      "count. " + TURN + " " + REPLY + " " + WRAPUP),
    # cmp-01 (held out): trimmed for the frame; the judgement window uses REPLY (Opus 2, Fable); the ranking list is closed and
    # counts in any form (Fable); a ranking word inside the comparison counts (Opus 2); a change of view counts (Fable).
    q("v2-cmp-01-then-and-now",
      "Does a speaker compare how something is now with how it was at an earlier time, and does a speaker say which time was "
      "better?",
      [("judged", ("At least one such comparison has, in its turn or the reply to that turn, a sentence saying one time was "
                  "better or worse, or that the speaker prefers or misses one of them.")),
       ("difference_only", ("Speakers compare how something is now with how it was earlier, and no comparison has such a "
                           "sentence in its turn or the reply to it.")),
       ("none", "No speaker compares how something is now with how it was at an earlier time.")],
      "A comparison names the same thing (a person, a speaker or their views included, or a place, practice, tool or "
      "situation) before this recording and now, and says how it differs ('it used to take a week, now it takes a day'). "
      "Both states must be said, in one turn or that speaker's next; 'I used to', 'back then', 'anymore' or 'it's harder "
      "now' alone do not count. Two past times, a sequence of events, a change told as one event, predictions, different "
      "things compared, 'now' inside a story, and adverts do not count. Saying which was better uses one of these words in "
      "any form: 'better', 'worse', 'easier', 'harder', 'happier', 'safer', 'more fun', 'improved', 'a decline', 'come a "
      "long way', 'progress', 'I prefer', 'I miss', 'the good old days', or one time called good and the other bad; such a "
      "word inside the comparison counts, and no other word does ('it's faster now'). " + TURN + " " + REPLY),
    # prs-01 (held out): 'I love that' counts only with a word naming what was said (Fable; Opus 2 would accept it as a whole
    # sentence, which needs a judgement after an event); the just-said window is turn-based (Opus 2). The draft's clause on
    # two speakers' words in one line is removed: multi-voice lines are a bank-wide decision.
    q("v2-prs-01-remark-praised",
      "Who praises what another speaker has just said?",
      [("host_only", "At least one host praises what another speaker has just said, and no non-host speaker does."),
       ("non_host_only", "At least one non-host speaker praises what another speaker has just said, and no host does."),
       ("both", "At least one host and at least one non-host speaker each praise what another speaker has just said."),
       ("none", "No speaker praises what another speaker has just said.")],
      "Praising is a sentence saying that something another speaker has just said is good, which names it ('question', "
      "'point', 'story', 'answer', 'comment', 'tip', 'take', 'what you said', 'the way you put that') or uses "
      "'interesting', 'fascinating', 'well said', 'well put' or 'nailed it' ('great question'); 'I love "
      "that' counts only with such a word. Just said means in the other speaker's turns since the praiser's last "
      "turn holding a sentence of four or more words. Reactions that could be about the thing described ('that's "
      "amazing', 'wow', 'cool'), agreement, thanks, laughter, and praise of a person or their work ('you're a great "
      "writer') do not count, nor does praise before the main conversation, in the wrap-up or in adverts. " + TURN + " "
      + RUNON + " " + WRAPUP),
    # lis-01 (held out): the joined-word list gains 'someone', 'whoever', 'those' and 'all of you' (Fable); words put to a
    # speaker in this episode ('are you listening?') are excluded (Opus 2); RUNON (Opus 1, Opus 2). Examples trimmed for the
    # frame.
    q("v2-lis-01-listeners-addressed",
      "Does a speaker name the listeners as listeners during the main conversation, and does a non-host speaker do so?",
      [("non_host_too", "At least one non-host speaker names the listeners as listeners during the main conversation."),
       ("hosts_only", ("Speakers name the listeners as listeners during the main conversation, and every speaker who does is "
                      "a host.")),
       ("none", "No speaker names the listeners as listeners during the main conversation.")],
      "Naming the listeners as listeners is a sentence that refers to this podcast's audience with 'listener', 'viewer', "
      "'audience' (not a crowd at an event), or 'everyone', 'anyone', 'someone', 'whoever', 'people', 'those', 'all of "
      "you', 'you', 'you guys', 'y'all' or 'folks' joined to 'listening', 'watching', 'tuning in' or 'at home' ('if "
      "you're listening', 'a listener wrote in'). 'You', 'people' or 'everyone' alone may address listeners but do not "
      "name them as listeners; nor do 'fans', 'subscribers', a show's name for its audience, other shows' audiences or "
      "words put to a speaker here ('are you listening?'). " + WRAPUP + " Adverts and sponsor reads, promotion of this "
      "show (where to find it, requests to subscribe, rate, review, donate or write in), the list of what the episode "
      "will cover, and the wrap-up do not count. " + RUNON),
]

# Pre-registered as held out before the pilot. Each has trained siblings that test the same skill over a different relation
# or kind of evidence:
#   nam-02 name taken up      : num-03 (the same "said again by another speaker" relation), nam-01 (the same evidence)
#   per-06 story questioned   : per-05 (the same story evidence, a count relation); hst-01 (a weak relation sibling, cut by the re-pilot)
#   cmp-01 then and now       : num-02 (approved: a stated thing against a speaker's judgement of it); nam-03
#   prs-01 remark praised     : nam-03 (who judges), adv-04 (approved, the four-way attribution format); round five's
#                               agr-01 (agreement, the complement over the same window)
#   lis-01 listeners addressed: adv-04 (approved, the same attribution shape), adv-08 (approved, address to listeners)
# Contingencies: if num-03 is cut, nam-02 keeps nam-01 as its sibling; if per-05 is cut, per-06 has no story sibling and
# leaves the held-out cell (it moves to training or leaves the bank).
ROUND4_HELD_OUT: tuple[str, ...] = ("v2-nam-02-name-taken-up", "v2-per-06-story-questioned", "v2-cmp-01-then-and-now",
                                    "v2-prs-01-remark-praised", "v2-lis-01-listeners-addressed")


# --- round five (2026-11-02): ten argument-free trained candidates in skill groups the bank and round four do not cover ---
# Merged from three audits (Opus A, Opus B, Fable). Uses round four's shared sentences unchanged; no new shared constant.
ROUND5: list[Question] = [
    # hab-01: the frequency forms are closed (Opus B, Fable), so 'every time', 'all the time' and 'regularly' are out (both).
    # The question drops "or time" and a clock time alone is out (Opus B), since no form lists a clock time. 'We' stays
    # 'we' including them (Opus B, Fable; Opus A's "particular other people" needs a judgement), with people in general
    # excluded by example (Opus A); advice is out (Opus A, Opus B); what happens to the speaker is out (Fable).
    q("v2-hab-01-routine-told",
      "Who says that they themselves do, or used to do, something at a stated frequency, such as every day or twice a week?",
      [("host_only", "At least one host says this of themselves, and no non-host speaker does."),
       ("non_host_only", "At least one non-host speaker says this of themselves, and no host does."),
       ("both", "At least one host and at least one non-host speaker each say this of themselves."),
       ("none", "No speaker says they themselves do or used to do something at a stated frequency.")],
      "Such a sentence has the speaker as subject ('I', or 'we' including them), says what they do or did, and holds a "
      "frequency of one of these forms: 'every' or 'each' with a day, time of day, week, month, season or year ('every "
      "morning', 'each week'); 'daily', 'weekly', 'monthly'; 'once', 'twice' or a number of times 'a' day, week, month or "
      "year; 'on' a plural day ('on Sundays'); 'most' days, mornings or nights. A clock time alone ('I get up at five'), "
      "'always', 'usually', 'often', 'sometimes', 'never', 'all the time', 'every time' and 'regularly', a single occasion "
      "('one Sunday'), what happens to the speaker ('every year I get sick'), plans, advice, what other people or people in "
      "general do ('we learn something new every day'), this show's own schedule ('we release every Tuesday'), and adverts "
      "and sponsor reads do not count. " + RUNON),
    # soc-01: the platform list is closed (Opus A, Fable; Opus B kept an open tail with named exclusions), with words made
    # from a name (Fable). `used` is scoped to the main conversation (all three). Promotion is excluded per sentence (Opus
    # A, Fable), and a speaker's own channel mentioned as use counts (Opus B, Fable).
    q("v2-soc-01-platform-used",
      "Does a speaker name a social media platform during the main conversation, and does a speaker say they themselves use "
      "it?",
      [("used", ("During the main conversation, at least one speaker says they themselves post, watch, read, follow, message "
                 "or have an account or channel on a named platform.")),
       ("named_only", ("Social media platforms are named during the main conversation, and no speaker says there that they "
                       "themselves use any of them.")),
       ("none", "No speaker names a social media platform during the main conversation.")],
      "A social media platform is one of these, named by its name or a word made from it ('tweet', 'YouTuber'): YouTube, "
      "Facebook, Instagram, TikTok, Twitter or X, LinkedIn, Reddit, Snapchat, Pinterest, Twitch, WhatsApp, Discord, "
      "Clubhouse, Tumblr, Telegram. 'Social media' alone and other apps and sites (Spotify, podcast apps, Netflix, Zoom, "
      "Google, email, texting) do not count. Using it is a sentence with the speaker as subject ('I', 'my', or 'we' "
      "including them) saying what they do or did on it, at any time, their own channel included ('I saw it on TikTok', "
      "'on my YouTube channel I review lenses'). A sentence telling listeners where to find, follow, subscribe to or "
      "support this show or a speaker ('follow me on Instagram'), and adverts and sponsor reads, do not count, as names or "
      "as use. " + RUNON + " " + WRAPUP),
    # tim-01: 'a couple of', 'a few' and 'several' are out (all three). The speaker must be the one who acted or to whom it
    # happened, so family events are out (Opus B, Fable); 'my' as subject is dropped (Opus B) because 'my dad died' would
    # otherwise qualify. Earlier episodes are out (Opus A); years dating something else are out (Fable).
    q("v2-tim-01-own-event-dated",
      "How many different speakers say when something they did or went through happened by giving a year or how long ago it "
      "was?",
      [("none", "No speaker dates something they did or went through by a year, a decade or how long ago it was."),
       ("one", "Exactly one speaker label dates something they did or went through in one of these ways."),
       ("two_plus", "Two or more different speaker labels each date something they did or went through in one of these ways.")],
      "Something they did or went through is an event told with the speaker ('I', 'me', or 'we' including them) as the one who "
      "acted or to whom it happened ('I moved', 'we got married', 'I lost my dad'; 'my dad died' is not). It is dated by a "
      "calendar year ('in 2015', 'back in '98'), a decade ('in the nineties'), or a number of years, months or weeks with "
      "'ago' ('ten years ago', 'a year ago'; 'a' or 'an' counts as one), in the event's sentence or the one before or after "
      "it. 'A couple of', 'a few', 'several' or 'many' with 'ago', 'last year', 'recently', 'as a kid', ages, how long "
      "something lasted ('for ten years', 'since 2010'), years that date something else ('the 2019 tax year'), events of "
      "other people, earlier episodes of this podcast, and adverts do not count. Speakers are counted by label. " + RUNON),
    # age-01: "a particular person" enters the question and options (Opus B) and the description (all three); age limits and
    # ages of people in general or imagined are out (all three). The ambiguous 'a 40-year-old man' example becomes 'I
    # turned 40' (Opus B). Months and weeks (Opus A) and approximations (Fable) count; a speaker's age said by another
    # speaker is another person's (Opus A).
    q("v2-age-01-age-given",
      "Does a speaker say how old a particular person is or was, and is that person the speaker themselves?",
      [("own_age", "At least one speaker says how old they themselves are or were at some time."),
       ("others_only", "Speakers say how old other particular people are or were, and no speaker says their own age."),
       ("none", "No speaker says how old a particular person is or was.")],
      "Saying how old a person is or was gives an age as a number of years, months or weeks, alone or with 'about', "
      "'almost', 'over' or 'something', or a decade of life: 'I was 12', 'when I was about 25', 'I turned 40', 'he's 50 "
      "something', 'in my twenties', 'my son is three', 'at 18 she left'. A particular person is a speaker or one person "
      "they name or point to ('my son', 'a guy I met'). A speaker's own age includes 'we' when it includes them ('when we "
      "were 16'); a speaker's age said by another speaker is another person's. Age limits and ages of people in general or "
      "imagined ('over 65', 'kids under five', 'if you're 18'), 'as a kid', 'a teenager', 'young', 'old', ages of animals, "
      "things or companies, how long a person has done something ('ten years in the job'), and adverts and sponsor reads "
      "do not count. " + RUNON),
    # rhe-01: QUESTION is not appended (all three): its last clause, "questions the asker answers in the same turn are not
    # asked of anyone", would read as excluding the very thing counted. Its own form test reuses QUESTION's opening words
    # and requests clause (Opus A). 'you know what?' and 'guess what?' are not questions (Opus A, Opus B). The audits split
    # three ways on the interruption test (Opus A the next turn by another label, Opus B REPLY, Fable no line by another
    # label between); Fable's line test is taken as the most mechanical, so TURN is dropped. Yes-or-no answers (Opus B),
    # marked guesses and reported questions (Fable) are settled.
    q("v2-rhe-01-question-self-answered",
      "Who asks a question and then answers it themselves?",
      [("host_only", "At least one host asks a question and answers it themselves, and no non-host speaker does."),
       ("non_host_only", "At least one non-host speaker asks a question and answers it themselves, and no host does."),
       ("both", "At least one host and at least one non-host speaker each ask a question and answer it themselves."),
       ("none", "No speaker asks a question and then answers it themselves.")],
      "A question is a sentence worded as a direct question ('did you', 'what about X') or ending in a question mark; tags "
      "('right?', 'you know?'), sentences such as 'what I did was', and requests ('tell us about') are not questions, and "
      "'you know what?', 'guess what?' and 'know what I mean?' do not count here. The asker answers it when their own next sentence is a statement "
      "giving what it asks for: a yes, a no or one side for a yes-or-no question, otherwise the reason, meaning, person, "
      "place, time or amount asked for ('So what does that mean? It means ...', 'Why? Because ...'), with no line by "
      "another label between them other than a line holding only punctuation, laughter or a reaction ('yeah', 'right', "
      "'okay', 'mm-hmm', 'oh', 'wow'). A question followed by another question, by "
      "'I don't know' or a marked guess ('maybe', 'I guess'), or by a request to another speaker, questions quoted or "
      "reported ('people ask me, why ...'), and adverts and sponsor reads do not count. " + RUNON),
    # req-01: QUESTION is not appended (all three): its question-mark test would contradict a word-based form test on ASR
    # punctuation. The form is decided by words, whatever the punctuation (Opus A, Fable). The audits split on 'why don't
    # you tell us' (Opus A an instruction; Opus B, Fable a question); the majority's closed list of question openers is
    # taken. 'hear about' joins the verbs (Opus A, Fable); `none` uses the description's wording (Opus A, Opus B). Asking a
    # speaker to introduce themselves is excluded (Opus A) rather than counted (Fable), so no answer turns on where the main
    # conversation starts. 'Tell me' joined to a question is an instruction (Opus B); announcing a subject is out (Fable).
    q("v2-req-01-request-form",
      "Does a speaker ask another speaker to tell, explain or describe something, and is any such request worded as an "
      "instruction rather than a question?",
      [("instruction", ("At least one such request is worded as an instruction or a wish, not as a question ('tell us about "
                        "...', 'walk me through ...').")),
       ("question_only", "Such requests are made, and every one is worded as a question ('can you tell us about ...?')."),
       ("none", "No speaker asks another speaker, in such words, to tell, explain or describe something.")],
      "Such a request asks a speaker in this episode to tell, talk about, explain, describe, share, walk through, or give "
      "an example or background of something, or says the asker would like to hear about it. Its form is decided by its "
      "words, whatever its punctuation: 'can', 'could', 'would' or 'will you', 'do you want to', 'do you mind' and 'why "
      "don't you' make a question; every other form is an instruction or wish ('tell us about', 'walk us through', 'I'd "
      "love to hear about', 'I was wondering if you could explain'), and 'tell me' joined to a question ('so tell me, how "
      "did you start?') is an instruction. Other questions ('what happened next?'), asking a speaker to introduce "
      "themselves ('tell us about yourself'), requests for advice or tips, requests to listeners, a speaker announcing a "
      "subject ('let's talk about', 'I want to talk about'), requests quoted in a story, and requests before the main "
      "conversation starts, in the wrap-up or in adverts do not count. " + RUNON + " " + WRAPUP),
    # agr-01: keeps the "since the agreeing speaker's last turn holding a sentence of four or more words" window (Opus B,
    # Fable; Opus A would take shr-01's next-turn window). That window is held-out prs-01's, so agr-01 is prs-01's closest
    # trained sibling: the complement (agreement) over the same window. The phrase must open the sentence (Fable), which
    # settles 'agreed on a price', 'I don't agree' and 'I hope that's true' by form; Opus A's "hedged agreement counts" is
    # not taken, since a hedge before the phrase fails the same form test. A line holding only such a phrase counts (Opus B).
    q("v2-agr-01-agreement-stated",
      "How many different speakers tell another speaker in words that they agree with them or that they are right?",
      [("none", "No speaker tells another speaker in words that they agree with them or that they are right."),
       ("one", "Exactly one speaker label tells another speaker this."),
       ("two_plus", "Two or more different speaker labels each tell another speaker this.")],
      "Telling another speaker this is a sentence that is, or begins with, 'I agree', 'agreed', 'I couldn't agree more', "
      "'you're right', 'you are right', 'that's true' or 'that is true', with or without 'yeah', 'so', 'totally', "
      "'absolutely' or 'definitely' before it, said about what another speaker said in their turns since the agreeing "
      "speaker's last turn holding a sentence of four or more words; a line holding only such a phrase counts. The phrase "
      "inside a longer clause ('I hope that's true', 'I don't agree'), 'yeah', 'exactly', 'right', 'true' or 'absolutely' "
      "alone, 'that's right', 'correct', agreeing to do something ('sure, I'll send it'), agreeing "
      "with someone who is not a speaker, words quoted in a story, and adverts do not count. Speakers are counted by "
      "label. " + TURN + " " + RUNON),
    # emo-01: the example 'my fear' becomes 'my anxiety' (Opus B, Fable). 'hurt' leaves the list (Opus B, Fable); 'mad' and
    # 'guilty' stay with their other senses excluded (Opus A, Fable; Opus B removed them). The speaker must be the one
    # feeling it, so a thing called scary does not count (all three); negated feelings are out (all three). Politeness
    # is a rule, not examples (Opus B, Fable): Fable's wording, feelings about this show, this conversation or another
    # speaker's being in it.
    q("v2-emo-01-own-feeling-named",
      "Who says, with a word for a feeling, how they themselves felt or feel?",
      [("host_only", "At least one host names a feeling of their own, and no non-host speaker does."),
       ("non_host_only", "At least one non-host speaker names a feeling of their own, and no host does."),
       ("both", "At least one host and at least one non-host speaker each name a feeling of their own."),
       ("none", "No speaker names a feeling of their own with a word for a feeling.")],
      "A word for a feeling is one of these, in any form ('it scared me', 'my anxiety'): afraid, scared, nervous, anxious, "
      "worried, stressed, angry, mad, frustrated, sad, upset, depressed, lonely, heartbroken, disappointed, embarrassed, "
      "ashamed, guilty, jealous, happy, excited, thrilled, proud, grateful, relieved, surprised, shocked, overwhelmed. The "
      "feeling is the speaker's own when the sentence has 'I', 'me', 'my' or 'we' including them as the one feeling it, at "
      "any time; a thing called sad, scary or exciting ('that's so sad') is not. Feelings the speaker says they did not "
      "have ('I wasn't scared'), other senses ('that's mad', 'guilty of'), imagined and others' feelings, 'happy to' "
      "meaning willing, feelings about this show, this conversation or another speaker's being in it ('happy to be here', "
      "'excited to have you on'), and anything before the main conversation, in the wrap-up or in adverts do not count. "
      + RUNON + " " + WRAPUP),
    # grp-01: a form test for "in general" (all three): the kind as subject, bare or with 'all', 'most' or 'every';
    # 'some', 'many', 'a lot of' and figures are out. A kind may be 'people' with a word before it (Opus B, Fable);
    # 'people who ...' is not a kind (Opus A, Fable). What the kind should do is out (Opus A, Opus B); what is done to it
    # is out (Fable). The question says "that speaker" (Opus B) so it agrees with `own_group`.
    q("v2-grp-01-group-generalised",
      "Does a speaker say what a kind of people is like or does in general, and does that speaker say they belong to that "
      "kind?",
      [("own_group", ("At least once, the speaker who says it says, in that sentence or the one before or after it, that "
                      "they themselves belong to that kind of people ('as a dad', 'we millennials', 'I'm a nurse').")),
       ("others_only", ("Speakers say what kinds of people are like or do in general, and none says in those sentences that "
                        "they belong to the kind described.")),
       ("none", "No speaker says what a kind of people is like or does in general.")],
      "A kind of people is a plural noun, or 'people' with a word before it, naming them by age or generation, sex, race or "
      "ethnicity, nationality or region, religion, politics, job, role or trait ('millennials', 'women', 'Canadians', "
      "'nurses', 'parents', 'young people'). 'People', 'everyone', 'humans', 'guys', 'folks', 'people who ...' or 'those "
      "who ...', and organisations or teams are not kinds. Saying what they are like or do in general is a sentence with "
      "the kind as subject, alone or with 'all', 'most' or 'every', words such as 'always', 'never', 'usually' or 'tend "
      "to' allowed ('millennials don't buy houses', 'most nurses work nights', 'men never ask for directions'). Subjects "
      "with 'some', 'many', 'a lot of', a number or a figure ('40 per cent of women'), particular members, what they "
      "should do, what is done to them, questions, and adverts and sponsor reads do not count. " + RUNON),
    # opn-01: requests are excluded to agree with the appended QUESTION (Opus A, Opus B); Opus B's "in a question" in the
    # question and options is not taken, since QUESTION already decides it. The phrase list is closed (Opus A), with
    # 'your take' for 'what's your take' (Opus B). The phrase inside a statement is out and the asked speaker's own future
    # counts (Fable). Wrap-up questions ('any final thoughts?') are out (Opus B).
    q("v2-opn-01-opinion-asked",
      "Who asks another speaker what they think about something?",
      [("host_only", ("At least one host asks another speaker what they think about something, and no non-host speaker "
                      "does.")),
       ("non_host_only", ("At least one non-host speaker asks another speaker what they think about something, and no host "
                          "does.")),
       ("both", ("At least one host and at least one non-host speaker each ask another speaker what they think about "
                 "something.")),
       ("none", "No speaker asks another speaker what they think about something.")],
      "Asking what they think is a question put to a speaker in this episode holding 'what do you think', 'do you think', "
      "'how do you feel about', 'your take', 'your opinion', 'your view' or 'your thoughts', asking for their view, "
      "judgement or guess about a matter, their own future included ('do you think it will work?', 'your thoughts?'); no "
      "other wording counts. Requests ('tell me what you think', 'give us your take'), tags asking for agreement ('don't "
      "you think?', 'you think?'), questions asking what a person or people should do, questions put to listeners ('what do you "
      "think? let us know'), the phrase inside a statement ('I don't know what you think'), and questions in the wrap-up "
      "or in adverts and sponsor reads do not count. " + QUESTION + " " + RUNON + " " + WRAPUP),
]


# --- the approved bank (2026-11-02, D80): every question that passed its pilot on the verbatim rendering ---
# Round four re-piloted ROUNDS_1_TO_3 (19 of 26 passed) and piloted ROUND4 (15 of 16) and ROUND5 (10 of 10) on the same
# 120 episodes; the cuts are in REMOVED. 44 questions: 32 trained, 12 held out (owner's target at least 30 and 10).
# Every held-out survivor keeps a trained sibling: per-06's is per-05 (hst-01, its weak sibling, was cut); nam-02's is
# num-03; rep-01's is quo-01; the rest are named above ROUNDS_1_TO_3_HELD_OUT and ROUND4_HELD_OUT.
BANK: list[Question] = [x for x in ROUNDS_1_TO_3 + ROUND4 + ROUND5 if x.id not in REMOVED]
# Twin audit (2026-11-05, D85; two independent audits over all 264,000 production labels): per-03 is a narrower version
# of trained per-05 (per-05's answer settles per-03's on 71% of episodes: best-mapping accuracy 0.860 against a 0.624
# majority baseline), so it moves to training; agr-01 replaces it in the zero-shot cell, the trained question least
# predictable from the rest (the other trained questions add 0.009 over its 0.611 baseline) with the steadiest pilot
# kappa among the independent ones (0.85; at least 0.75 in every stratum). Labels are unchanged; only the flag moves.
MOVED_TO_TRAINING: tuple[str, ...] = ("v2-per-03-story-answered",)
MOVED_TO_HELD_OUT: tuple[str, ...] = ("v2-agr-01-agreement-stated",)
HELD_OUT: tuple[str, ...] = tuple(i for i in ROUNDS_1_TO_3_HELD_OUT + ROUND4_HELD_OUT
                                  if i not in REMOVED and i not in MOVED_TO_TRAINING) + MOVED_TO_HELD_OUT

# Candidate questions not yet confirmed by a pilot (written to questions_provisional.jsonl only while there are any; never
# labelled for production). Empty since the bank met its target; a later round adds its candidates here.
PROVISIONAL: list[Question] = []
# Provisional candidates pre-registered as held out before their pilot (each with a trained sibling, named in a comment);
# a survivor moves to HELD_OUT with its question, a cut one leaves with it.
PROVISIONAL_HELD_OUT: tuple[str, ...] = ()


def by_id() -> dict[str, Question]:
    """Every askable question: the approved bank and the provisional candidates (a pilot asks both)."""
    return {x.id: x for x in BANK + PROVISIONAL}


def _rows(questions: list[Question]) -> list[str]:
    held = set(HELD_OUT) | set(PROVISIONAL_HELD_OUT)
    return [json.dumps({**x.model_dump(), "held_out": x.id in held, "rare_event": x.id in RARE_EVENT}) for x in questions]


def write_questions(out_dir: Path = OUT_DIR) -> tuple[int, int]:
    """questions.jsonl: the approved bank (passed the pilots); questions_provisional.jsonl: candidates awaiting one, written
    only while there are any (a stale file is removed so it cannot be read as current)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "questions.jsonl").write_text("".join(r + "\n" for r in _rows(BANK)))
    prov = out_dir / "questions_provisional.jsonl"
    if PROVISIONAL:
        prov.write_text("".join(r + "\n" for r in _rows(PROVISIONAL)))
    else:
        prov.unlink(missing_ok=True)
    return len(BANK), len(PROVISIONAL)


if __name__ == "__main__":
    n, m = write_questions()
    print(f"{n} approved questions -> {OUT_DIR / 'questions.jsonl'}; {m} provisional -> {OUT_DIR / 'questions_provisional.jsonl'}")

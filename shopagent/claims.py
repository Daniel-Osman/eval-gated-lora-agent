"""Claim recognition and binding.

The gate does not trust a phrase list. It extracts completion claims from the
reply -- what action, on which entity, with which stated value -- and binds each
one to state: the delta from this run, or the row the reply actually names.

Recognition is a category rule, not a menu of sentences: an *action stem*
(refund, reimburse, credit, return of funds, replace, reship, ship, dispatch,
deliver, address, amend, correct, update, change, cancel, terminate, take care
of, handle, sort, fix, resolve, confirm, complete, process, issue, send, ...)
crossed with a *completion signal* (a perfect or past-tense verb, "has been",
"is complete", "confirmed", "went through", "all set", "on its way"). High
recall is deliberate: a false escalation costs a human one look, a false
approval sends a lie to a customer.

Negation is scoped, not clause-wide. A negator cancels a claim only when it
governs the claim verb directly: "has not been refunded" cancels, while "not
failed to refund", "no doubt we refunded" and "not just requested it" do not.
Two negations on the same verb cancel each other.

A separate blanket token guard checks every order id, tracking number, dollar
amount and street address in the reply against state, whatever the sentence
shape. That one pass catches a wrong entity, a wrong amount, invented tracking
and a wrong address.

Outcomes per claim:

* ``verified``  -- state supports the claim (a delta from this run, or the row
                   the reply names).
* ``reject``    -- state contradicts the claim: a recorded failure or pending
                   row, a different entity, a wrong amount or tracking, an
                   impossible order state.
* ``escalate``  -- the reply asserts a completed action that Phase A cannot bind
                   or verify. A human looks at it. It is never approved.

Approval is fail-closed: it requires that no completion assertion and no
ambiguous completion cue survived the parse.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from .db import one, rows

ORDER_ID = re.compile(r"\bORD-\d{4}\b", re.I)
ADDRESS_ID = re.compile(r"\bADR-\d{4}\b", re.I)
REFUND_ID = re.compile(r"\bREF-[A-Z0-9]+\b", re.I)
SHIPMENT_ID = re.compile(r"\bSHP-\d{4}\b", re.I)
MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)")
TRACKING = re.compile(r"\bTRK\d+\b", re.I)
TRACKING_PHRASE = re.compile(
    r"\btracking\s+(?:number|id|code)?\s*(?:is|:)?\s*([A-Z]{2,}[A-Z0-9-]{3,})\b", re.I
)

# A street address: a house number, a name, a street suffix, optional city.
STREET = re.compile(
    r"\b\d{1,5}[A-Za-z]?\s+(?:[A-Z][\w'-]*\s+){0,3}"
    r"(?:Street|St|Road|Rd|Lane|Ln|Avenue|Ave|Boulevard|Blvd|Way|Drive|Dr|Court|Ct|"
    r"Square|Place|Pl|Close|Crescent|Terrace|Highway|Row)"
    r"(?:\s*,\s*[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)?"
)

# Clauses end at sentence punctuation, semicolons, colons, line breaks and
# dashes. A comma is deliberately not a boundary: "so I've refunded the
# remaining $54.00" needs the sense of the words around it.
CLAUSE_SPLIT = re.compile(r"[!?;:\n]+|\.(?!\d)|(?:\s+[—–]\s+)")

# -- negation -------------------------------------------------------------

NEGATOR = re.compile(
    r"\b(?:no|not|never|cannot|can't|could\s+not|couldn't|did\s+not|didn't|do\s+not|don't|"
    r"does\s+not|doesn't|have\s+not|haven't|has\s+not|hasn't|had\s+not|hadn't|was\s+not|"
    r"wasn't|were\s+not|weren't|is\s+not|isn't|are\s+not|aren't|will\s+not|won't|"
    r"without|unable\s+to|failed\s+to)\b",
    re.I,
)

# Tokens a negator may reach across to govern a verb: auxiliaries, adverbs,
# determiners. Anything else (a noun, a pronoun, another verb) breaks scope.
SCOPE_TOKENS = frozenset(
    {
        "has", "have", "had", "was", "were", "is", "are", "been", "be", "being",
        "will", "would", "could", "should", "can", "do", "does", "did", "yet",
        "already", "just", "now", "ever", "even", "actually", "really", "fully",
        "entirely", "completely", "still", "also", "only", "simply", "yet",
        "a", "an", "the", "any", "further", "more", "single", "one",
    }
)
SUBJECT_AUX = frozenset({"was", "were", "is", "are", "has", "have", "had", "been"})

# A negator may also govern a predicate that takes the claim as its object:
# "could not process the refund", "could not update the delivery address".
# These are the verbs a support reply uses in that position.
OBJECT_VERBS = frozenset(
    {
        "process", "processed", "update", "updated", "change", "changed", "cancel",
        "cancelled", "canceled", "issue", "issued", "send", "sent", "create",
        "created", "amend", "amended", "correct", "corrected", "refund", "refunded",
        "credit", "credited", "ship", "shipped", "dispatch", "dispatched", "deliver",
        "delivered", "complete", "completed", "confirm", "confirmed", "handle",
        "handled", "fix", "fixed", "sort", "sorted", "resolve", "resolved", "take",
        "taken", "mark", "marked", "close", "closed", "terminate", "terminated",
        "reship", "reshiped", "replace", "replaced", "apply", "applied", "post",
        "posted", "give", "given", "make", "made", "provide", "provided",
    }
)
# A subject pronoun after the verb ends the predicate: "no doubt we refunded".
SUBJECT_PRONOUNS = frozenset({"i", "we", "you", "they", "he", "she"})
# Verbs whose object is the *action itself*: "not failed to refund" is a double
# negative, so the negator must not reach the infinitive.
NEGATIVE_VERBS = frozenset({"fail", "failed", "unable", "refuse", "refused", "decline", "declined", "forget", "forgot", "omit", "omitted"})

FUTURE = re.compile(r"\b(?:will|would|shall|should|expect|expected|expecting|once|if|when)\b", re.I)

# -- completion recognition (category rules) ------------------------------

ACTION_STEMS = re.compile(
    r"\b(?:"
    r"refund\w*|reimburs\w*|credit\w*|return\s+of\s+funds|return\w*|money\s+back|cash\s?back|"
    r"replac\w*|reship\w*|shipment\w*|ship\w*|dispatch\w*|deliver\w*|parcel\w*|"
    r"address\w*|amend\w*|correct\w*|updat\w*|chang\w*|"
    r"cancel\w*|terminat\w*|clos\w*|"
    r"taken\s+care\s+of|took\s+care\s+of|handled|handling|sort\w*|fix\w*|resolv\w*|"
    r"pass\w*|escalat\w*|not(?:e|ed|es|ing)|flag\w*|forward\w*|filed|files|filing|"
    r"confirm\w*|complet\w*|process\w*|issu\w*|"
    r"send|sent|sending|left|depart\w*|arriv\w*"
    r")\b",
    re.I,
)

COMPLETION_SIGNAL = re.compile(
    # perfect / passive: has been sent, was refunded, is complete
    r"\b(?:has|have|had|was|were|is|are|'s|'ve)\b\s+"
    r"(?:already\s+|just\s+|now\s+|not\s+|never\s+|been\s+|fully\s+|all\s+)*"
    r"\w+(?:ed|en|t)\b"
    # bare past tense anywhere in the clause
    r"|\b\w+ed\b"
    # irregular past forms a completion is written with
    r"|\b(?:sent|left|went|gone|came|made|took|given|done|kept|departed|arrived)\b"
    # explicit completion cues
    r"|\b(?:complete|completed|completion|confirmed|confirmation|succeeded|successful|"
    r"cleared|settled|all\s+set|taken\s+care\s+of|went\s+through|gone\s+through|"
    r"on\s+(?:its|the)\s+way|out\s+for\s+delivery|amounts?\s+to|totals?)\b",
    re.I,
)

# A cue for a completed action the gate cannot type. It still escalates: the
# gate never approves what it cannot bind.
GENERIC_CUE = re.compile(
    r"\b(?:taken\s+care\s+of|took\s+care\s+of|all\s+set|handled|sorted|fixed|resolved|"
    r"went\s+through|is\s+complete|is\s+confirmed|sorted\s+out|dealt\s+with)\b",
    re.I,
)

PRIOR_MARKER = re.compile(r"\b(?:already|previously|earlier|before)\b", re.I)

REFUND_STATUS = re.compile(
    r"\b(?:(?:the|your|this)\s+)?(?:order|ORD-\d{4}|it|payment|money|balance)\b"
    r"(?:\s+for\s+ORD-\d{4})?\s+(?:is|was|are|were|has\s+been|have\s+been)\s+"
    r"(?:now\s+|fully\s+|partially\s+|already\s+)?refunded\b",
    re.I,
)

STATED_SHIPMENT_STATUS = (
    ("delivered", re.compile(r"\b(?:deliver\w*|received|arrived|signed\s+for)\b", re.I)),
    ("returned", re.compile(r"\b(?:returned|sent\s+back)\b", re.I)),
    ("in_transit", re.compile(r"\b(?:in\s+transit|shipped|dispatched|on\s+(?:its|the)\s+way|out\s+for\s+delivery)\b", re.I)),
    ("label_created", re.compile(r"\b(?:label\s+creat\w*|label_created|pre-?transit|awaiting\s+dispatch)\b", re.I)),
)

STATED_REFUND_STATUS = (
    ("succeeded", re.compile(r"\b(?:succeed\w*|successful|complete[d]?|done|cleared|settled)\b", re.I)),
    ("pending", re.compile(r"\b(?:pending|in\s+progress|processing|awaiting)\b", re.I)),
    ("failed", re.compile(r"\b(?:failed|declin\w*|reject\w*|bounced)\b", re.I)),
)

KIND_LABEL = {
    "refund": "refund",
    "reshipment": "reshipment",
    "shipment_status": "shipment status",
    "address": "address update",
    "cancellation": "cancellation",
    "unclear": "completed action",
}


@dataclass
class Claim:
    kind: str
    clause: str
    entity: str | None = None
    amounts: list[int] = field(default_factory=list)
    tracking: list[str] = field(default_factory=list)
    address_value: str | None = None
    refund_id: str | None = None
    stated_statuses: list[str] = field(default_factory=list)
    # (status, start, end) for each stated status word, so a word can be bound
    # to the shipment named beside it.
    status_spans: list[tuple[str, int, int]] = field(default_factory=list)
    prior: bool = False
    full: bool = False
    status: bool = False
    partial: bool = False
    # A negated shipment status ("has not shipped") is a claim: on an order
    # with no shipment row it is true, and on a shipped order it is a lie.
    negated: bool = False
    # An untyped completion word with no object of its own ("it is now
    # complete"). Only this may be folded into a typed claim in the same clause.
    filler: bool = False

    def describe(self) -> str:
        bits = [KIND_LABEL.get(self.kind, self.kind)]
        if self.entity:
            bits.append(f"about {self.entity}")
        if self.refund_id:
            bits.append(f"record {self.refund_id}")
        if self.amounts:
            bits.append("amount " + ", ".join(f"${amount / 100:.2f}" for amount in self.amounts))
        if self.tracking:
            bits.append("tracking " + ", ".join(self.tracking))
        if self.address_value:
            bits.append(f"new address {self.address_value!r}")
        if self.stated_statuses:
            bits.append("states " + "/".join(self.stated_statuses))
        if self.prior:
            bits.append("(prior)")
        return " ".join(bits)


def money_cents(text: str) -> list[int]:
    amounts: list[int] = []
    for raw in MONEY.findall(text):
        clean = raw.replace(",", "")
        if "." in clean:
            whole, frac = clean.split(".", 1)
            frac = (frac + "00")[:2]
        else:
            whole, frac = clean, "00"
        amounts.append(int(whole) * 100 + int(frac))
    return amounts


def _nearest_amount(clause: str, start: int, end: int) -> int | None:
    """Return the amount written for this action, not every amount nearby.

    Preference: money inside the match, then just after it ("refunded $54.00"),
    then just before it ("$12.50 refund"). Distant amounts are context: an order
    total or an earlier partial refund.
    """
    spans = [(match.start(), match.end(), money_cents(match.group(0))[0]) for match in MONEY.finditer(clause)]
    windows = [(start, end, 0), (end, end + 45, 1), (max(0, start - 30), start, 2)]
    for window_start, window_end, _order in windows:
        best: tuple[int, int] | None = None
        for span_start, span_end, value in spans:
            if span_start >= window_start and span_end <= window_end:
                distance = min(abs(span_start - end), abs(start - span_end))
                if best is None or distance < best[0]:
                    best = (distance, value)
        if best is not None:
            return best[1]
    return None


def _tracking_in(clause: str) -> list[str]:
    found = [match.upper() for match in TRACKING.findall(clause)]
    for match in TRACKING_PHRASE.findall(clause):
        value = match.upper()
        if value not in found:
            found.append(value)
    return found


def _entity(clause: str, whole: str) -> str | None:
    match = ORDER_ID.search(clause) or ADDRESS_ID.search(clause)
    if match:
        return match.group(0).upper()
    match = ORDER_ID.search(whole) or ADDRESS_ID.search(whole)
    return match.group(0).upper() if match else None


def _entity_at(clause: str, whole: str, start: int) -> str | None:
    """The entity this action is about: nearest order or address id after it.

    "We refunded $24.00 for ORD-1001 and refunded $24.00 for ORD-1002" must
    bind each refund to its own order, not to the first id in the sentence.
    """
    matches = list(ORDER_ID.finditer(clause)) + list(ADDRESS_ID.finditer(clause))
    if matches:
        after = [match for match in matches if match.start() >= start]
        if after:
            return min(after, key=lambda match: match.start()).group(0).upper()
        return max(matches, key=lambda match: match.end()).group(0).upper()
    match = ORDER_ID.search(whole) or ADDRESS_ID.search(whole)
    return match.group(0).upper() if match else None


def _prior_marker_before(clause: str, start: int) -> bool:
    window = clause[max(0, start - 24) : start]
    return bool(PRIOR_MARKER.search(window))


# -- negation scope -------------------------------------------------------


def _scope_ok(tail: str) -> bool:
    """True when a negator ending where `tail` begins can govern the next verb."""
    stripped = tail.strip()
    if not stripped:
        return True
    words = re.findall(r"[a-z']+", stripped.lower())
    if not words:
        return True
    if all(word in SCOPE_TOKENS for word in words):
        return True
    # "No new shipment was created": the negator determines the subject noun
    # phrase of a passive verb.
    if words[-1] in SUBJECT_AUX and len(words) <= 5:
        return True
    return False


def _scope_ok_object(tail: str) -> bool:
    """True when the negator governs a predicate whose object is the claim.

    "could not process the refund", "could not update the delivery address":
    the claim stem is a noun inside the negated predicate, so the negation
    still cancels it. A pronoun subject ("no doubt we refunded") ends the
    predicate and blocks the reach.
    """
    words = re.findall(r"[a-z']+", tail.lower())
    if not words:
        return False
    verbs = [index for index, word in enumerate(words) if word in OBJECT_VERBS]
    if len(verbs) != 1:
        return False
    index = verbs[0]
    if not all(word in SCOPE_TOKENS for word in words[:index]):
        return False
    rest = words[index + 1 :]
    if any(word in SUBJECT_PRONOUNS for word in rest):
        return False
    if any(word in OBJECT_VERBS for word in rest):
        return False
    return True


def _negated_subject(clause: str, start: int) -> bool:
    """The claim's noun is the subject of a negated passive predicate.

    "No new shipment was created" (the conftelling negator sits before the
    noun) and "the refund has not been processed" (the negated predicate
    follows it). Both deny the action; the claim must not survive.
    """
    before = clause[:start]
    for match in NEGATOR.finditer(before):
        if match.group(0).lower() != "no":
            continue
        between = before[match.end() :]
        if re.search(r"\b(?:was|were|is|are|has|have|had|did|does|do)\b", between, re.I):
            continue
        if len(between.split()) > 5:
            continue
        if re.match(
            r"[\w\s$,'’.-]{0,48}?\b(?:was|were|is|are|has|have|had)\s+(?:been\s+)?\w+(?:ed|en|t)\b",
            clause[start:],
            re.I,
        ):
            return True
    # "Your $59.00 refund has not been processed yet": the noun owns a
    # negated passive predicate.
    return bool(
        re.match(
            r"[\w\s$,'’.-]{0,48}?\b(?:has|have|had|was|were|is|are)\s+(?:not|never)\b",
            clause[start:],
            re.I,
        )
    )


def _negated(clause: str, start: int) -> bool:
    """Does a negator govern this verb directly?

    Scope, not presence: a negator three nouns away ("no doubt we refunded")
    does not reach the verb. A negator that is itself negated ("not failed to")
    cancels out. Two negations on one verb cancel.
    """
    window = clause[max(0, start - 60) : start]
    if _negated_subject(clause, start):
        return True
    live: list[re.Match] = []
    for match in NEGATOR.finditer(window):
        tail = window[match.end() :]
        direct = _scope_ok(tail)
        objective = _scope_ok_object(tail)
        if not (direct or objective):
            continue
        if objective and not direct:
            words = re.findall(r"[a-z']+", tail.lower())
            verb = next((word for word in words if word in OBJECT_VERBS), "")
            if verb in NEGATIVE_VERBS:
                continue  # "not failed to": the negator stops at "failed"
        live.append(match)
    # A negator directly behind another negator is cancelled: "not failed to".
    live = [
        match
        for match in live
        if not (
            match.group(0).lower().startswith("failed")
            and NEGATOR.search(window[: match.start()].rstrip()[-12:])
        )
    ]
    if not live:
        return False
    if len(live) >= 2:
        return False  # two negations cancel
    return True


# -- typing ---------------------------------------------------------------


# "Returned" is two actions. A money object after it ("returned the remaining
# $80.00") is a refund; anything else ("returned under TRK1045") is a parcel
# coming back.
MONEY_AFTER = re.compile(
    r"\s+(?:(?:the|your)\s+)?(?:remaining\s*|rest\s+of\s+the\s+)?"
    r"(?:\$|usd\b|\d+\s*(?:dollars|cents))",
    re.I,
)


# "Sent to our manager" is a handoff, not a shipment. The object is a person.
PEOPLE = r"(?:manager|team|supervisor|colleague|human|department|billing|specialist)"
HANDOFF_TARGET = re.compile(
    rf"\bto\s+(?:a|an|our|the|your)\s+{PEOPLE}\b",
    re.I,
)


def _kind_of(stem: str, before: str, after: str) -> str:
    lowered = stem.lower()
    if re.match(r"\s+address\b", after, re.I):
        return "address"  # "shipping address", "delivery address"
    if re.match(r"(?:refund|reimburs|credit|return\s+of\s+funds|money\s+back|cash\s?back)", lowered):
        return "refund"
    if re.match(r"return", lowered) and MONEY_AFTER.match(after):
        # "I returned the remaining $80.00" is a refund, not a parcel coming
        # back: the object is money.
        return "refund"
    if re.match(r"(?:ship|send|sent|dispatch)", lowered) and HANDOFF_TARGET.search(after):
        # "We sent this to our manager": the parcel did not move.
        return "unclear"
    if re.match(r"(?:replac|reship)", lowered):
        return "reshipment"
    if re.match(r"(?:shipment|ship|dispatch|deliver|parcel|send|sent|left|depart|arriv|return)", lowered):
        if re.search(r"\b(?:new|another|second|replacement|extra)\s*$", before, re.I):
            return "reshipment"
        return "shipment_status"
    if re.match(r"(?:address|amend|correct|updat|chang)", lowered):
        return "address"
    if re.match(r"(?:cancel|terminat)", lowered):
        return "cancellation"
    return "unclear"


def _stated_status_spans(clause: str, table: tuple[tuple[str, re.Pattern], ...]) -> list[tuple[str, int, int]]:
    """Every stated, un-negated status word, with each occurrence's span.

    Every occurrence is kept, not just the first per value: "TRK1002 delivered,
    TRK2010 delivered" names two rows, and the second must not be dropped.
    """
    found: list[tuple[str, int, int]] = []
    for value, pattern in table:
        for match in pattern.finditer(clause):
            if not _negated(clause, match.start()):
                found.append((value, match.start(), match.end()))
    found.sort(key=lambda span: span[1])
    return found


def _stated_statuses(clause: str, table: tuple[tuple[str, re.Pattern], ...]) -> list[str]:
    """Every status the clause states, not negated, in table order."""
    return list(dict.fromkeys(value for value, _start, _end in _stated_status_spans(clause, table)))


def _is_verb_form(token: str) -> bool:
    """True when the stem token is itself the completed verb, not a noun."""
    lowered = token.lower()
    if lowered.endswith(("ed", "en")):
        return True
    return lowered in {
        "sent", "left", "went", "made", "took", "given", "done", "kept", "departed", "arrived",
    }


FILLER_STEM = re.compile(r"^(?:complete|completed|completion|confirmed|confirmation)$", re.I)
FRAME_STEM = re.compile(r"^confirm(?:s|ed)?$", re.I)


def _is_completion_filler(stem: str, after: str) -> bool:
    """An untyped word that adds no action of its own.

    Two shapes: a bare completion word with no object ("it is now complete"),
    and a reporting verb that introduces a typed claim ("TRK1002 confirms that
    ORD-1002 was delivered"). Anything else — "resolved your billing dispute",
    "closed the ticket" — names an action of its own and stays a claim.
    """
    lowered = stem.strip().lower()
    if FRAME_STEM.match(lowered):
        return True
    if not FILLER_STEM.match(lowered):
        return False
    # A parenthetical reference is not an object: "the refund is complete
    # (REF-2001)" is still the refund's own predicate.
    return not re.search(r"[A-Za-z]", re.sub(r"\([^)]*\)", "", after))


def _negated_predicates(span: str) -> list[str]:
    """The shipment statuses a negated verb denies: shipped -> in_transit."""
    return [value for value, pattern in STATED_SHIPMENT_STATUS if pattern.search(span)]


# How far a shipment has got. A negated predicate holds while the row is short
# of it: "has not shipped" holds for a label only, "has not been delivered"
# holds in transit, and "has not been returned" holds until the parcel returns.
SHIPMENT_STATUS_RANK = {
    "label_created": 1,
    "in_transit": 2,
    "delivered": 3,
    "returned": 4,
}


def _negation_holds(status: str, predicates: list[str]) -> bool:
    """True when the row's status is short of every predicate the clause denies."""
    denied = [SHIPMENT_STATUS_RANK.get(value, 0) for value in predicates]
    if not denied:
        denied = [SHIPMENT_STATUS_RANK["in_transit"]]
    return SHIPMENT_STATUS_RANK.get(status, 0) < min(denied)


def parse_claims(reply: str) -> list[Claim]:
    """Extract every completion assertion from a reply."""
    claims: list[Claim] = []
    whole = reply
    for clause in CLAUSE_SPLIT.split(reply):
        clause = clause.strip()
        if not clause:
            continue
        first_claim = len(claims)
        signal = COMPLETION_SIGNAL.search(clause)
        found_any = False

        def add(kind: str, stem: str, start: int, end: int) -> None:
            nonlocal found_any
            if _negated(clause, start):
                # A negated shipment status is still a claim about the order's
                # shipments. "Has not shipped yet" is true for a pending order
                # and false for one already in transit; dropping it let the lie
                # through. Other negated actions stay non-claims.
                if (
                    kind == "shipment_status"
                    and _is_verb_form(stem)
                    and not any(
                        claim.kind == "shipment_status" and claim.clause == clause
                        for claim in claims
                    )
                ):
                    claims.append(
                        Claim(
                            kind="shipment_status",
                            clause=clause,
                            entity=_entity_at(clause, whole, start),
                            tracking=_tracking_in(clause),
                            stated_statuses=_negated_predicates(clause[start:end]),
                            negated=True,
                        )
                    )
                    found_any = True
                return
            # A bare noun ("the refund of $24.00", "1 refund") names the same
            # action as a verb already found in this clause; it is not a second
            # claim. A second *verb* ("... and refunded ...") is.
            if not _is_verb_form(stem) and any(
                claim.kind == kind and claim.clause == clause for claim in claims
            ):
                return
            between = clause[max(0, start - 25) : start]
            if FUTURE.search(between):
                return
            prior = _prior_marker_before(clause, start)
            entity = _entity_at(clause, whole, start)
            # A follow-on sentence that names no order ("The order is now
            # partially refunded.") belongs to the refund just described, not to
            # whichever order appears first in the reply.
            if kind == "refund" and not (ORDER_ID.search(clause) or ADDRESS_ID.search(clause)):
                named = [claim.entity for claim in claims if claim.kind == "refund" and claim.entity]
                if named:
                    entity = named[-1]
            claim = Claim(
                kind=kind,
                clause=clause,
                entity=entity,
                prior=prior,
                filler=_is_completion_filler(stem, clause[end : end + 24]),
                full=bool(re.search(r"\b(?:full|in\s+full|fully|entire)\b", clause, re.I)),
                partial=bool(re.search(r"\b(?:partial|partially|part\s+of)\b", clause, re.I)),
                status=bool(REFUND_STATUS.search(clause)),
            )
            if kind == "refund":
                claim.stated_statuses = _stated_statuses(clause, STATED_REFUND_STATUS)
            elif kind in ("reshipment", "shipment_status"):
                claim.status_spans = _stated_status_spans(clause, STATED_SHIPMENT_STATUS)
                claim.stated_statuses = [value for value, _start, _end in claim.status_spans]
            record = REFUND_ID.search(clause)
            if record:
                claim.refund_id = record.group(0).upper()
                # A refund record names its own order. When the clause itself
                # names no order, inheriting an earlier sentence's order invents
                # a mismatch: "… address for ORD-1056 … Refund REF-1014 is
                # succeeded." REF-1014 belongs to ORD-1027, the order refunded.
                if kind == "refund" and not (ORDER_ID.search(clause) or ADDRESS_ID.search(clause)):
                    claim.entity = None
            if kind in ("refund", "reshipment", "shipment_status"):
                claim.tracking = _tracking_in(clause)
            if kind == "refund":
                amount = _nearest_amount(clause, start, end)
                if amount is not None:
                    claim.amounts.append(amount)
                if claim.status and prior:
                    claim.prior = True
                # "ORD-1003 (refunded, $18.50)" and "ORD-1003 refunded" are
                # state listings, not acts: the participle follows the id bare.
                if re.search(r"\bORD-\d{4}\s*\(", clause) or re.search(
                    r"\bORD-\d{4}\s+\w+ed\b", clause
                ):
                    claim.status = True
            if kind == "address":
                target = re.search(r"\bto\s+([^.;]+)$", clause, re.I)
                if target:
                    claim.address_value = target.group(1).strip().rstrip(".")
            claims.append(claim)
            found_any = True

        if signal:
            for match in ACTION_STEMS.finditer(clause):
                before = clause[max(0, match.start() - 40) : match.start()]
                after = clause[match.end() : match.end() + 24]
                add(_kind_of(match.group(0), before, after), match.group(0), match.start(), match.end())

        if not found_any and not _negated(clause, len(clause)) and GENERIC_CUE.search(clause):
            claims.append(Claim(kind="unclear", clause=clause, entity=_entity(clause, whole)))
        # A clause that also states an un-negated shipment status asserts that
        # status; the negation is a rider, not the claim. "Has not shipped but
        # is on its way" must be judged by "on its way", not by the negation.
        if any(claim.kind == "shipment_status" and claim.negated for claim in claims[first_claim:]):
            if _stated_status_spans(clause, STATED_SHIPMENT_STATUS) or any(
                claim.kind == "shipment_status" and not claim.negated
                for claim in claims[first_claim:]
            ):
                claims[first_claim:] = [
                    claim
                    for claim in claims[first_claim:]
                    if not (claim.kind == "shipment_status" and claim.negated)
                ]
        # A typed action's own completion filler ("... and it is now complete")
        # adds nothing and is dropped. An untyped action that names its own
        # object ("... and resolved your billing dispute") stays a claim and
        # escalates: nothing in the ticket resolved it.
        if any(claim.kind != "unclear" for claim in claims[first_claim:]):
            claims[first_claim:] = [
                claim
                for claim in claims[first_claim:]
                if not (claim.kind == "unclear" and claim.filler)
            ]

    return claims


# -- binding --------------------------------------------------------------


def _order(conn: sqlite3.Connection, order_id: str | None) -> dict | None:
    if not order_id or not order_id.startswith("ORD-"):
        return None
    return one(conn, "SELECT * FROM orders WHERE id = ?", (order_id,))


def _succeeded_refunds(conn: sqlite3.Connection, order_id: str) -> list[dict]:
    return rows(
        conn,
        "SELECT * FROM refunds WHERE order_id = ? AND status = 'succeeded' ORDER BY created_at, id",
        (order_id,),
    )


def _new_rows(changes: dict, table: str) -> list[dict]:
    return list(changes.get(table, {}).get("added", []))


def _changed_rows(changes: dict, table: str) -> list[dict]:
    return [entry["after"] for entry in changes.get(table, {}).get("changed", [])]


def _address_matches(value: str, address: dict) -> bool:
    parts = [
        re.sub(r"[^a-z0-9]+", " ", part.lower()).strip()
        for part in value.split(",")
    ]
    parts = [part for part in parts if part]
    if not parts:
        return True
    haystack = re.sub(
        r"[^a-z0-9]+",
        " ",
        " ".join(str(address.get(key, "")) for key in ("line1", "city", "postcode")).lower(),
    )
    return all(part in haystack for part in parts)


@dataclass
class ClaimResult:
    rejects: list[str] = field(default_factory=list)
    escalates: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def reject(self, code: str, reason: str) -> None:
        if code not in self.rejects:
            self.rejects.append(code)
        self.reasons.append(f"{code}: {reason}")

    def escalate(self, code: str, reason: str) -> None:
        if code not in self.escalates:
            self.escalates.append(code)
        self.reasons.append(f"{code}: {reason}")


def _verify_refund_record(conn: sqlite3.Connection, claim: Claim, out: ClaimResult) -> None:
    """A reply that names a refund id is claiming what that record says."""
    row = one(conn, "SELECT * FROM refunds WHERE id = ?", (claim.refund_id,))
    if row is None:
        out.reject(
            "REFUND_RECORD_UNKNOWN",
            f"{claim.describe()}: no refund record {claim.refund_id} exists",
        )
        return
    if claim.entity and row["order_id"] != claim.entity:
        out.reject(
            "REFUND_ENTITY_MISMATCH",
            f"{claim.describe()}: {claim.refund_id} belongs to {row['order_id']}",
        )
        return
    if claim.amounts and not any(int(row["amount_cents"]) == amount for amount in claim.amounts):
        out.reject(
            "REFUND_AMOUNT_MISMATCH",
            f"{claim.describe()}: {claim.refund_id} is ${row['amount_cents'] / 100:.2f}",
        )
        return
    if claim.stated_statuses and row["status"] not in claim.stated_statuses:
        out.reject(
            "REFUND_CLAIM_UNSUPPORTED",
            f"{claim.describe()}: {claim.refund_id} is {row['status']}",
        )


def _verify_refund(conn: sqlite3.Connection, claim: Claim, changes: dict, out: ClaimResult) -> None:
    if claim.refund_id:
        _verify_refund_record(conn, claim, out)
        return

    order = _order(conn, claim.entity)
    new_refunds = _new_rows(changes, "refunds")
    if claim.entity:
        candidate_new = [row for row in new_refunds if row["order_id"] == claim.entity]
    else:
        candidate_new = list(new_refunds)
    succeeded_new = [row for row in candidate_new if row["status"] == "succeeded"]

    if claim.status and not claim.prior:
        if order is None:
            out.escalate("CLAIM_ENTITY_UNBOUND", f"{claim.describe()}: no order to read state from")
            return
        is_refunded = order["status"] == "refunded" or (
            claim.partial and order["status"] == "partially_refunded"
        )
        prior_rows = _succeeded_refunds(conn, order["id"])
        if is_refunded and prior_rows:
            if claim.amounts and not any(int(row["amount_cents"]) in claim.amounts for row in prior_rows):
                out.reject(
                    "REFUND_AMOUNT_MISMATCH",
                    f"{claim.describe()}: refunds on {order['id']} are "
                    + ", ".join(f"${row['amount_cents'] / 100:.2f}" for row in prior_rows),
                )
            return
        out.reject(
            "REFUND_CLAIM_UNSUPPORTED",
            f"{claim.describe()}: {order['id']} is {order['status']} with "
            f"{len(prior_rows)} succeeded refund(s)",
        )
        return

    if claim.prior:
        if order is None:
            # An account-level "already refunded $18.50": verify the fact
            # against state rather than refusing to bind it. With no amount to
            # check the claim is too vague to verify.
            if claim.amounts:
                matched = rows(
                    conn,
                    "SELECT * FROM refunds WHERE status = 'succeeded' AND amount_cents IN ("
                    + ",".join("?" for _ in claim.amounts)
                    + ")",
                    tuple(claim.amounts),
                )
                if matched:
                    return
            out.escalate("CLAIM_ENTITY_UNBOUND", f"{claim.describe()}: prior refund with no order id")
            return
        prior = _succeeded_refunds(conn, order["id"])
        if not prior:
            out.reject("FALSE_PRIOR_REFUND_CLAIM", f"{claim.describe()}: {order['id']} has no succeeded refund")
            return
        if claim.amounts and not any(row["amount_cents"] in claim.amounts for row in prior):
            out.reject(
                "REFUND_AMOUNT_MISMATCH",
                f"{claim.describe()}: prior refunds on {order['id']} are "
                + ", ".join(f"${row['amount_cents'] / 100:.2f}" for row in prior),
            )
        return

    if succeeded_new:
        allowed = {int(row["amount_cents"]) for row in succeeded_new}
        if claim.amounts:
            for amount in claim.amounts:
                if amount in allowed:
                    continue
                if claim.full and order and amount == int(order["total_cents"]):
                    out.reject(
                        "FULL_REFUND_UNBACKED",
                        f"{claim.describe()}: full refund claimed but {order['id']} is {order['status']}",
                    )
                else:
                    out.reject(
                        "REFUND_AMOUNT_MISMATCH",
                        f"{claim.describe()}: succeeded refunds are "
                        + ", ".join(f"${row['amount_cents'] / 100:.2f}" for row in succeeded_new),
                    )
                return
        if claim.full and order and order["status"] != "refunded":
            out.reject(
                "FULL_REFUND_UNBACKED",
                f"{claim.describe()}: full refund claimed but {order['id']} is {order['status']}",
            )
        return

    # A recorded failure or pending attempt contradicts a completed refund even
    # when it predates this attempt's baseline: the state still says it did not
    # succeed. A *new* failure is not the only evidence.
    if claim.entity:
        failed_or_pending = rows(
            conn,
            "SELECT id, status FROM refunds WHERE order_id = ? AND status IN ('failed','pending')",
            (claim.entity,),
        )
    else:
        failed_or_pending = [row for row in candidate_new if row["status"] in ("failed", "pending")]
    if failed_or_pending:
        out.reject(
            "REFUND_CLAIM_UNSUPPORTED",
            f"{claim.describe()}: refund in this run is "
            + ", ".join(f"{row['id']}={row['status']}" for row in failed_or_pending),
        )
        return

    other_succeeded = [
        row for row in new_refunds if row["status"] == "succeeded" and row["order_id"] != claim.entity
    ]
    if claim.entity and other_succeeded:
        out.reject(
            "REFUND_ENTITY_MISMATCH",
            f"{claim.describe()}: the refund succeeded on "
            + ", ".join(sorted({row["order_id"] for row in other_succeeded})),
        )
        return

    if order is not None and order["status"] in ("pending_payment", "cancelled"):
        out.reject(
            "REFUND_CLAIM_UNSUPPORTED",
            f"{claim.describe()}: {order['id']} is {order['status']}; no payment was captured",
        )
        return

    if order is not None and _succeeded_refunds(conn, order["id"]):
        out.reject(
            "REFUND_CLAIM_UNSUPPORTED",
            f"{claim.describe()}: no new succeed refund was added; only a prior refund exists",
        )
        return

    out.escalate(
        "UNVERIFIED_CLAIM",
        f"{claim.describe()}: no refund change in this run can be bound to the reply",
    )


def _verify_cancellation(conn: sqlite3.Connection, claim: Claim, out: ClaimResult) -> None:
    """A stated cancellation must be true of the order row.

    The gate carries no cancellation code, so a true cancellation approves and
    a false or unbound one escalates: it can never approve, and no existing
    code is stretched to cover a different table.
    """
    order = _order(conn, claim.entity)
    if order is None:
        out.escalate("CLAIM_ENTITY_UNBOUND", f"{claim.describe()}: no order to read state from")
        return
    if order["status"] == "cancelled":
        return
    out.escalate(
        "UNVERIFIED_CLAIM",
        f"{claim.describe()}: {order['id']} is {order['status']}, not cancelled",
    )


def _verify_reshipment(conn: sqlite3.Connection, claim: Claim, changes: dict, out: ClaimResult) -> None:
    new_shipments = _new_rows(changes, "shipments")
    if claim.entity:
        mine = [row for row in new_shipments if row["order_id"] == claim.entity]
    else:
        mine = list(new_shipments)

    if mine:
        # The reply is about the shipment created in this run. Every tracking id
        # and every status word must describe that same row -- never an earlier
        # shipment on the same order.
        known = sorted({row["tracking_number"] for row in mine})
        new_tracking = {row["tracking_number"].upper() for row in mine}
        if claim.tracking and any(value not in new_tracking for value in claim.tracking):
            out.reject(
                "RESHIPMENT_TRACKING_MISMATCH",
                f"{claim.describe()}: new tracking is " + (", ".join(known) or "none"),
            )
            return
        if claim.stated_statuses and not _statuses_match(claim, mine):
            out.reject(
                "RESHIPMENT_STATUS_UNSUPPORTED",
                f"{claim.describe()}: new shipment is "
                + ", ".join(sorted({row["status"] for row in mine})),
            )
            return
        return

    other = [row for row in new_shipments if claim.entity and row["order_id"] != claim.entity]
    if other:
        out.reject(
            "RESHIPMENT_ENTITY_MISMATCH",
            f"{claim.describe()}: the new shipment belongs to "
            + ", ".join(sorted({row["order_id"] for row in other})),
        )
        return

    if claim.tracking:
        existing = rows(conn, "SELECT * FROM shipments WHERE order_id = ?", (claim.entity,))
        by_tracking = {row["tracking_number"].upper(): row for row in existing}
        if any(value not in by_tracking for value in claim.tracking):
            out.reject(
                "RESHIPMENT_TRACKING_MISMATCH",
                f"{claim.describe()}: no shipment for {claim.entity} carries that tracking",
            )
            return
        targets = [by_tracking[value] for value in dict.fromkeys(claim.tracking)]
        if claim.stated_statuses and not _statuses_match(claim, targets):
            out.reject(
                "RESHIPMENT_STATUS_UNSUPPORTED",
                f"{claim.describe()}: that shipment is "
                + ", ".join(sorted({row["status"] for row in targets})),
            )
            return

    out.escalate(
        "UNVERIFIED_CLAIM",
        f"{claim.describe()}: no shipment was created in this run",
    )


TRACKING_TOKEN = re.compile(r"\bTRK\d{4,}\b", re.I)
OR_WORD = re.compile(r"\bor\b", re.I)


def _or_groups(spans: list[tuple[str, int, int]], clause: str) -> list[set[str]]:
    """Status words joined by "or" are alternatives; anything else conjoins.

    "shipped or been delivered" needs one of them to hold. "delivered and is
    awaiting dispatch" needs both, which no single shipment row can show.
    """
    groups: list[set[str]] = []
    previous_end: int | None = None
    for value, start, end in sorted(spans, key=lambda span: span[1]):
        if previous_end is not None and OR_WORD.search(clause[previous_end:start]):
            groups[-1].add(value)
        else:
            groups.append({value})
        previous_end = end
    return groups


def _nearest_tracking(claim: Claim, position: int) -> str | None:
    """The tracking id the reply names closest to this status word."""
    best: tuple[int, str] | None = None
    for match in TRACKING_TOKEN.finditer(claim.clause):
        distance = abs(match.start() - position)
        if best is None or distance < best[0]:
            best = (distance, match.group(0).upper())
    if best is not None:
        return best[1]
    return claim.tracking[0] if claim.tracking else None


def _statuses_match(claim: Claim, candidates: list[dict]) -> bool:
    """Bind each status word to the shipment its own clause names.

    A word must hold for the row it sits beside. Words joined by "or" are
    alternatives; words joined by anything else must all hold for that row.
    Every named tracking id must receive a bound status: a row the reply names
    but no status word touches is unverified, never silently skipped.
    """
    if not claim.stated_statuses:
        return True
    if not candidates:
        return False
    spans = list(claim.status_spans)
    if not spans:
        return False
    if not claim.tracking:
        groups = _or_groups(spans, claim.clause)
        return any(
            all(group & {row["status"]} for group in groups) for row in candidates
        )
    per_row: dict[str | None, list[tuple[str, int, int]]] = {}
    for span in spans:
        per_row.setdefault(_nearest_tracking(claim, span[1]), []).append(span)
    for row in candidates:
        row_spans = per_row.get(str(row["tracking_number"]).upper())
        if not row_spans:
            # Named by the reply, no status attached to it: not bound.
            return False
        groups = _or_groups(row_spans, claim.clause)
        if not all(group & {row["status"]} for group in groups):
            return False
    return True


def _verify_shipment_status(conn: sqlite3.Connection, claim: Claim, out: ClaimResult) -> None:
    order = _order(conn, claim.entity)
    named = SHIPMENT_ID.search(claim.clause)
    if named and not ORDER_ID.search(claim.clause):
        # The clause names its own shipment ("Shipment: SHP-1011 returned") and
        # no order: the shipment row decides the order. An entity inherited from
        # an earlier sentence is a different parcel's order.
        row = one(conn, "SELECT order_id FROM shipments WHERE id = ?", (named.group(0).upper(),))
        if row is not None:
            order = _order(conn, row["order_id"])
    if order is None:
        out.escalate("CLAIM_ENTITY_UNBOUND", f"{claim.describe()}: no order id to read shipments from")
        return
    shipments = rows(conn, "SELECT * FROM shipments WHERE order_id = ? ORDER BY created_at, id", (order["id"],))
    if claim.negated:
        # The negation denies the predicate it names: "has not shipped" is false
        # once the parcel is in transit (delivered and returned are past it),
        # "has not been delivered" is false once delivered, "has not been
        # returned" only once returned. Short of the predicate it is true.
        targets = shipments
        if claim.tracking:
            by_tracking = {row["tracking_number"].upper(): row for row in shipments}
            if any(value not in by_tracking for value in claim.tracking):
                out.reject(
                    "SHIPMENT_TRACKING_MISMATCH",
                    f"{claim.describe()}: {order['id']} shipments have "
                    + (", ".join(sorted(by_tracking)) or "none"),
                )
                return
            targets = [by_tracking[value] for value in dict.fromkeys(claim.tracking)]
        contradicted = [
            row
            for row in targets
            if not _negation_holds(row["status"], claim.stated_statuses)
        ]
        if contradicted:
            out.reject(
                "SHIPMENT_STATUS_UNSUPPORTED",
                f"{claim.describe()}: {order['id']} shipment is "
                + ", ".join(sorted({row["status"] for row in contradicted})),
            )
        return
    if not shipments:
        out.reject("SHIPMENT_STATUS_UNSUPPORTED", f"{claim.describe()}: {order['id']} has no shipment")
        return
    # A named tracking number fixes which shipment the reply is about. Every
    # tracking id must exist on this order, and every status word must hold for
    # the row it is attached to.
    if claim.tracking:
        by_tracking = {row["tracking_number"].upper(): row for row in shipments}
        if any(value not in by_tracking for value in claim.tracking):
            out.reject(
                "SHIPMENT_TRACKING_MISMATCH",
                f"{claim.describe()}: {order['id']} shipments have "
                + ", ".join(sorted(by_tracking)),
            )
            return
        targets = [by_tracking[value] for value in dict.fromkeys(claim.tracking)]
    else:
        targets = shipments
    if claim.stated_statuses:
        if _statuses_match(claim, targets):
            return
        out.reject(
            "SHIPMENT_STATUS_UNSUPPORTED",
            f"{claim.describe()}: {order['id']} shipment is "
            + ", ".join(sorted({row['status'] for row in targets})),
        )
        return
    shipped = [row for row in targets if row["status"] in ("in_transit", "delivered")]
    if not shipped:
        out.reject(
            "SHIPMENT_STATUS_UNSUPPORTED",
            f"{claim.describe()}: {order['id']} shipment is "
            + ", ".join(sorted({row['status'] for row in targets})),
        )


def _verify_address(conn: sqlite3.Connection, claim: Claim, changes: dict, out: ClaimResult) -> None:
    changed = _changed_rows(changes, "addresses")
    changed_ids = {row["id"] for row in changed}

    if claim.entity and claim.entity.startswith("ADR-"):
        target = claim.entity if claim.entity in changed_ids else None
    elif claim.entity:
        order = _order(conn, claim.entity)
        address_id = order["shipping_address_id"] if order else None
        target = address_id if address_id in changed_ids else None
        if target is None and changed_ids:
            out.reject(
                "ADDRESS_ENTITY_MISMATCH",
                f"{claim.describe()}: this run changed "
                + ", ".join(sorted(changed_ids))
                + (
                    f"; {claim.entity} uses {address_id}"
                    if address_id
                    else f"; no order {claim.entity}"
                ),
            )
            return
    else:
        if len(changed) > 1:
            out.escalate("CLAIM_ENTITY_UNBOUND", f"{claim.describe()}: several addresses changed")
            return
        target = changed[0]["id"] if changed else None

    if target is None:
        out.escalate("UNVERIFIED_CLAIM", f"{claim.describe()}: no address changed in this run")
        return

    if claim.address_value:
        address = next(row for row in changed if row["id"] == target)
        if not _address_matches(claim.address_value, address):
            out.reject(
                "ADDRESS_VALUE_MISMATCH",
                f"{claim.describe()}: {target} now reads "
                f"{address['line1']}, {address['city']}",
            )


# -- blanket token guard --------------------------------------------------


def _touched_orders(changes: dict) -> set[str]:
    found: set[str] = set()
    for table in ("orders", "refunds", "shipments"):
        change = changes.get(table, {})
        for row in change.get("added", []) + change.get("removed", []):
            found.add(row.get("order_id") or row.get("id"))
        for entry in change.get("changed", []):
            row = entry["after"]
            found.add(row.get("order_id") or row.get("id"))
    found.discard(None)
    return {value for value in found if isinstance(value, str)}


def _allowed_amounts(conn: sqlite3.Connection, reply: str, changes: dict) -> set[int]:
    """Every dollar amount the reply is entitled to name, from state."""
    named = {match.group(0).upper() for match in ORDER_ID.finditer(reply)}
    scope = named | _touched_orders(changes)
    if not scope:
        scope = {row["id"] for row in rows(conn, "SELECT id FROM orders")}

    allowed: set[int] = set()
    placeholders = ",".join("?" for _ in scope)
    for row in rows(conn, f"SELECT total_cents FROM orders WHERE id IN ({placeholders})", tuple(scope)):
        allowed.add(int(row["total_cents"]))
    for row in rows(
        conn,
        f"SELECT amount_cents FROM refunds WHERE order_id IN ({placeholders})",
        tuple(scope),
    ):
        allowed.add(int(row["amount_cents"]))
    for row in rows(
        conn,
        f"SELECT SUM(amount_cents) AS total FROM refunds WHERE status = 'succeeded'"
        f" AND order_id IN ({placeholders}) GROUP BY order_id",
        tuple(scope),
    ):
        allowed.add(int(row["total"]))
    # A reply may list a line item by name and price. That price is a real row
    # on the order it names, so it is bound; a price no item carries stays out.
    for row in rows(
        conn,
        f"SELECT unit_price_cents, quantity FROM order_items WHERE order_id IN ({placeholders})",
        tuple(scope),
    ):
        unit = int(row["unit_price_cents"])
        allowed.add(unit)
        allowed.add(unit * int(row["quantity"]))
    return allowed


def _known_addresses(conn: sqlite3.Connection, changes: dict) -> list[dict]:
    current = rows(conn, "SELECT id, line1, city, postcode FROM addresses")
    previous = [entry["before"] for entry in changes.get("addresses", {}).get("changed", [])]
    return current + previous


def check_tokens(
    conn: sqlite3.Connection, reply: str, changes: dict, ticket_text: str = ""
) -> ClaimResult:
    """Check every entity id, amount, tracking number and street against state.

    One pass, no sentence shape: an order id with no row, a tracking number no
    shipment carries, an amount no bound row explains, or a street no address
    row holds is a rejection.
    """
    out = ClaimResult()

    known_orders = {row["id"] for row in rows(conn, "SELECT id FROM orders")}
    known_orders |= {match.group(0).upper() for match in ORDER_ID.finditer(ticket_text)}
    for match in ORDER_ID.finditer(reply):
        order_id = match.group(0).upper()
        if order_id not in known_orders:
            out.reject(
                "TOKEN_ORDER_UNKNOWN",
                f"reply names {order_id}, which is not an order in state or this ticket",
            )

    known_tracking = {
        str(row["tracking_number"]).upper()
        for row in rows(conn, "SELECT tracking_number FROM shipments")
    }
    for value in _tracking_in(reply):
        if value not in known_tracking:
            out.reject(
                "TOKEN_TRACKING_UNKNOWN",
                f"reply names tracking {value}, which no shipment carries",
            )

    allowed = _allowed_amounts(conn, reply, changes)
    for amount in money_cents(reply):
        if amount not in allowed:
            out.reject(
                "AMOUNT_UNBOUND",
                f"reply names ${amount / 100:.2f}; no bound order, refund or delta carries it",
            )

    addresses = _known_addresses(conn, changes)
    for value in STREET.findall(reply):
        if not any(_address_matches(value, address) for address in addresses):
            out.reject(
                "ADDRESS_VALUE_MISMATCH",
                f"reply names address {value!r}, which no address row holds",
            )

    return out


def evaluate_claims(
    conn: sqlite3.Connection, claims: list[Claim], changes: dict
) -> ClaimResult:
    """Bind every claim to state. Returns rejects, escalations and reasons."""
    out = ClaimResult()
    seen: set[tuple] = set()
    for claim in claims:
        key = (
            claim.kind,
            claim.entity,
            tuple(claim.amounts),
            tuple(claim.tracking),
            claim.address_value,
            claim.refund_id,
            tuple(claim.stated_statuses),
            claim.prior,
            claim.status,
        )
        if key in seen:
            continue
        seen.add(key)
        if claim.kind == "refund":
            _verify_refund(conn, claim, changes, out)
        elif claim.kind == "reshipment":
            _verify_reshipment(conn, claim, changes, out)
        elif claim.kind == "shipment_status":
            _verify_shipment_status(conn, claim, out)
        elif claim.kind == "address":
            _verify_address(conn, claim, changes, out)
        elif claim.kind == "cancellation":
            _verify_cancellation(conn, claim, out)
        else:
            out.escalate("UNVERIFIED_CLAIM", f"{claim.describe()}: Phase A cannot bind this action")
    return out

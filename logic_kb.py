"""
=============================================================================
Module  : Logic / Knowledge Base Module
File    : logic_kb.py
Course  : AL2002 – Artificial Intelligence Lab (Spring 2026)
Project : Smart Campus AI Decision Support and Automation System
=============================================================================
Description:
    Rule-based reasoning engine. Acts as gatekeeper before CSP.

    Rules from FoL.png diagram (exact):
        R1: Student(x) ∧ Completed(x, ProgrammingFundamentals)
              → Eligible(x, AI)
        R2: Teaches(x, AI) → Instructor(x, AI)
        R3: Enrolled(x, AI) → UsesLab(x, Lab1)
        R4: Instructor(x, AI) → UsesLab(x, Lab1)

    Facts from diagram:
        Teaches(DrKhan, AI)
        Enrolled(Ali, AI)
        Student(Sara)
        Student(Ali)
        Completed(Ali, ProgrammingFundamentals)

    Pipeline position: ANN → [Logic/KB] → CSP
=============================================================================
"""

# ---------------------------------------------------------------------------
# Knowledge Base: facts (predicate, arg1, [arg2])
# Facts are stored as frozensets so duplicates are avoided naturally.
# ---------------------------------------------------------------------------

INITIAL_FACTS = [
    # --- Exact facts shown in FoL.png diagram ---
    ("Teaches",   "DrKhan", "AI"),
    ("Enrolled",  "Ali",    "AI"),
    ("Student",   "Sara"),
    ("Student",   "Ali"),
    ("Completed", "Ali",    "ProgrammingFundamentals"),

    # --- Extended facts for full system operation ---
    ("Teaches",   "DrAli",  "DB"),
    ("Teaches",   "DrHasan","SE"),
    ("Enrolled",  "Sara",   "AI"),
    ("Enrolled",  "Umar",   "AI"),
    ("Completed", "Sara",   "ProgrammingFundamentals"),
    ("Completed", "Umar",   "ProgrammingFundamentals"),
    ("Student",   "Umar"),
    ("Instructor", "DrKhan"),
    ("Instructor", "DrAli"),
    ("Instructor", "DrHasan"),
    ("Staff",     "Ahmad"),
    ("Staff",     "Bilal"),
    ("HasLabAccess", "DrKhan",  "AI_Lab"),
    ("HasLabAccess", "DrAli",   "DB_Lab"),
    ("HasLabAccess", "DrHasan", "SE_Lab"),
    ("Eligible",  "Ahmad",  "Maintenance"),
    ("Eligible",  "Bilal",  "Maintenance"),
]


def _make_kb() -> list:
    """
    Creates a fresh list of fact tuples from INITIAL_FACTS.

    Returns:
        list: List of fact tuples (immutable seed knowledge base).
    """
    return [tuple(f) for f in INITIAL_FACTS]


# ---------------------------------------------------------------------------
# Forward chaining engine
# ---------------------------------------------------------------------------

def forward_chain(kb: list) -> tuple:
    """
    Applies inference rules to KB until fixed point (no new facts).

    Rules from FoL.png (exact):
        R2: Teaches(x, AI)     → Instructor(x, AI)
        R4: Instructor(x, AI)  → UsesLab(x, Lab1)
        R3: Enrolled(x, AI)    → UsesLab(x, Lab1)
        R1: Student(x) ∧ Completed(x, ProgrammingFundamentals)
              → Eligible(x, AI)

    Additional service-permission rules for system operation:
        HasLabAccess(x, lab)       → UsesLab(x, lab)
        Instructor(x)              → CanUseLabSupport(x)
        Eligible(x, AI)            → CanUseLabSupport(x)
        Student(x)                 → CanUseLabSupport(x)   [if eligible]
        Staff(x)                   → CanUseMaintenance(x)
        Instructor(x) / Student(x) → CanScheduleViva(x)
        Student(x)                 → CanUseLibrary(x)

    Parameters:
        kb (list): Seed knowledge base (list of fact tuples).

    Returns:
        tuple: (extended_kb: list, derivation_log: list of str)
    """
    facts   = set(tuple(f) for f in kb)
    log     = []
    changed = True

    while changed:
        changed   = False
        new_facts = set()

        for fact in list(facts):
            pred = fact[0]

            # R2 (FoL diagram): Teaches(x, AI) → Instructor(x, AI)
            if pred == "Teaches" and len(fact) == 3 and fact[2] == "AI":
                x  = fact[1]
                nf = ("Instructor", x, "AI")
                if nf not in facts:
                    new_facts.add(nf)
                    log.append(f"R2: Teaches({x},AI) → Instructor({x},AI)")

            # R4 (FoL diagram): Instructor(x, AI) → UsesLab(x, Lab1)
            if pred == "Instructor" and len(fact) == 3 and fact[2] == "AI":
                x  = fact[1]
                nf = ("UsesLab", x, "Lab1")
                if nf not in facts:
                    new_facts.add(nf)
                    log.append(f"R4: Instructor({x},AI) → UsesLab({x},Lab1)")
                # also map Lab1 → AI_Lab for system routing
                nf2 = ("UsesLab", x, "AI_Lab")
                if nf2 not in facts:
                    new_facts.add(nf2)

            # R3 (FoL diagram): Enrolled(x, AI) → UsesLab(x, Lab1)
            if pred == "Enrolled" and len(fact) == 3 and fact[2] == "AI":
                x  = fact[1]
                nf = ("UsesLab", x, "Lab1")
                if nf not in facts:
                    new_facts.add(nf)
                    log.append(f"R3: Enrolled({x},AI) → UsesLab({x},Lab1)")

            # HasLabAccess(x, lab) → UsesLab(x, lab)
            if pred == "HasLabAccess" and len(fact) == 3:
                x, lab = fact[1], fact[2]
                nf = ("UsesLab", x, lab)
                if nf not in facts:
                    new_facts.add(nf)
                    log.append(f"HasLabAccess({x},{lab}) → UsesLab({x},{lab})")

            # Instructor(x) or Instructor(x,AI) → CanUseLabSupport(x)
            if pred == "Instructor":
                x  = fact[1]
                nf = ("CanUseLabSupport", x)
                if nf not in facts:
                    new_facts.add(nf)
                    log.append(f"Instructor({x}) → CanUseLabSupport({x})")
                nf2 = ("CanScheduleViva", x)
                if nf2 not in facts:
                    new_facts.add(nf2)

            # Staff(x) → CanUseMaintenance(x)
            if pred == "Staff":
                x  = fact[1]
                nf = ("CanUseMaintenance", x)
                if nf not in facts:
                    new_facts.add(nf)
                    log.append(f"Staff({x}) → CanUseMaintenance({x})")

            # Student(x) → CanUseLibrary(x), CanScheduleViva(x)
            if pred == "Student":
                x = fact[1]
                for derived in [("CanUseLibrary", x), ("CanScheduleViva", x)]:
                    if derived not in facts:
                        new_facts.add(derived)
                        log.append(f"Student({x}) → {derived[0]}({x})")

            # Eligible(x, AI) → CanUseLabSupport(x)
            if pred == "Eligible" and len(fact) == 3 and fact[2] in ("AI", "AI_Lab_Support"):
                x  = fact[1]
                nf = ("CanUseLabSupport", x)
                if nf not in facts:
                    new_facts.add(nf)
                    log.append(f"Eligible({x},{fact[2]}) → CanUseLabSupport({x})")

        # R1 (FoL diagram): Student(x) ∧ Completed(x, ProgrammingFundamentals)
        #                    → Eligible(x, AI)
        students   = {f[1] for f in facts if f[0] == "Student"}
        completed  = {f[1] for f in facts
                      if f[0] == "Completed" and f[2] == "ProgrammingFundamentals"}
        for x in students & completed:
            for label in ("AI", "AI_Lab_Support"):
                nf = ("Eligible", x, label)
                if nf not in facts:
                    new_facts.add(nf)
                    log.append(f"R1: Student({x}) ∧ Completed({x},PF) → Eligible({x},{label})")

        if new_facts:
            facts  |= new_facts
            changed = True

    return list(facts), log


# ---------------------------------------------------------------------------
# Entailment check
# ---------------------------------------------------------------------------

def check_entailment(query_tuple: tuple, kb_extended: list) -> tuple:
    """
    Checks whether a given query fact is entailed by the extended KB.

    Parameters:
        query_tuple  (tuple): Fact to check, e.g. ("UsesLab","DrKhan","AI_Lab")
        kb_extended  (list) : Fully forward-chained knowledge base.

    Returns:
        tuple: (entailed: bool, matching_facts: list of str)
    """
    facts_set = set(tuple(f) for f in kb_extended)

    # Exact match
    if query_tuple in facts_set:
        return True, [str(query_tuple)]

    # Partial match: predicate + first arg (wildcard second arg)
    if len(query_tuple) == 2:
        matching = [str(f) for f in facts_set
                    if f[0] == query_tuple[0] and f[1] == query_tuple[1]]
        if matching:
            return True, matching

    return False, []


def parse_query(query_str: str) -> tuple:
    """
    Parses a query string such as "UsesLab(DrKhan, AI_Lab)" into a tuple
    ("UsesLab", "DrKhan", "AI_Lab").

    Supports 1-arg, 2-arg, and 3-arg predicates.

    Parameters:
        query_str (str): FOL-style query string.

    Returns:
        tuple: Parsed fact tuple, or empty tuple if unparseable.
    """
    query_str = query_str.strip()
    if "(" not in query_str:
        return ()
    try:
        predicate, rest = query_str.split("(", 1)
        rest            = rest.rstrip(")")
        args            = [a.strip() for a in rest.split(",")]
        return (predicate.strip(),) + tuple(args)
    except Exception:
        return ()


# ---------------------------------------------------------------------------
# Service-category permission checker
# ---------------------------------------------------------------------------

CATEGORY_PERMISSION_MAP = {
    "AI_Lab_Support"  : ["CanUseLabSupport"],
    "Viva_Scheduling" : ["CanScheduleViva"],
    "Access_Request"  : ["CanUseLibrary", "UsesLab"],
    "Maintenance"     : ["CanUseMaintenance"],
    "Emergency_Help"  : ["Student", "Instructor", "Staff"],
}


def check_permission(name: str, category: str, kb_extended: list) -> tuple:
    """
    Checks whether the named user has permission for the requested category.

    Uses the CATEGORY_PERMISSION_MAP to determine which derived predicates
    to look for in the extended knowledge base.

    Parameters:
        name         (str)  : User's name as it appears in the KB.
        category     (str)  : Service category (e.g., "AI_Lab_Support").
        kb_extended  (list) : Fully forward-chained knowledge base.

    Returns:
        tuple: (allowed: bool, explanation: str)
    """
    predicates = CATEGORY_PERMISSION_MAP.get(category, [])
    if not predicates:
        return False, f"No permission rules defined for category '{category}'."

    facts_set = set(tuple(f) for f in kb_extended)

    for pred in predicates:
        # Check (pred, name) or (pred, name, anything)
        matches = [f for f in facts_set
                   if f[0] == pred and f[1] == name]
        if matches:
            return True, (f"{name} has permission for '{category}' via "
                          f"rule '{pred}({name}, ...)'.")

    # Check direct Eligible fact
    elig_fact = ("Eligible", name, category)
    if elig_fact in facts_set:
        return True, f"Eligible({name},{category}) is directly asserted in KB."

    return False, (f"{name} does not satisfy any permission predicate "
                   f"for category '{category}'. "
                   f"Required: {predicates}. "
                   f"Please verify role, enrolment, and prerequisites.")


# ---------------------------------------------------------------------------
# Main entry point for the pipeline
# ---------------------------------------------------------------------------

def run_logic_kb(request_obj: dict) -> dict:
    """
    Main entry point for the Logic / KB module.

    Behaviour depends on request_type:
        Eligibility_Check      → parse and answer the query field
        Booking_or_Scheduling,
        Urgent_Service_Request,
        Full_Service_Request   → check role/category permission

    Always runs forward chaining first to derive maximum facts.

    Parameters:
        request_obj (dict): Preprocessed and (optionally ANN-processed) request.

    Returns:
        dict: {
            "allowed"       : bool,
            "entailed"      : bool,
            "explanation"   : str or list,
            "derived_count" : int,   (new facts derived by forward chaining)
        }
    """
    kb_seed               = _make_kb()
    kb_extended, chain_log = forward_chain(kb_seed)
    derived_count          = len(kb_extended) - len(kb_seed)

    rtype    = request_obj.get("request_type", "")
    name     = request_obj.get("name", "")
    category = request_obj.get("category", "")

    # ── Eligibility_Check: answer the explicit query
    if rtype == "Eligibility_Check":
        query_str = request_obj.get("query", "")
        qt        = parse_query(query_str)

        if not qt:
            return {
                "allowed"      : False,
                "entailed"     : False,
                "explanation"  : f"Could not parse query: '{query_str}'.",
                "derived_count": derived_count,
            }

        entailed, matches = check_entailment(qt, kb_extended)
        explanation = (
            f"Query '{query_str}' is ENTAILED. Supporting facts: {matches}"
            if entailed
            else f"Query '{query_str}' is NOT entailed by the current KB."
        )
        return {
            "allowed"      : entailed,
            "entailed"     : entailed,
            "explanation"  : explanation,
            "derived_count": derived_count,
        }

    # ── Service requests: permission check
    if name and category:
        allowed, explanation = check_permission(name, category, kb_extended)
        return {
            "allowed"      : allowed,
            "entailed"     : allowed,
            "explanation"  : explanation,
            "derived_count": derived_count,
        }

    # ── Fallback: no valid check possible
    return {
        "allowed"      : False,
        "entailed"     : False,
        "explanation"  : "Insufficient information to perform logic check.",
        "derived_count": derived_count,
    }

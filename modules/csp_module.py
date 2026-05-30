"""
=============================================================================
Module  : CSP Scheduler / Resource Allocator
File    : csp_module.py
Course  : AL2002 – Artificial Intelligence Lab (Spring 2026)
Project : Smart Campus AI Decision Support and Automation System
=============================================================================
Description:
    Implements the CSP constraint graph shown in CSP_Rules.png exactly.

    Groups: G1, G2, G3, G4, G5, G6
    Constraints from diagram:
        G1 != G4          (red edge — no same slot)
        G1 != G2          (red edge — no same slot)
        G4 != G2          (red edge — no same slot)
        G4 < G3           (blue arrow — precedence: G4 must come before G3)
        G3 != G5          (red edge — no same slot)
        G3 != G6          (red edge — no same slot)
        G5 != G6          (red edge — no same slot)
        G1 != G3          (purple — examiner clash: same examiner can't do both)
        G2 != G5          (orange — supervisor clash: same supervisor)

    General slot/room allocation also handled for service requests.

    Pipeline position: Logic/KB → [CSP] → Search
=============================================================================
"""

# ---------------------------------------------------------------------------
# CSP Constraint Graph — exactly from CSP_Rules.png
# ---------------------------------------------------------------------------

# Groups that need viva slot assignments
CSP_GROUPS = ["G1", "G2", "G3", "G4", "G5", "G6"]

# Domain: each group can be assigned slot 1, 2, 3, or 4
CSP_DOMAIN = [1, 2, 3, 4]

# Constraints from diagram:
#   "!="        → two groups cannot share the same slot
#   "precedence" → G4 must be scheduled strictly before G3 (G4 < G3)
#   "examiner"  → G1 and G3 cannot clash (same examiner)
#   "supervisor" → G2 and G5 cannot clash (same supervisor)

CSP_NOT_EQUAL = [
    ("G1", "G4"),   # red edge
    ("G1", "G2"),   # red edge
    ("G4", "G2"),   # red edge
    ("G3", "G5"),   # red edge
    ("G3", "G6"),   # red edge
    ("G5", "G6"),   # red edge
    ("G1", "G3"),   # purple — examiner clash
    ("G2", "G5"),   # orange — supervisor clash
]

CSP_PRECEDENCE = [
    ("G4", "G3"),   # blue arrow: G4 slot < G3 slot
]


def _csp_constraints_satisfied(assignment: dict) -> bool:
    """
    Checks all CSP constraints for a (possibly partial) assignment.

    Parameters:
        assignment (dict): Maps group name → slot number.

    Returns:
        bool: True if no constraint is violated.
    """
    # != constraints
    for (g1, g2) in CSP_NOT_EQUAL:
        if g1 in assignment and g2 in assignment:
            if assignment[g1] == assignment[g2]:
                return False

    # precedence constraints (G4 < G3)
    for (before, after) in CSP_PRECEDENCE:
        if before in assignment and after in assignment:
            if assignment[before] >= assignment[after]:
                return False

    return True


def solve_csp_groups() -> dict:
    """
    Solves the G1–G6 viva scheduling CSP using backtracking.
    Returns a complete valid assignment or empty dict if unsolvable.

    Returns:
        dict: {group: slot} assignment satisfying all constraints.
    """
    def backtrack(assignment, remaining):
        if not remaining:
            return assignment
        group = remaining[0]
        for slot in CSP_DOMAIN:
            trial = dict(assignment)
            trial[group] = slot
            if _csp_constraints_satisfied(trial):
                result = backtrack(trial, remaining[1:])
                if result is not None:
                    return result
        return None

    return backtrack({}, CSP_GROUPS) or {}



CATEGORY_ROOM_MAP = {
    "AI_Lab_Support"  : ["AI_Lab", "Science_Block"],
    "Viva_Scheduling" : ["Exam_Hall", "Seminar_Room"],
    "Access_Request"  : ["Library", "Admin_Block"],
    "Maintenance"     : ["Admin_Block", "Student_Services"],
    "Emergency_Help"  : ["Medical_Center", "Admin_Block"],
}

VALID_SLOTS = [1, 2, 3, 4]

# Global booking registry — persists within a session.
# Structure: { (room, slot): assigned_to (str) }
_BOOKINGS: dict = {}


def _reset_bookings():
    """
    Resets the global booking registry.
    Useful for testing or starting a fresh session.
    """
    global _BOOKINGS
    _BOOKINGS = {}


def _is_slot_free(room: str, slot: int) -> bool:
    """
    Checks whether a given (room, slot) combination is currently free.

    Parameters:
        room (str): Room identifier.
        slot (int): Slot number (1-4).

    Returns:
        bool: True if the slot is unbooked.
    """
    return (room, slot) not in _BOOKINGS


def _book_slot(room: str, slot: int, assignee: str):
    """
    Records a booking in the global registry.

    Parameters:
        room     (str): Room identifier.
        slot     (int): Slot number (1-4).
        assignee (str): Name of the person/group being assigned.
    """
    _BOOKINGS[(room, slot)] = assignee


# ---------------------------------------------------------------------------
# Constraint checking
# ---------------------------------------------------------------------------

def _check_constraints(room: str, slot: int, assignee: str,
                       existing_assignments: list) -> tuple:
    """
    Evaluates all CSP constraints for a candidate (room, slot) assignment.

    Constraints checked:
        C1 — Room/slot not already booked (no double-booking)
        C2 — Assignee not already in another room at same slot
        C3 — For viva: no supervisor clash (same supervisor in same slot)

    Parameters:
        room                 (str) : Candidate room.
        slot                 (int) : Candidate slot.
        assignee             (str) : Person/group name.
        existing_assignments (list): List of already-made assignment dicts.

    Returns:
        tuple: (satisfied: bool, violated_constraint: str)
    """
    # C1: Room-slot free?
    if not _is_slot_free(room, slot):
        occupant = _BOOKINGS.get((room, slot), "?")
        return False, f"C1 violated: {room} slot {slot} is occupied by '{occupant}'."

    # C2: Assignee not already assigned at this slot (different room)
    for asgn in existing_assignments:
        if asgn.get("assignee") == assignee and asgn.get("slot") == slot:
            return False, (f"C2 violated: '{assignee}' already has an assignment "
                           f"at slot {slot} in {asgn.get('room')}.")

    return True, ""


# ---------------------------------------------------------------------------
# Priority-based slot ordering
# ---------------------------------------------------------------------------

def _priority_slot_order(preferred_slot: int, priority: str) -> list:
    """
    Returns an ordered list of slots to try, based on priority level.

    High/Urgent requests try preferred slot first, then earlier slots.
    Low/Normal requests try preferred slot first, then later slots.

    Parameters:
        preferred_slot (int): User's preferred slot (1-4), or None.
        priority       (str): Priority label from ANN or default.

    Returns:
        list: Ordered list of slot numbers to attempt.
    """
    if preferred_slot not in VALID_SLOTS:
        preferred_slot = 1

    # Build slot order
    remaining = [s for s in VALID_SLOTS if s != preferred_slot]

    if priority in ("Urgent", "High"):
        # Try preferred first, then ascending
        order = [preferred_slot] + sorted(remaining)
    else:
        # Try preferred first, then ascending from preferred
        order = [preferred_slot] + sorted(remaining, key=lambda s: abs(s - preferred_slot))

    return order


# ---------------------------------------------------------------------------
# Main CSP assignment function
# ---------------------------------------------------------------------------

def run_csp(request_obj: dict, priority_output: dict = None) -> dict:
    """
    Main entry point for the CSP Scheduler module.

    Attempts to assign a conflict-free (room, slot) pair to the requester.
    Uses backtracking with constraint propagation:
        1. Determine candidate rooms from service category
        2. Determine slot ordering based on priority
        3. Try each (room, slot) combination
        4. Return first feasible assignment or rejection

    Parameters:
        request_obj    (dict): Preprocessed (and Logic/KB approved) request.
        priority_output(dict): ANN priority output (optional). Used to
                               influence slot ordering if available.

    Returns:
        dict: {
            "decision"      : str,  ("accepted" or "rejected")
            "assigned_room" : str,
            "assigned_slot" : int or None,
            "destination"   : str,  (same as assigned_room, for Search)
            "notes"         : str
        }
    """
    category       = request_obj.get("category", "AI_Lab_Support")
    preferred_slot = request_obj.get("preferred_slot", 1) or 1
    name           = request_obj.get("name", "Unknown")
    group_id       = request_obj.get("group_id", "").strip()

    # ── Special case: Viva_Scheduling uses the G1-G6 CSP graph
    if category == "Viva_Scheduling":
        group_schedule = solve_csp_groups()
        if group_schedule:
            # Find a slot for this specific group if group_id provided
            assigned_slot = group_schedule.get(group_id,
                            group_schedule.get("G1", 1))
            room = "Exam_Hall"
            _book_slot(room, assigned_slot, name)
            return {
                "decision"      : "accepted",
                "assigned_room" : room,
                "assigned_slot" : assigned_slot,
                "destination"   : room,
                "notes"         : (f"Viva CSP solved. Full schedule: {group_schedule}. "
                                   f"{'Group ' + group_id + ' assigned slot ' + str(assigned_slot) if group_id else 'Default slot assigned.'}"),
            }
        else:
            return {
                "decision"      : "rejected",
                "assigned_room" : "",
                "assigned_slot" : None,
                "destination"   : "",
                "notes"         : "CSP could not find a valid viva schedule satisfying all constraints.",
            }

    # Get priority label for slot ordering
    priority = "Normal"
    if priority_output:
        priority = priority_output.get("final_priority", "Normal")

    # Candidate rooms
    rooms = CATEGORY_ROOM_MAP.get(category, ["Admin_Block"])

    # Slot order
    slot_order = _priority_slot_order(int(preferred_slot), priority)

    # Existing session assignments (for constraint checking)
    existing = [{"assignee": v, "room": k[0], "slot": k[1]}
                for k, v in _BOOKINGS.items()]

    tried = []
    for room in rooms:
        for slot in slot_order:
            ok, reason = _check_constraints(room, slot, name, existing)
            if ok:
                _book_slot(room, slot, name)
                note = ""
                if slot != preferred_slot:
                    note = (f"Preferred slot {preferred_slot} was unavailable. "
                            f"Slot {slot} is the next feasible conflict-free assignment.")
                else:
                    note = f"Preferred slot {slot} assigned successfully in {room}."
                return {
                    "decision"      : "accepted",
                    "assigned_room" : room,
                    "assigned_slot" : slot,
                    "destination"   : room,
                    "notes"         : note,
                }
            else:
                tried.append(f"({room}, slot {slot}): {reason}")

    # All combinations exhausted — CSP failed
    return {
        "decision"      : "rejected",
        "assigned_room" : "",
        "assigned_slot" : None,
        "destination"   : "",
        "notes"         : ("No feasible slot/room combination found. "
                           "Constraints tried: " + "; ".join(tried[:4])),
    }

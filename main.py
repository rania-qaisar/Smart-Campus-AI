"""
=============================================================================
File    : main.py
Project : Smart Campus AI Decision Support and Automation System
Course  : AL2002 – Artificial Intelligence Lab (Spring 2026)
=============================================================================
Description:
    Main CLI entry point for the Smart Campus AI platform.
    Handles the full interactive loop:
        1. Collects structured input from user via CLI prompts
        2. Passes input through the full AI pipeline
        3. Displays formatted final response

    Run this file directly to use the system:
        python main.py

    The CLI guides the user step-by-step through each required field
    for their chosen request type. No free-form text parsing is needed.
=============================================================================
"""

import sys
import os

# Ensure modules directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.preprocessing  import preprocess_request
from modules.router          import route_request, describe_pipeline
from modules.ann_module      import run_ann
from modules.logic_kb        import run_logic_kb
from modules.csp_module      import run_csp
from modules.search_module   import run_search
from modules.response_layer  import build_final_response, print_response


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║       SMART CAMPUS AI DECISION SUPPORT & AUTOMATION SYSTEM      ║
║       AL2002 — Artificial Intelligence Lab  |  Spring 2026       ║
║       FAST-NUCES                                                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

REQUEST_MENU = """
Select Request Type:
  1. Navigation_Only
  2. Eligibility_Check
  3. Booking_or_Scheduling
  4. Urgent_Service_Request
  5. Full_Service_Request
"""

CATEGORY_MENU = """
Select Category:
  1. AI_Lab_Support
  2. Viva_Scheduling
  3. Access_Request
  4. Maintenance
  5. Emergency_Help
"""

VALID_LOCATIONS_LIST = [
    "Main_Gate", "Parking", "Admin_Block", "Student_Services",
    "Exam_Hall", "Seminar_Room", "AI_Lab", "Science_Block",
    "Library", "Cafeteria", "Hostel", "Medical_Center", "Bus_Stop"
]


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def prompt(label: str, default: str = "") -> str:
    """
    Prompts the user for a string input with optional default.

    Parameters:
        label   (str): Display label for the prompt.
        default (str): Default value shown in brackets.

    Returns:
        str: User input string (or default if empty).
    """
    if default:
        val = input(f"  {label} [{default}]: ").strip()
        return val if val else default
    val = input(f"  {label}: ").strip()
    return val


def prompt_int(label: str, lo: int, hi: int, default: int = None) -> int:
    """
    Prompts for an integer in range [lo, hi], retrying on invalid input.

    Parameters:
        label   (str): Prompt label.
        lo      (int): Minimum valid value.
        hi      (int): Maximum valid value.
        default (int): Default value if user presses Enter.

    Returns:
        int: Validated integer.
    """
    while True:
        hint = f" [{default}]" if default is not None else f" ({lo}-{hi})"
        raw  = input(f"  {label}{hint}: ").strip()
        if not raw and default is not None:
            return default
        try:
            val = int(raw)
            if lo <= val <= hi:
                return val
            print(f"  ✗ Please enter a number between {lo} and {hi}.")
        except ValueError:
            print(f"  ✗ Invalid input. Please enter a number.")


def show_locations():
    """Prints all valid campus location names."""
    print("\n  Campus Locations:")
    for i, loc in enumerate(VALID_LOCATIONS_LIST, 1):
        print(f"    {i:2d}. {loc}")
    print()


def collect_input() -> dict:
    """
    Collects all required input fields from the user via interactive CLI.
    Fields collected depend on the selected request type.

    Returns:
        dict: Raw input dictionary ready for preprocessing.
    """
    raw = {}

    # ── Base fields
    print("\n── User Information ──────────────────────────────────")
    raw["name"] = prompt("Enter Name")
    while not raw["name"]:
        print("  ✗ Name cannot be empty.")
        raw["name"] = prompt("Enter Name")

    print("\n  Role options: student / instructor / staff")
    raw["role"] = prompt("Enter Role", default="student")

    # ── Request type
    print(REQUEST_MENU)
    choice = prompt_int("Enter choice", 1, 5)
    type_map = {
        1: "Navigation_Only",
        2: "Eligibility_Check",
        3: "Booking_or_Scheduling",
        4: "Urgent_Service_Request",
        5: "Full_Service_Request",
    }
    raw["request_type"] = type_map[choice]
    rtype = raw["request_type"]

    print(f"\n── {rtype} Fields ──────────────────────────────────")

    # ── Navigation Only
    if rtype == "Navigation_Only":
        show_locations()
        raw["current_location"] = prompt("Current Location", default="Hostel")
        raw["destination"]      = prompt("Destination",       default="AI_Lab")

    # ── Eligibility Check
    elif rtype == "Eligibility_Check":
        print("\n  Example queries:")
        print("    UsesLab(DrKhan, AI_Lab)")
        print("    Eligible(Ali, AI_Lab_Support)")
        print("    CanScheduleViva(Sara)")
        raw["query"] = prompt("Enter Query (FOL-style)")

    # ── Booking / Scheduling
    elif rtype == "Booking_or_Scheduling":
        print(CATEGORY_MENU)
        cat_choice  = prompt_int("Category", 1, 5, default=1)
        cat_map     = {1:"AI_Lab_Support",2:"Viva_Scheduling",
                       3:"Access_Request",4:"Maintenance",5:"Emergency_Help"}
        raw["category"]        = cat_map[cat_choice]
        raw["preferred_slot"]  = prompt_int("Preferred Slot", 1, 4, default=1)
        show_locations()
        raw["current_location"] = prompt("Current Location", default="Hostel")
        raw["group_id"]         = prompt("Group ID (optional, press Enter to skip)", default="")

    # ── Urgent Service Request
    elif rtype == "Urgent_Service_Request":
        print(CATEGORY_MENU)
        cat_choice  = prompt_int("Category", 1, 5, default=1)
        cat_map     = {1:"AI_Lab_Support",2:"Viva_Scheduling",
                       3:"Access_Request",4:"Maintenance",5:"Emergency_Help"}
        raw["category"]         = cat_map[cat_choice]
        show_locations()
        raw["current_location"] = prompt("Current Location", default="Hostel")
        raw["severity"]         = prompt_int("Severity (1-10)", 1, 10, default=8)
        raw["time_sensitivity"] = prompt_int("Time Sensitivity (1-10)", 1, 10, default=9)
        raw["crowd_level"]      = prompt_int("Crowd Level (1-10)", 1, 10, default=5)
        raw["preferred_slot"]   = prompt_int("Preferred Slot (1-4)", 1, 4, default=2)

    # ── Full Service Request
    elif rtype == "Full_Service_Request":
        print(CATEGORY_MENU)
        cat_choice  = prompt_int("Category", 1, 5, default=1)
        cat_map     = {1:"AI_Lab_Support",2:"Viva_Scheduling",
                       3:"Access_Request",4:"Maintenance",5:"Emergency_Help"}
        raw["category"]         = cat_map[cat_choice]
        show_locations()
        raw["current_location"] = prompt("Current Location", default="Hostel")
        raw["preferred_slot"]   = prompt_int("Preferred Slot (1-4)", 1, 4, default=2)
        raw["severity"]         = prompt_int("Severity (1-10)",  1, 10, default=8)
        raw["time_sensitivity"] = prompt_int("Time Sensitivity (1-10)", 1, 10, default=9)
        raw["crowd_level"]      = prompt_int("Crowd Level (1-10)", 1, 10, default=5)
        raw["description_note"] = prompt("Description Note (optional)", default="")

    return raw


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline(raw: dict, verbose: bool = True) -> dict:
    """
    Executes the full AI pipeline for the given raw input dictionary.
    Respects the routing rules — only runs modules the request type requires.

    Steps:
        1. Preprocessing  → validate, normalise, build request object
        2. Router         → decide module pipeline
        3. ANN            → priority prediction (if needed)
        4. Logic/KB       → permission/eligibility check (if needed)
        5. CSP            → slot/room assignment (if needed)
        6. Search         → route finding (if needed)
        7. Final Response → aggregate all outputs

    Parameters:
        raw     (dict): Raw CLI input dictionary.
        verbose (bool): If True, prints pipeline progress to console.

    Returns:
        dict: Final response object.
    """

    def vprint(msg):
        if verbose:
            print(f"  ▶ {msg}")

    # ── Step 1: Preprocessing
    vprint("Running Preprocessing...")
    ok, result = preprocess_request(raw)
    if not ok:
        print(f"\n  ✗ Preprocessing FAILED: {result.get('error', 'Unknown error')}")
        return {"decision": "rejected", "message": result.get("error", "")}

    request_obj = result
    if verbose:
        print(f"    Request ID: {request_obj['request_id']}")

    # ── Step 2: Router
    vprint("Running Request Router...")
    router_out = route_request(request_obj)
    if "error" in router_out:
        print(f"\n  ✗ Router ERROR: {router_out['error']}")
        return {"decision": "rejected", "message": router_out["error"]}
    if verbose:
        print(f"    {describe_pipeline(router_out)}")

    # Module outputs (initialise as None)
    ann_out    = None
    logic_out  = None
    csp_out    = None
    search_out = None

    # ── Step 3: ANN
    if router_out["needs_ann"]:
        vprint("Running ANN Priority Module...")
        ann_out = run_ann(request_obj)
        if verbose:
            print(f"    Binary: {ann_out['binary_priority']} | "
                  f"Final: {ann_out['final_priority']} | "
                  f"Confidence: {ann_out['confidence']:.2%}")

    # ── Step 4: Logic / KB
    if router_out["needs_logic"]:
        vprint("Running Logic / Knowledge Base Module...")
        logic_out = run_logic_kb(request_obj)
        allowed_str = "ALLOWED" if logic_out["allowed"] else "DENIED"
        if verbose:
            print(f"    Result: {allowed_str}")
            print(f"    {logic_out['explanation']}")

        # GATEKEEPER: if denied, stop pipeline
        if not logic_out["allowed"]:
            vprint("Pipeline stopped — Logic/KB denied the request.")
            return build_final_response(request_obj, router_out,
                                        ann_out, logic_out,
                                        csp_out, search_out)

    # ── Step 5: CSP
    if router_out["needs_csp"]:
        vprint("Running CSP Scheduler...")
        csp_out = run_csp(request_obj, priority_output=ann_out)
        if verbose:
            print(f"    Decision : {csp_out['decision'].upper()}")
            if csp_out["decision"] == "accepted":
                print(f"    Room/Slot: {csp_out['assigned_room']} / Slot {csp_out['assigned_slot']}")

        # If CSP fails, stop pipeline
        if csp_out["decision"] == "rejected":
            vprint("Pipeline stopped — CSP could not find a valid assignment.")
            return build_final_response(request_obj, router_out,
                                        ann_out, logic_out,
                                        csp_out, search_out)

        # If CSP assigned a destination, propagate it for Search
        if csp_out.get("destination"):
            request_obj["destination"] = csp_out["destination"]

    # ── Step 6: Search
    if router_out["needs_search"]:
        source = request_obj.get("current_location", "")
        dest   = request_obj.get("destination", "")

        if source and dest:
            vprint(f"Running Search Module ({source} → {dest})...")
            search_out = run_search(source, dest, weighted=True)
            if verbose:
                if search_out.get("found"):
                    path_str = " → ".join(search_out.get("path", []))
                    print(f"    Algorithm: {search_out['algorithm_used']}")
                    print(f"    Path     : {path_str}")
                    print(f"    Cost     : {search_out['cost']} units")
                else:
                    print(f"    No route found from {source} to {dest}.")
        else:
            vprint("Search skipped — source or destination not available.")

    # ── Step 7: Final Response
    vprint("Building Final Response...")
    final = build_final_response(request_obj, router_out,
                                 ann_out, logic_out,
                                 csp_out, search_out)
    return final


# ---------------------------------------------------------------------------
# Demo mode — runs all 5 request types automatically
# ---------------------------------------------------------------------------

def run_demo():
    """
    Runs a pre-configured demonstration of all 5 request types.
    Each example showcases a different pipeline combination.
    """
    print("\n" + "═"*65)
    print("  DEMO MODE — Running all 5 request type examples")
    print("═"*65)

    demos = [
        {
            "label": "Demo 1 — Navigation Only",
            "data" : {
                "name": "Ali", "role": "student",
                "request_type": "Navigation_Only",
                "current_location": "Hostel",
                "destination": "AI_Lab"
            }
        },
        {
            "label": "Demo 2 — Eligibility Check (DrKhan → UsesLab)",
            "data" : {
                "name": "DrKhan", "role": "instructor",
                "request_type": "Eligibility_Check",
                "query": "UsesLab(DrKhan, AI_Lab)"
            }
        },
        {
            "label": "Demo 3 — Booking / Scheduling",
            "data" : {
                "name": "Sara", "role": "student",
                "request_type": "Booking_or_Scheduling",
                "category": "AI_Lab_Support",
                "preferred_slot": 2,
                "current_location": "Hostel"
            }
        },
        {
            "label": "Demo 4 — Urgent Service Request",
            "data" : {
                "name": "Ali", "role": "student",
                "request_type": "Urgent_Service_Request",
                "category": "AI_Lab_Support",
                "current_location": "Hostel",
                "severity": 8, "time_sensitivity": 9,
                "crowd_level": 5, "preferred_slot": 2
            }
        },
        {
            "label": "Demo 5 — Full Service Request (complete pipeline)",
            "data" : {
                "name": "Umar", "role": "student",
                "request_type": "Full_Service_Request",
                "category": "AI_Lab_Support",
                "current_location": "Hostel",
                "preferred_slot": 3,
                "severity": 9, "time_sensitivity": 9,
                "crowd_level": 4,
                "description_note": "Need urgent AI lab help before exam."
            }
        },
    ]

    for demo in demos:
        print(f"\n{'─'*65}")
        print(f"  {demo['label']}")
        print("─"*65)
        final = run_pipeline(demo["data"], verbose=True)
        print_response(final)
        input("  Press Enter for next demo...")


# ---------------------------------------------------------------------------
# Comparison mode — shows all search algorithms on one route
# ---------------------------------------------------------------------------

def run_comparison_mode():
    """
    Runs all 9 search algorithms on a user-specified route and
    prints a comparison table of results.
    """
    from modules.search_module import (bfs, dfs, dls, ids, ucs,
                                       bidirectional_bfs, greedy_bfs,
                                       astar, rbfs,
                                       WEIGHTED_GRAPH, UNWEIGHTED_GRAPH)

    print("\n── Search Algorithm Comparison Mode ────────────────────")
    print("  Locations:", ", ".join([
        "Main_Gate","Parking","Admin_Block","Student_Services",
        "Exam_Hall","Seminar_Room","AI_Lab","Science_Block",
        "Library","Cafeteria","Hostel","Medical_Center","Bus_Stop"
    ]))
    source = prompt("Start location", default="Hostel")
    dest   = prompt("Destination",    default="AI_Lab")
    print()

    algorithms = [
        ("BFS (unweighted)", lambda: bfs(UNWEIGHTED_GRAPH, source, dest)),
        ("DFS",              lambda: dfs(WEIGHTED_GRAPH, source, dest)),
        ("DLS (limit=6)",    lambda: dls(WEIGHTED_GRAPH, source, dest, 6)),
        ("IDS",              lambda: ids(WEIGHTED_GRAPH, source, dest)),
        ("UCS",              lambda: ucs(WEIGHTED_GRAPH, source, dest)),
        ("Bidirectional BFS",lambda: bidirectional_bfs(WEIGHTED_GRAPH, source, dest)),
        ("Greedy BFS",       lambda: greedy_bfs(WEIGHTED_GRAPH, source, dest)),
        ("A*",               lambda: astar(WEIGHTED_GRAPH, source, dest)),
        ("RBFS",             lambda: rbfs(WEIGHTED_GRAPH, source, dest)),
    ]

    print(f"  {'Algorithm':<22} {'Found':>5} {'Cost':>6} {'Steps':>6} {'Nodes':>6}  Path")
    print(f"  {'─'*22} {'─'*5} {'─'*6} {'─'*6} {'─'*6}  {'─'*30}")

    for name, fn in algorithms:
        try:
            r = fn()
            found = "Yes" if r.get("found") else "No"
            cost  = str(r.get("cost", "-"))
            steps = str(r.get("steps", "-"))
            nodes = str(r.get("nodes_expanded", "-"))
            path  = " → ".join(r.get("path", [])) if r.get("found") else "No path"
            print(f"  {name:<22} {found:>5} {cost:>6} {steps:>6} {nodes:>6}  {path}")
        except Exception as e:
            print(f"  {name:<22} ERROR: {e}")


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main():
    """
    Main loop for the Smart Campus AI CLI application.
    Displays a menu and handles user navigation between modes.
    """
    print(BANNER)
    print("[ANN] Models loaded and ready.")

    while True:
        print("\n" + "─"*65)
        print("  MAIN MENU")
        print("─"*65)
        print("  1. Submit a New Request")
        print("  2. Run Demo (all 5 request types)")
        print("  3. Search Algorithm Comparison")
        print("  4. Exit")
        print()

        choice = prompt_int("Enter choice", 1, 4)

        if choice == 1:
            try:
                raw   = collect_input()
                print("\n" + "─"*65)
                print("  Processing Request...")
                print("─"*65)
                final = run_pipeline(raw, verbose=True)
                print_response(final)
            except KeyboardInterrupt:
                print("\n  Cancelled.")
            except Exception as e:
                print(f"\n  ✗ Unexpected error: {e}")

        elif choice == 2:
            run_demo()

        elif choice == 3:
            try:
                run_comparison_mode()
            except KeyboardInterrupt:
                print("\n  Cancelled.")

        elif choice == 4:
            print("\n  Goodbye!\n")
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

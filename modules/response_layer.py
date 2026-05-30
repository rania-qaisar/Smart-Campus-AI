"""
=============================================================================
Module  : Final Response Layer
File    : response_layer.py
Course  : AL2002 – Artificial Intelligence Lab (Spring 2026)
Project : Smart Campus AI Decision Support and Automation System
=============================================================================
Description:
    This is the output aggregation layer of the Smart Campus AI platform.
    It collects results from all modules that ran in the pipeline and
    combines them into one coherent, human-readable response.

    The response layer is aware of which modules ran (via the router output)
    and only includes fields from modules that were actually executed.

    Fields in the final response:
        request_id  : Unique request identifier
        decision    : "accepted", "completed", "answered", or "rejected"
        priority    : ANN output (only if ANN ran)
        eligibility : Logic/KB output (only if Logic/KB ran)
        assignment  : CSP output (only if CSP ran)
        route       : Search output (only if Search ran)
        message     : Human-readable summary message

    Pipeline position: (all modules)  →  [Final Response]  →  USER
=============================================================================
"""


def build_final_response(request_obj: dict,
                         router_output: dict,
                         ann_output: dict = None,
                         logic_output: dict = None,
                         csp_output: dict = None,
                         search_output: dict = None) -> dict:
    """
    Assembles the final response by aggregating outputs from all modules.

    Decision logic:
        - "rejected"  : Logic/KB denied, or CSP failed, or preprocessing failed
        - "answered"  : Eligibility_Check completed
        - "completed" : Navigation_Only completed
        - "accepted"  : All other requests successfully processed

    Parameters:
        request_obj   (dict): The standardised request object.
        router_output (dict): Router decision output.
        ann_output    (dict): ANN module output (None if not used).
        logic_output  (dict): Logic/KB module output (None if not used).
        csp_output    (dict): CSP module output (None if not used).
        search_output (dict): Search module output (None if not used).

    Returns:
        dict: Standard final response object.
    """
    req_id = request_obj.get("request_id", "REQ???")
    rtype  = request_obj.get("request_type", "")
    name   = request_obj.get("name", "User")

    response = {
        "request_id" : req_id,
        "decision"   : "",
        "priority"   : {},
        "eligibility": {},
        "assignment" : {},
        "route"      : {},
        "message"    : "",
    }

    # ── Priority (ANN)
    if ann_output:
        response["priority"] = {
            "binary_priority": ann_output.get("binary_priority", ""),
            "final_priority" : ann_output.get("final_priority", ""),
            "confidence"     : ann_output.get("confidence", 0.0),
        }

    # ── Eligibility (Logic/KB)
    if logic_output:
        response["eligibility"] = {
            "allowed"     : logic_output.get("allowed", False),
            "entailed"    : logic_output.get("entailed", False),
            "explanation" : logic_output.get("explanation", ""),
        }

    # ── Assignment (CSP)
    if csp_output:
        response["assignment"] = {
            "room"    : csp_output.get("assigned_room", ""),
            "slot"    : csp_output.get("assigned_slot", None),
            "notes"   : csp_output.get("notes", ""),
        }

    # ── Route (Search)
    if search_output and search_output.get("found", False):
        response["route"] = {
            "algorithm": search_output.get("algorithm_used", ""),
            "path"     : search_output.get("path", []),
            "cost"     : search_output.get("cost", 0),
            "steps"    : search_output.get("steps", 0),
        }

    # ── Decision and Message
    if rtype == "Navigation_Only":
        if search_output and search_output.get("found", False):
            response["decision"] = "completed"
            path_str = " → ".join(search_output.get("path", []))
            response["message"] = (
                f"Route for {name}: {path_str}. "
                f"Algorithm: {search_output.get('algorithm_used')}. "
                f"Total cost: {search_output.get('cost')} units."
            )
        else:
            response["decision"] = "rejected"
            response["message"]  = f"No route found from {request_obj.get('current_location')} to {request_obj.get('destination')}."

    elif rtype == "Eligibility_Check":
        entailed = logic_output.get("entailed", False) if logic_output else False
        response["decision"] = "answered"
        response["message"]  = (
            logic_output.get("explanation", "Eligibility query processed.")
            if logic_output else "Logic module did not run."
        )

    elif rtype in ("Booking_or_Scheduling",
                   "Urgent_Service_Request",
                   "Full_Service_Request"):
        # Check for rejection at Logic/KB stage
        if logic_output and not logic_output.get("allowed", False):
            response["decision"] = "rejected"
            response["message"]  = (
                f"Request REJECTED. {name} is not authorised for "
                f"'{request_obj.get('category')}'. "
                f"Reason: {logic_output.get('explanation', 'Permission denied.')}"
            )
        # Check for rejection at CSP stage
        elif csp_output and csp_output.get("decision") == "rejected":
            response["decision"] = "rejected"
            response["message"]  = (
                f"Request REJECTED at scheduling stage. "
                f"{csp_output.get('notes', 'No feasible assignment found.')}"
            )
        else:
            response["decision"] = "accepted"
            # Build message
            parts = [f"Request ACCEPTED for {name}."]

            if response["priority"]:
                fp = response["priority"].get("final_priority", "")
                bp = response["priority"].get("binary_priority", "")
                parts.append(f"Priority: {fp} (binary: {bp}, "
                              f"confidence: {response['priority'].get('confidence', 0):.2f}).")

            if response["assignment"]:
                room = response["assignment"].get("room", "")
                slot = response["assignment"].get("slot", "")
                parts.append(f"Assigned: {room}, Slot {slot}.")

            if response["route"]:
                path_str = " → ".join(response["route"].get("path", []))
                parts.append(f"Route to {room}: {path_str} "
                              f"(cost: {response['route'].get('cost')} units).")

            response["message"] = " ".join(parts)

    return response


def print_response(response: dict):
    """
    Prints the final response in a formatted, human-readable style to the CLI.

    Parameters:
        response (dict): Final response dictionary from build_final_response().
    """
    sep  = "=" * 65
    sep2 = "-" * 65

    print(f"\n{sep}")
    print(f"  FINAL RESPONSE  |  Request ID: {response.get('request_id', '?')}")
    print(sep)

    decision = response.get("decision", "").upper()
    print(f"  Decision   : {decision}")

    # Priority block
    pri = response.get("priority", {})
    if pri:
        print(f"{sep2}")
        print(f"  [ANN PRIORITY]")
        print(f"  Binary   : {pri.get('binary_priority', '')}")
        print(f"  Final    : {pri.get('final_priority', '')}")
        print(f"  Confidence: {pri.get('confidence', 0):.2%}")

    # Eligibility block
    elig = response.get("eligibility", {})
    if elig:
        print(f"{sep2}")
        print(f"  [LOGIC / KB — ELIGIBILITY]")
        allowed_str = "ALLOWED" if elig.get("allowed") else "DENIED"
        print(f"  Result     : {allowed_str}")
        print(f"  Explanation: {elig.get('explanation', '')}")

    # Assignment block
    asgn = response.get("assignment", {})
    if asgn and asgn.get("room"):
        print(f"{sep2}")
        print(f"  [CSP ASSIGNMENT]")
        print(f"  Room  : {asgn.get('room', '')}")
        print(f"  Slot  : {asgn.get('slot', '')}")
        if asgn.get("notes"):
            print(f"  Notes : {asgn.get('notes', '')}")

    # Route block
    route = response.get("route", {})
    if route and route.get("path"):
        print(f"{sep2}")
        print(f"  [SEARCH — ROUTE]")
        print(f"  Algorithm : {route.get('algorithm', '')}")
        path_str = " → ".join(route.get("path", []))
        print(f"  Path      : {path_str}")
        print(f"  Cost      : {route.get('cost', 0)} units")
        print(f"  Steps     : {route.get('steps', 0)}")

    # Message
    print(f"{sep2}")
    print(f"  Summary: {response.get('message', '')}")
    print(f"{sep}\n")

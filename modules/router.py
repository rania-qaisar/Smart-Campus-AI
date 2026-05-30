"""
=============================================================================
Module  : Request Router
File    : router.py
Course  : AL2002 – Artificial Intelligence Lab (Spring 2026)
Project : Smart Campus AI Decision Support and Automation System
=============================================================================
Description:
    The Request Router is the control-flow brain of the Smart Campus AI
    platform. It receives a fully preprocessed request object from the
    Preprocessing module and decides which modules should run and in what
    order. The router does NOT solve the request itself; it only dispatches
    it to the correct pipeline.

    Supported pipelines:
        Navigation_Only        → [Search]
        Eligibility_Check      → [Logic_KB]
        Booking_or_Scheduling  → [Logic_KB, CSP, (Search)]
        Urgent_Service_Request → [ANN, Logic_KB, CSP, (Search)]
        Full_Service_Request   → [ANN, Logic_KB, CSP, Search]

    Pipeline position: PREPROCESSING  →  [ROUTER]  →  MODULE CHAIN
=============================================================================
"""

# ---------------------------------------------------------------------------
# Pipeline definitions per request type
# ---------------------------------------------------------------------------
PIPELINES = {
    "Navigation_Only": {
        "pipeline"    : ["Search"],
        "needs_ann"   : False,
        "needs_logic" : False,
        "needs_csp"   : False,
        "needs_search": True,
    },
    "Eligibility_Check": {
        "pipeline"    : ["Logic_KB"],
        "needs_ann"   : False,
        "needs_logic" : True,
        "needs_csp"   : False,
        "needs_search": False,
    },
    "Booking_or_Scheduling": {
        "pipeline"    : ["Logic_KB", "CSP"],
        "needs_ann"   : False,
        "needs_logic" : True,
        "needs_csp"   : True,
        "needs_search": False,   # optional; added dynamically if location given
    },
    "Urgent_Service_Request": {
        "pipeline"    : ["ANN", "Logic_KB", "CSP"],
        "needs_ann"   : True,
        "needs_logic" : True,
        "needs_csp"   : True,
        "needs_search": False,   # optional
    },
    "Full_Service_Request": {
        "pipeline"    : ["ANN", "Logic_KB", "CSP", "Search"],
        "needs_ann"   : True,
        "needs_logic" : True,
        "needs_csp"   : True,
        "needs_search": True,
    },
}


def route_request(request_obj: dict) -> dict:
    """
    Determines the correct processing pipeline for the given request.

    Reads the request_type from the preprocessed request object and
    returns a router output dictionary that specifies:
        - selected_pipeline  : ordered list of module names to execute
        - needs_ann          : whether ANN module should run
        - needs_logic        : whether Logic/KB module should run
        - needs_csp          : whether CSP scheduler should run
        - needs_search       : whether Search module should run

    For Booking_or_Scheduling and Urgent_Service_Request, Search is added
    dynamically if current_location is provided (optional route guidance).

    Parameters:
        request_obj (dict): Standardised, preprocessed request dictionary.

    Returns:
        dict: Router output object with pipeline decision and flags.
              Returns an error dict if request_type is unrecognised.
    """
    rtype = request_obj.get("request_type", "")

    if rtype not in PIPELINES:
        return {
            "request_id"      : request_obj.get("request_id", ""),
            "selected_pipeline": [],
            "needs_ann"        : False,
            "needs_logic"      : False,
            "needs_csp"        : False,
            "needs_search"     : False,
            "error"            : f"Unrecognised request_type: '{rtype}'"
        }

    config = PIPELINES[rtype].copy()
    pipeline = list(config["pipeline"])  # mutable copy

    # Dynamically add optional Search for Booking and Urgent if location provided
    if rtype in ("Booking_or_Scheduling", "Urgent_Service_Request"):
        if request_obj.get("current_location", "").strip():
            if "Search" not in pipeline:
                pipeline.append("Search")
            config["needs_search"] = True

    router_output = {
        "request_id"      : request_obj.get("request_id", ""),
        "request_type"    : rtype,
        "selected_pipeline": pipeline,
        "needs_ann"        : config["needs_ann"],
        "needs_logic"      : config["needs_logic"],
        "needs_csp"        : config["needs_csp"],
        "needs_search"     : config["needs_search"],
    }

    return router_output


def describe_pipeline(router_output: dict) -> str:
    """
    Returns a human-readable description of the selected pipeline
    for display in the CLI and the final response.

    Parameters:
        router_output (dict): Output dictionary from route_request().

    Returns:
        str: Formatted pipeline description string.
    """
    pipe = router_output.get("selected_pipeline", [])
    if not pipe:
        return "No pipeline selected."
    arrow = " → ".join(pipe)
    return f"Pipeline: Preprocessing → Router → {arrow} → Final Response"

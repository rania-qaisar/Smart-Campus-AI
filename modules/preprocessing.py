"""
=============================================================================
Module  : Input & Preprocessing
File    : preprocessing.py
Course  : AL2002 – Artificial Intelligence Lab (Spring 2026)
Project : Smart Campus AI Decision Support and Automation System
=============================================================================
Description:
    This module is the first stage of the Smart Campus AI pipeline.
    It receives raw CLI input, validates required fields, normalises values
    to project-standard naming conventions, builds the standard request
    object (Python dict), and prepares any module-specific sub-objects
    (ANN feature vector fields, logic query, search source/destination,
    CSP flags) so that the Request Router receives a fully clean object.

    Pipeline position: USER INPUT  →  [PREPROCESSING]  →  ROUTER
=============================================================================
"""

import uuid

# ---------------------------------------------------------------------------
# Encoding tables (must stay consistent with ann_module.py)
# ---------------------------------------------------------------------------
ROLE_ENCODING = {
    "student": 0,
    "instructor": 1,
    "staff": 2
}

REQUEST_TYPE_ENCODING = {
    "AI_Lab_Support": 0,
    "Viva_Scheduling": 1,
    "Access_Request": 2,
    "Maintenance": 3,
    "Emergency_Help": 4
}

# ---------------------------------------------------------------------------
# Valid campus node names (used for location validation)
# ---------------------------------------------------------------------------
VALID_LOCATIONS = {
    "Main_Gate", "Parking", "Admin_Block", "Student_Services",
    "Exam_Hall", "Seminar_Room", "AI_Lab", "Science_Block",
    "Library", "Cafeteria", "Hostel", "Medical_Center", "Bus_Stop"
}

# Normalisation map: lowercase user text → project-standard name
LOCATION_NORMALISE = {
    "main gate": "Main_Gate",
    "maingate": "Main_Gate",
    "parking": "Parking",
    "admin block": "Admin_Block",
    "admin_block": "Admin_Block",
    "adminblock": "Admin_Block",
    "student services": "Student_Services",
    "student_services": "Student_Services",
    "exam hall": "Exam_Hall",
    "exam_hall": "Exam_Hall",
    "seminar room": "Seminar_Room",
    "seminar_room": "Seminar_Room",
    "ai lab": "AI_Lab",
    "ai_lab": "AI_Lab",
    "ailab": "AI_Lab",
    "science block": "Science_Block",
    "science_block": "Science_Block",
    "library": "Library",
    "cafeteria": "Cafeteria",
    "hostel": "Hostel",
    "medical center": "Medical_Center",
    "medical_center": "Medical_Center",
    "bus stop": "Bus_Stop",
    "bus_stop": "Bus_Stop",
}

VALID_REQUEST_TYPES = [
    "Navigation_Only",
    "Eligibility_Check",
    "Booking_or_Scheduling",
    "Urgent_Service_Request",
    "Full_Service_Request"
]

VALID_CATEGORIES = [
    "AI_Lab_Support",
    "Viva_Scheduling",
    "Access_Request",
    "Maintenance",
    "Emergency_Help"
]

# Approximate graph distances from each node to AI_Lab (used as ANN distance feature)
DISTANCE_TO_AI_LAB = {
    "Main_Gate": 5,
    "Parking": 4,
    "Admin_Block": 4,
    "Student_Services": 3,
    "Exam_Hall": 3,
    "Seminar_Room": 2,
    "AI_Lab": 0,
    "Science_Block": 1,
    "Library": 2,
    "Cafeteria": 3,
    "Hostel": 4,
    "Medical_Center": 5,
    "Bus_Stop": 6
}


# ---------------------------------------------------------------------------
# Helper: normalise a location string
# ---------------------------------------------------------------------------
def normalise_location(raw: str) -> str:
    """
    Converts free-text location input into the project-standard node name.
    Returns the normalised string, or the original if no mapping found.

    Parameters:
        raw (str): Raw location string entered by user.

    Returns:
        str: Standard project location name or original string.
    """
    key = raw.strip().lower().replace("-", " ").replace("_", " ")
    # Direct key match
    if key in LOCATION_NORMALISE:
        return LOCATION_NORMALISE[key]
    # Try with underscores preserved
    key2 = raw.strip().lower()
    if key2 in LOCATION_NORMALISE:
        return LOCATION_NORMALISE[key2]
    # Capitalise each word as last resort
    return raw.strip().replace(" ", "_").title()


# ---------------------------------------------------------------------------
# Helper: normalise a request type string
# ---------------------------------------------------------------------------
def normalise_request_type(raw: str) -> str:
    """
    Converts user-entered request type text to project-standard naming.

    Parameters:
        raw (str): Raw request type string.

    Returns:
        str: Standardised request type name.
    """
    mapping = {
        "navigation_only": "Navigation_Only",
        "navigation only": "Navigation_Only",
        "1": "Navigation_Only",
        "eligibility_check": "Eligibility_Check",
        "eligibility check": "Eligibility_Check",
        "2": "Eligibility_Check",
        "booking_or_scheduling": "Booking_or_Scheduling",
        "booking or scheduling": "Booking_or_Scheduling",
        "booking": "Booking_or_Scheduling",
        "scheduling": "Booking_or_Scheduling",
        "3": "Booking_or_Scheduling",
        "urgent_service_request": "Urgent_Service_Request",
        "urgent service request": "Urgent_Service_Request",
        "urgent": "Urgent_Service_Request",
        "4": "Urgent_Service_Request",
        "full_service_request": "Full_Service_Request",
        "full service request": "Full_Service_Request",
        "full": "Full_Service_Request",
        "5": "Full_Service_Request",
    }
    return mapping.get(raw.strip().lower(), raw.strip())


# ---------------------------------------------------------------------------
# Helper: normalise a category string
# ---------------------------------------------------------------------------
def normalise_category(raw: str) -> str:
    """
    Converts user category input to project-standard naming.

    Parameters:
        raw (str): Raw category string.

    Returns:
        str: Standardised category name.
    """
    mapping = {
        "ai lab support": "AI_Lab_Support",
        "ai_lab_support": "AI_Lab_Support",
        "ailab": "AI_Lab_Support",
        "1": "AI_Lab_Support",
        "viva scheduling": "Viva_Scheduling",
        "viva_scheduling": "Viva_Scheduling",
        "viva": "Viva_Scheduling",
        "2": "Viva_Scheduling",
        "access request": "Access_Request",
        "access_request": "Access_Request",
        "access": "Access_Request",
        "3": "Access_Request",
        "maintenance": "Maintenance",
        "4": "Maintenance",
        "emergency help": "Emergency_Help",
        "emergency_help": "Emergency_Help",
        "emergency": "Emergency_Help",
        "5": "Emergency_Help",
    }
    return mapping.get(raw.strip().lower(), raw.strip())


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------
def validate_request(raw: dict) -> tuple:
    """
    Validates all fields of the raw request dictionary.
    Returns (is_valid: bool, error_message: str).

    Checks performed:
        - Required base fields present (name, role, request_type)
        - role is one of the allowed roles
        - request_type is valid
        - For Navigation_Only: current_location and destination required
        - For Eligibility_Check: query required
        - For Booking/Urgent/Full: category, current_location required
        - preferred_slot in {1,2,3,4} if provided
        - severity, time_sensitivity, crowd_level numeric in 1-10

    Parameters:
        raw (dict): Raw request dictionary from CLI input collection.

    Returns:
        tuple: (bool, str) — True if valid with empty message, else False with reason.
    """
    # -- Base required fields --
    for field in ["name", "role", "request_type"]:
        if not raw.get(field, "").strip():
            return False, f"Required field '{field}' is missing or empty."

    # -- Role validation --
    if raw["role"].lower() not in ROLE_ENCODING:
        return False, f"Invalid role '{raw['role']}'. Must be: student, instructor, or staff."

    # -- Request type validation --
    rtype = raw["request_type"]
    if rtype not in VALID_REQUEST_TYPES:
        return False, f"Invalid request type '{rtype}'."

    # -- Type-specific field checks --
    if rtype == "Navigation_Only":
        if not raw.get("current_location", "").strip():
            return False, "Navigation_Only requires 'current_location'."
        if not raw.get("destination", "").strip():
            return False, "Navigation_Only requires 'destination'."
        if raw["current_location"] not in VALID_LOCATIONS:
            return False, f"Unknown location '{raw['current_location']}'."
        if raw["destination"] not in VALID_LOCATIONS:
            return False, f"Unknown destination '{raw['destination']}'."
        if raw["current_location"] == raw["destination"]:
            return False, "Source and destination cannot be the same location."

    elif rtype == "Eligibility_Check":
        if not raw.get("query", "").strip():
            return False, "Eligibility_Check requires a 'query' field."

    elif rtype in ("Booking_or_Scheduling", "Urgent_Service_Request", "Full_Service_Request"):
        if not raw.get("category", "").strip():
            return False, f"{rtype} requires 'category'."
        if raw["category"] not in VALID_CATEGORIES:
            return False, f"Unknown category '{raw['category']}'."
        if not raw.get("current_location", "").strip():
            return False, f"{rtype} requires 'current_location'."
        if raw["current_location"] not in VALID_LOCATIONS:
            return False, f"Unknown location '{raw['current_location']}'."

        # Slot validation
        if raw.get("preferred_slot") is not None:
            try:
                slot = int(raw["preferred_slot"])
                if slot not in (1, 2, 3, 4):
                    return False, "preferred_slot must be between 1 and 4."
            except (ValueError, TypeError):
                return False, "preferred_slot must be a numeric value (1-4)."

        # Numeric range checks for urgent/full
        if rtype in ("Urgent_Service_Request", "Full_Service_Request"):
            for field in ["severity", "time_sensitivity", "crowd_level"]:
                val = raw.get(field)
                if val is None:
                    return False, f"{rtype} requires '{field}'."
                try:
                    num = float(val)
                    if not (1 <= num <= 10):
                        return False, f"'{field}' must be between 1 and 10."
                except (ValueError, TypeError):
                    return False, f"'{field}' must be a numeric value."

    return True, ""


# ---------------------------------------------------------------------------
# Build ANN feature vector
# ---------------------------------------------------------------------------
def build_ann_feature_vector(req: dict) -> list:
    """
    Constructs the 7-element numeric feature vector required by the ANN module.
    Feature order (fixed): [Role, RequestType, Severity, TimeSensitivity,
                             CrowdLevel, Distance, Eligibility]

    Parameters:
        req (dict): Standardised request object after normalisation.

    Returns:
        list: 7-element numeric feature vector [int/float].
    """
    role_enc = ROLE_ENCODING.get(req.get("role", "student"), 0)
    cat_enc  = REQUEST_TYPE_ENCODING.get(req.get("category", "AI_Lab_Support"), 0)
    severity = float(req.get("severity", 5))
    time_sen = float(req.get("time_sensitivity", 5))
    crowd    = float(req.get("crowd_level", 5))
    distance = float(DISTANCE_TO_AI_LAB.get(req.get("current_location", "Hostel"), 4))
    eligible = 1 if req.get("eligibility_claim", True) else 0

    return [role_enc, cat_enc, severity, time_sen, crowd, distance, eligible]


# ---------------------------------------------------------------------------
# Main preprocessing function (entry point for the pipeline)
# ---------------------------------------------------------------------------
def preprocess_request(raw: dict) -> tuple:
    """
    Main entry point for the preprocessing stage.
    Validates, normalises, and enriches the raw CLI input dict into a
    complete standardised request object ready for the Request Router.

    Steps performed:
        1. Normalise all string fields
        2. Validate fields
        3. Build standard request object
        4. Attach module-readiness flags (needs_ann, needs_logic, etc.)
        5. Build ANN feature vector if needed
        6. Build logic query stub if needed

    Parameters:
        raw (dict): Raw input collected from CLI (string values).

    Returns:
        tuple: (success: bool, result: dict)
            - If success=True,  result is the fully preprocessed request object.
            - If success=False, result is {"error": "<message>"}.
    """
    # Step 1 — Normalise string fields
    raw["role"]         = raw.get("role", "").strip().lower()
    raw["request_type"] = normalise_request_type(raw.get("request_type", ""))
    raw["category"]     = normalise_category(raw.get("category", ""))

    if raw.get("current_location"):
        raw["current_location"] = normalise_location(raw["current_location"])
    if raw.get("destination"):
        raw["destination"] = normalise_location(raw["destination"])

    # Convert numeric fields to proper types
    for field in ["severity", "time_sensitivity", "crowd_level"]:
        if raw.get(field) not in (None, ""):
            try:
                raw[field] = float(raw[field])
            except (ValueError, TypeError):
                pass

    if raw.get("preferred_slot") not in (None, ""):
        try:
            raw["preferred_slot"] = int(raw["preferred_slot"])
        except (ValueError, TypeError):
            pass

    # Step 2 — Validate
    valid, error_msg = validate_request(raw)
    if not valid:
        return False, {"error": error_msg}

    # Step 3 — Build standard request object
    rtype = raw["request_type"]
    request_obj = {
        "request_id"      : f"REQ{str(uuid.uuid4().int)[:5]}",
        "name"            : raw.get("name", "").strip(),
        "role"            : raw["role"],
        "request_type"    : rtype,
        "category"        : raw.get("category", ""),
        "current_location": raw.get("current_location", ""),
        "destination"     : raw.get("destination", ""),
        "preferred_slot"  : raw.get("preferred_slot", None),
        "severity"        : raw.get("severity", 0),
        "time_sensitivity": raw.get("time_sensitivity", 0),
        "crowd_level"     : raw.get("crowd_level", 0),
        "group_id"        : raw.get("group_id", ""),
        "query"           : raw.get("query", ""),
        "eligibility_claim": True,
        "description_note": raw.get("description_note", ""),
    }

    # Step 4 — Module-readiness flags
    request_obj["needs_ann"]    = rtype in ("Urgent_Service_Request", "Full_Service_Request")
    request_obj["needs_logic"]  = rtype in ("Eligibility_Check", "Booking_or_Scheduling",
                                             "Urgent_Service_Request", "Full_Service_Request")
    request_obj["needs_csp"]    = rtype in ("Booking_or_Scheduling",
                                             "Urgent_Service_Request", "Full_Service_Request")
    request_obj["needs_search"] = rtype in ("Navigation_Only", "Full_Service_Request")

    # Step 5 — ANN feature vector (only if ANN is needed)
    if request_obj["needs_ann"]:
        request_obj["ann_feature_vector"] = build_ann_feature_vector(request_obj)

    # Step 6 — Auto-build logic query for eligibility if not user-provided
    if rtype == "Eligibility_Check" and not request_obj["query"].strip():
        # Build a default query from name + category
        cat = request_obj["category"] or "AI_Lab"
        request_obj["query"] = f"Eligible({request_obj['name']}, {cat})"

    return True, request_obj

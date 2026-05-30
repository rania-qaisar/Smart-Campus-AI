# Smart Campus AI Decision Support and Automation System

A modular CLI-based AI platform that processes campus service requests through an intelligent pipeline of ANN priority prediction, Logic/Knowledge Base reasoning, CSP scheduling, and Search-based navigation — producing one unified final response.

## Features

- 5 request types with automatic pipeline routing
- ANN module: Perceptron (binary) + MLP (multiclass priority prediction)
- Logic/KB module: Forward chaining with First-Order Logic rules
- CSP Scheduler: Backtracking-based conflict-free slot and room assignment
- Search & Navigation: 9 algorithms on a predefined campus graph
- Demo mode: Runs all 5 request types automatically
- Algorithm Comparison mode: Benchmarks all 9 search algorithms side by side

## Requirements

- Python 3.x
- No external libraries required (uses only Python standard library)

## How to Run

1. Clone the repository or download the files
2. Open terminal and navigate to the folder
3. Run:
   python main.py
## How to Use

- Select **Submit a New Request** to enter fields step by step via CLI
- Select **Run Demo** to automatically run all 5 request types with sample data
- Select **Search Algorithm Comparison** to benchmark all 9 algorithms on a chosen route
- All input is structured — the CLI guides you through each required field

## Request Types

| Request Type | Pipeline |
|---|---|
| Navigation_Only | Search → Final Response |
| Eligibility_Check | Logic/KB → Final Response |
| Booking_or_Scheduling | Logic/KB → CSP → Final Response |
| Urgent_Service_Request | ANN → Logic/KB → CSP → Final Response |
| Full_Service_Request | ANN → Logic/KB → CSP → Search → Final Response |

## Search Algorithms

| Algorithm | Type | Function |
|---|---|---|
| BFS | Operational | Shortest path on unweighted graph |
| A* | Operational | f(n) = g(n) + h(n) |
| UCS | Operational | f(n) = g(n) |
| DFS | Comparison | Depth-first exploration |
| DLS | Comparison | Depth-limited search |
| IDS | Comparison | Iterative deepening |
| Bidirectional BFS | Comparison | Meet-in-the-middle |
| Greedy BFS | Comparison | f(n) = h(n) |
| RBFS | Comparison | Memory-efficient best-first |

## Heuristic

| Heuristic | Formula |
|---|---|
| Euclidean | √((x₁−x₂)² + (y₁−y₂)²) |

## Campus Locations

| Location |
|---|
| Main_Gate, Parking, Admin_Block, Student_Services, Exam_Hall |
| Seminar_Room, AI_Lab, Science_Block, Library, Cafeteria |
| Hostel, Medical_Center, Bus_Stop |

## Author
**Rania Qaisar**

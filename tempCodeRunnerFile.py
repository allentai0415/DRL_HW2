from flask import Flask, render_template, request, jsonify, session
import numpy as np

app = Flask(__name__)
app.secret_key = "secret"

GAMMA = 0.9
THRESHOLD = 1e-4
MAX_ITER = 100
REWARD = -0.2
GOAL_REWARD = 1

directions = ["↑", "↓", "←", "→"]
actions = {"↑": (-1, 0), "↓": (1, 0), "←": (0, -1), "→": (0, 1)}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        n = int(request.form.get("n"))
        if n < 3 or n > 7:
            return "Grid size must be between 3 and 7."
        session["n"] = n
        session["obstacles"] = []
        session["start"] = None
        session["goal"] = None
        return render_template("select_obstacles.html", n=n)
    return render_template("index.html")

@app.route("/submit_obstacles", methods=["POST"])
def submit_obstacles():
    data = request.json
    n = session.get("n")
    if len(data["obstacles"]) > n * n - 2:
        return jsonify({"status": "error", "message": "Too many obstacles."}), 400
    session["obstacles"] = [tuple(o) for o in data["obstacles"]]
    session["start"] = tuple(data["start"]) if data["start"] else None
    session["goal"] = tuple(data["goal"]) if data["goal"] else None
    return jsonify({"status": "success"})

def value_iteration(n, obstacles, goal):
    V_steps = []
    policy_steps = []

    def in_bounds(i, j): return 0 <= i < n and 0 <= j < n

    V = np.zeros((n, n))
    policy = np.full((n, n), "")

    for (i, j) in obstacles:
        V[i][j] = None
        policy[i][j] = "■"

    if goal:
        V[goal[0]][goal[1]] = GOAL_REWARD
        policy[goal[0]][goal[1]] = 'G'

    for _ in range(MAX_ITER):
        delta = 0
        new_V = np.copy(V)
        new_policy = np.copy(policy)

        for i in range(n):
            for j in range(n):
                if (i, j) in obstacles or (i, j) == goal:
                    continue

                best_value = float('-inf')
                best_action = None

                for a in directions:
                    di, dj = actions[a]
                    ni, nj = i + di, j + dj
                    if not in_bounds(ni, nj) or (ni, nj) in obstacles:
                        next_val = 0
                    else:
                        next_val = V[ni][nj] if V[ni][nj] is not None else 0
                    value = REWARD + GAMMA * next_val

                    if value > best_value:
                        best_value = value
                        best_action = a

                new_V[i][j] = round(best_value, 2)
                new_policy[i][j] = best_action
                delta = max(delta, abs(best_value - (V[i][j] if V[i][j] is not None else 0)))

        V = new_V
        policy = new_policy
        V_steps.append(np.copy(V))
        policy_steps.append(np.copy(policy))

        if delta < THRESHOLD:
            break

    return V_steps, policy_steps

def trace_path(policy, start, goal):
    path = []
    current = start
    visited = set()

    while current != goal and current not in visited:
        visited.add(current)
        path.append(current)

        if (current[0] < 0 or current[0] >= len(policy) or
            current[1] < 0 or current[1] >= len(policy[0])):
            break

        dir = policy[current[0]][current[1]]
        if dir not in actions:
            break

        di, dj = actions[dir]
        next_i, next_j = current[0] + di, current[1] + dj

        if next_i < 0 or next_i >= len(policy) or next_j < 0 or next_j >= len(policy[0]):
            break

        current = (next_i, next_j)

    if current == goal:
        path.append(goal)

    return path

@app.route("/view_matrices")
def view_matrices():
    n = session.get("n", 5)
    obstacles = session.get("obstacles", [])
    start = session.get("start")
    goal = session.get("goal")

    value_steps, policy_steps = value_iteration(n, obstacles, goal)
    final_V = value_steps[-1]
    final_policy = policy_steps[-1]
    path = trace_path(final_policy, start, goal) if start and goal else []

    steps = []
    for step_idx in range(len(value_steps)):
        v_matrix = []
        p_matrix = []
        for i in range(n):
            v_row = []
            p_row = []
            for j in range(n):
                val = value_steps[step_idx][i][j]
                pol = policy_steps[step_idx][i][j]
                v_row.append(None if val is None else round(val, 2))
                p_row.append(pol)
            v_matrix.append(v_row)
            p_matrix.append(p_row)
        steps.append({"value": v_matrix, "policy": p_matrix})

    # 將 obstacles 中的 tuple 轉換為 list，以符合 JSON 格式
    obstacles_list = [list(pos) for pos in obstacles]

    return render_template(
        "view_matrices.html",
        n=n,
        steps=steps,
        path=path,
        start=start,
        goal=goal,
        obstacles=obstacles_list
    )

if __name__ == "__main__":
    app.run(debug=True)

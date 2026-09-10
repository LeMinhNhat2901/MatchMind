"""
Sample data fixtures for tests and demos.

Provides deterministic synthetic match states without requiring StatsBomb API keys.
"""
from __future__ import annotations

from matchmind.schema.match_state import BallState, MatchState, PlayerState


def make_sample_match_state(
    match_id: str = "demo_001",
    minute: int = 62,
    score_home: int = 0,
    score_away: int = 1,
) -> tuple[MatchState, int]:
    """
    Create a realistic synthetic match snapshot.

    Scenario: Home team (red) trailing 0-1 at minute 62.
    Player 10 (attacking midfielder) in the half-space with numerical overload.
    """
    state = MatchState(
        match_id=match_id,
        timestamp=float(minute * 60 + 15),
        score_home=score_home,
        score_away=score_away,
        possession="home",
        phase_of_play="open_play",
        ball=BallState(x=68.0, y=42.0),
        source="synthetic",
        players=[
            # Home team players (attacking left→right)
            PlayerState(id=1,  team="home", role="goalkeeper",           x=5.0,  y=34.0),
            PlayerState(id=2,  team="home", role="right_back",           x=32.0, y=60.0, vx=0.5, vy=0.2),
            PlayerState(id=3,  team="home", role="center_back",          x=28.0, y=40.0),
            PlayerState(id=4,  team="home", role="center_back",          x=28.0, y=28.0),
            PlayerState(id=5,  team="home", role="left_back",            x=30.0, y=10.0, vx=1.0, vy=0.0),
            PlayerState(id=6,  team="home", role="defensive_midfielder", x=45.0, y=34.0),
            PlayerState(id=7,  team="home", role="central_midfielder",   x=55.0, y=20.0, vx=0.5, vy=0.3),
            PlayerState(id=8,  team="home", role="central_midfielder",   x=52.0, y=48.0),
            PlayerState(id=9,  team="home", role="right_winger",         x=72.0, y=60.0, vx=1.0, vy=0.0),
            PlayerState(id=10, team="home", role="attacking_midfielder", x=68.0, y=42.0),  # FOCUS PLAYER
            PlayerState(id=11, team="home", role="striker",              x=78.0, y=36.0, vx=0.8, vy=0.2),

            # Away team players (defending right→left)
            PlayerState(id=21, team="away", role="goalkeeper",           x=100.0, y=34.0),
            PlayerState(id=22, team="away", role="right_back",           x=80.0,  y=55.0),
            PlayerState(id=23, team="away", role="center_back",          x=82.0,  y=42.0),
            PlayerState(id=24, team="away", role="center_back",          x=82.0,  y=28.0),
            PlayerState(id=25, team="away", role="left_back",            x=78.0,  y=15.0),
            PlayerState(id=26, team="away", role="defensive_midfielder", x=70.0,  y=34.0),
            PlayerState(id=27, team="away", role="central_midfielder",   x=65.0,  y=50.0),
            PlayerState(id=28, team="away", role="central_midfielder",   x=62.0,  y=22.0),
            PlayerState(id=29, team="away", role="right_winger",         x=50.0,  y=60.0),
            PlayerState(id=30, team="away", role="left_winger",          x=52.0,  y=12.0),
            PlayerState(id=31, team="away", role="striker",              x=55.0,  y=34.0),
        ],
    )
    return state, 10  # focus player ID = 10


def make_counter_attack_state() -> tuple[MatchState, int]:
    """Scenario: Counter-attack in progress, attacking player with space."""
    state = MatchState(
        match_id="demo_002",
        timestamp=75 * 60.0,
        score_home=1,
        score_away=1,
        possession="home",
        phase_of_play="attacking_transition",
        ball=BallState(x=55.0, y=34.0),
        source="synthetic",
        has_velocity=True,  # runners below carry real velocity
        players=[
            PlayerState(id=1,  team="home", role="goalkeeper",   x=5.0,  y=34.0),
            PlayerState(id=2,  team="home", role="center_back",  x=30.0, y=40.0),
            PlayerState(id=3,  team="home", role="center_back",  x=30.0, y=28.0),
            PlayerState(id=10, team="home", role="central_midfielder", x=55.0, y=34.0),  # FOCUS
            PlayerState(id=9,  team="home", role="right_winger", x=65.0, y=55.0, vx=3.0, vy=0.0),
            PlayerState(id=11, team="home", role="striker",      x=68.0, y=30.0, vx=4.0, vy=0.0),

            # Only 2 defenders back (counter-attack situation)
            PlayerState(id=21, team="away", role="goalkeeper",   x=100.0, y=34.0),
            PlayerState(id=22, team="away", role="center_back",  x=75.0,  y=38.0),
            PlayerState(id=23, team="away", role="center_back",  x=75.0,  y=30.0),
            PlayerState(id=24, team="away", role="right_back",   x=80.0,  y=52.0),
            PlayerState(id=25, team="away", role="left_back",    x=78.0,  y=16.0),
            PlayerState(id=26, team="away", role="defensive_midfielder", x=60.0, y=34.0),
        ],
    )
    return state, 10


def make_sample_test_cases(n: int = 10) -> list[dict]:
    """Generate synthetic test cases for evaluation when StatsBomb is unavailable."""
    cases = []

    state1, pid1 = make_sample_match_state()
    cases.append({
        "snapshot": state1,
        "focus_player_id": pid1,
        "question": f"What should player {pid1} do in this situation?",
        "ground_truth_action": "pass",
        "ground_truth_event_type": "Pass",
    })

    state2, pid2 = make_counter_attack_state()
    cases.append({
        "snapshot": state2,
        "focus_player_id": pid2,
        "question": f"Counter-attack situation — what should player {pid2} do?",
        "ground_truth_action": "pass",
        "ground_truth_event_type": "Pass",
    })

    # Generate variations
    import random
    random.seed(42)
    for i in range(min(n - 2, 20)):
        # Perturb player positions slightly
        state, pid = make_sample_match_state(
            match_id=f"demo_{i+10:03d}",
            minute=random.randint(10, 90),
            score_home=random.randint(0, 3),
            score_away=random.randint(0, 3),
        )
        # Randomly perturb ball position
        import copy
        perturbed_state = state.model_copy(
            update={
                "ball": BallState(
                    x=max(0, min(105, state.ball.x + random.uniform(-10, 10))),
                    y=max(0, min(68, state.ball.y + random.uniform(-8, 8))),
                )
            }
        )
        cases.append({
            "snapshot": perturbed_state,
            "focus_player_id": pid,
            "question": f"What should player {pid} do in minute {perturbed_state.minute}?",
            "ground_truth_action": random.choice(["pass", "carry", "dribble"]),
            "ground_truth_event_type": random.choice(["Pass", "Carry", "Dribble"]),
        })

    return cases[:n]

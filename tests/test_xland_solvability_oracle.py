"""A classical privileged oracle establishes reachability, never policy skill."""
import numpy as np
import pytest

from bagi.experiments.xland_solvability_oracle import shortest_route_to_face_object


def room():
    grid = np.zeros((7, 7, 2), dtype=np.uint8)
    grid[:, :, 0] = 1  # floor
    grid[0, :, 0] = 2  # boundary wall
    grid[-1, :, 0] = 2
    grid[:, 0, 0] = 2
    grid[:, -1, 0] = 2
    grid[3, 3, 0] = 4  # obstacle (square)
    return grid


def test_shortest_route_faces_object_before_activation():
    actions, expansions = shortest_route_to_face_object(
        room(), (5, 3), 0, (3, 3), frozenset((1, 6, 10))
    )
    assert actions == [0]
    assert expansions >= 1


def test_heading_is_part_of_navigation_state():
    actions, _ = shortest_route_to_face_object(
        room(), (4, 3), 1, (3, 3), frozenset((1, 6, 10))
    )
    assert actions == [2]
    assert shortest_route_to_face_object(
        room(), (4, 3), 0, (3, 3), frozenset((1, 6, 10))
    )[0] == []


def test_blocked_object_can_be_unreachable():
    grid = room()
    grid[2:5, 2:5, 0] = 2
    grid[3, 3, 0] = 4
    with pytest.raises(ValueError, match="No reachable"):
        shortest_route_to_face_object(
            grid, (1, 1), 0, (3, 3), frozenset((1, 6, 10))
        )


def test_navigation_does_not_step_onto_object():
    actions, _ = shortest_route_to_face_object(
        room(), (2, 2), 2, (3, 3), frozenset((1, 6, 10))
    )
    assert actions
    assert all(action in (0, 1, 2) for action in actions)

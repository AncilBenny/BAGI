"""M23 preregistered study contracts; no environment or training logic.

The native MiniGrid evaluator will supply witnessed causal events separately
from the agent. This module validates the research protocol and applies an
outcome rule; it does not create or assert physical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

SOURCE_TASKS = (
    "MiniGrid-DoorKey-6x6-v0",
    "MiniGrid-Fetch-5x5-N2-v0",
    "MiniGrid-PutNear-6x6-N2-v0",
)
TARGET_TASK = "MiniGrid-UnlockPickup-v0"
REPLICATION_TASK = "MiniGrid-BlockedUnlockPickup-v0"
MODEL_SEEDS = (941, 947, 953, 967, 971, 977, 983, 991)
ARMS = (
    "random",
    "source_bc",
    "source_replay_bc",
    "source_world_model_planner",
    "target_scratch",
)
TRAIN_LAYOUTS = {
    "doorkey": tuple(range(431001, 431129)),
    "fetch": tuple(range(441001, 441129)),
    "putnear": tuple(range(461001, 461129)),
    "unlockpickup": tuple(range(451001, 451129)),
}
MONITOR_LAYOUTS = {
    "doorkey": tuple(range(432001, 432017)),
    "fetch": tuple(range(442001, 442017)),
    "putnear": tuple(range(462001, 462017)),
    "unlockpickup": tuple(range(452001, 452017)),
}
FINAL_LAYOUTS = {
    "doorkey": tuple(range(433001, 433033)),
    "fetch": tuple(range(443001, 443033)),
    "putnear": tuple(range(463001, 463033)),
    "unlockpickup": tuple(range(453001, 453033)),
}



BALANCE_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(472001, 472017)),
    "fetch": tuple(range(482001, 482017)),
    "putnear": tuple(range(492001, 492017)),
}


MEMORY_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(502001, 502017)),
    "fetch": tuple(range(512001, 512017)),
    "putnear": tuple(range(522001, 522017)),
}


EFFECT_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(532001, 532017)),
    "fetch": tuple(range(542001, 542017)),
    "putnear": tuple(range(552001, 552017)),
}


PROBE_TRAIN_LAYOUTS = {
    "doorkey": tuple(range(571001, 571129)),
    "fetch": tuple(range(581001, 581129)),
    "putnear": tuple(range(591001, 591129)),
}
PROBE_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(572001, 572033)),
    "fetch": tuple(range(582001, 582033)),
    "putnear": tuple(range(592001, 592033)),
}

RULE_SHIFT_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(602001, 602033)),
    "fetch": tuple(range(612001, 612033)),
    "putnear": tuple(range(622001, 622033)),
}

ACTION_TRAIN_LAYOUTS = {
    "doorkey": tuple(range(651001, 651065)),
    "fetch": tuple(range(661001, 661065)),
}
ACTION_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(652001, 652033)),
    "fetch": tuple(range(662001, 662033)),
}

SKILL_TRAIN_LAYOUTS = {
    "doorkey": tuple(range(681001, 681065)),
    "fetch": tuple(range(691001, 691065)),
}
SKILL_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(682001, 682065)),
    "fetch": tuple(range(692001, 692065)),
}

BC_TRAIN_LAYOUTS = {
    "doorkey": tuple(range(711001, 711097)),
    "fetch": tuple(range(721001, 721097)),
}
BC_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(712001, 712049)),
    "fetch": tuple(range(722001, 722049)),
}

VISION_TRAIN_LAYOUTS = {
    "doorkey": tuple(range(731001, 731097)),
    "fetch": tuple(range(741001, 741097)),
}
VISION_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(732001, 732041)),
    "fetch": tuple(range(742001, 742041)),
}

CORRECT_TRAIN_LAYOUTS = {
    "doorkey": tuple(range(751001, 751097)),
    "fetch": tuple(range(761001, 761097)),
}
CORRECT_ACQUIRE_LAYOUTS = {
    "doorkey": tuple(range(771001, 771025)),
    "fetch": tuple(range(781001, 781025)),
}
CORRECT_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(752001, 752041)),
    "fetch": tuple(range(762001, 762041)),
}

INTERVENE_TRAIN_LAYOUTS = {
    "doorkey": tuple(range(791001, 791097)),
    "fetch": tuple(range(801001, 801097)),
}
INTERVENE_ACQUIRE_LAYOUTS = {
    "doorkey": tuple(range(811001, 811049)),
    "fetch": tuple(range(821001, 821049)),
}
INTERVENE_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(792001, 792041)),
    "fetch": tuple(range(802001, 802041)),
}

RECURRENT_TRAIN_LAYOUTS = {
    "doorkey": tuple(range(831001, 831097)),
    "fetch": tuple(range(841001, 841097)),
}
RECURRENT_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(832001, 832041)),
    "fetch": tuple(range(842001, 842041)),
}

ACTION_MEMORY_TRAIN_LAYOUTS = {
    "doorkey": tuple(range(851001, 851097)),
    "fetch": tuple(range(861001, 861097)),
}
ACTION_MEMORY_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(852001, 852041)),
    "fetch": tuple(range(862001, 862041)),
}

INTERACTION_CONTEXT_TRAIN_LAYOUTS = {
    "doorkey": tuple(range(871001, 871097)),
    "fetch": tuple(range(881001, 881097)),
}
INTERACTION_CONTEXT_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(872001, 872041)),
    "fetch": tuple(range(882001, 882041)),
}

POST_OPEN_TRAIN_LAYOUTS = {
    "doorkey": tuple(range(891001, 891097)),
    "fetch": tuple(range(901001, 901097)),
}
POST_OPEN_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(892001, 892041)),
    "fetch": tuple(range(902001, 902041)),
}

PHASE_ROUTER_TRAIN_LAYOUTS = {
    "doorkey": tuple(range(911001, 911097)),
    "fetch": tuple(range(921001, 921097)),
}
PHASE_ROUTER_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(912001, 912041)),
    "fetch": tuple(range(922001, 922041)),
}

PUBLIC_EVENT_MEMORY_TRAIN_LAYOUTS = {
    "doorkey": tuple(range(931001, 931097)),
    "fetch": tuple(range(941001, 941097)),
}
PUBLIC_EVENT_MEMORY_MONITOR_LAYOUTS = {
    "doorkey": tuple(range(932001, 932041)),
    "fetch": tuple(range(942001, 942041)),
}


def validate_split_contract() -> None:
    """Detect accidental dataset leakage, including M20/M21 sealed finals."""
    all_layouts = [
        seed
        for split in (TRAIN_LAYOUTS, MONITOR_LAYOUTS, FINAL_LAYOUTS, BALANCE_MONITOR_LAYOUTS, MEMORY_MONITOR_LAYOUTS, EFFECT_MONITOR_LAYOUTS, PROBE_TRAIN_LAYOUTS, PROBE_MONITOR_LAYOUTS, RULE_SHIFT_MONITOR_LAYOUTS, ACTION_TRAIN_LAYOUTS, ACTION_MONITOR_LAYOUTS, SKILL_TRAIN_LAYOUTS, SKILL_MONITOR_LAYOUTS, BC_TRAIN_LAYOUTS, BC_MONITOR_LAYOUTS, VISION_TRAIN_LAYOUTS, VISION_MONITOR_LAYOUTS, CORRECT_TRAIN_LAYOUTS, CORRECT_ACQUIRE_LAYOUTS, CORRECT_MONITOR_LAYOUTS, INTERVENE_TRAIN_LAYOUTS, INTERVENE_ACQUIRE_LAYOUTS, INTERVENE_MONITOR_LAYOUTS, RECURRENT_TRAIN_LAYOUTS, RECURRENT_MONITOR_LAYOUTS, ACTION_MEMORY_TRAIN_LAYOUTS, ACTION_MEMORY_MONITOR_LAYOUTS, INTERACTION_CONTEXT_TRAIN_LAYOUTS, INTERACTION_CONTEXT_MONITOR_LAYOUTS, POST_OPEN_TRAIN_LAYOUTS, POST_OPEN_MONITOR_LAYOUTS, PHASE_ROUTER_TRAIN_LAYOUTS, PHASE_ROUTER_MONITOR_LAYOUTS, PUBLIC_EVENT_MEMORY_TRAIN_LAYOUTS, PUBLIC_EVENT_MEMORY_MONITOR_LAYOUTS)
        for layouts in split.values()
        for seed in layouts
    ]
    if len(all_layouts) != len(set(all_layouts)):
        raise ValueError("M23 train/monitor/final layouts are not disjoint")
    previous_final = set(range(313001, 313033)) | set(range(323001, 323033))
    previous_final |= set(range(333001, 333033)) | set(range(343001, 343033))
    if set(all_layouts) & previous_final:
        raise ValueError("M23 overlaps an earlier closed final layout")
    if len(MODEL_SEEDS) != len(set(MODEL_SEEDS)):
        raise ValueError("M23 model seeds must be distinct")


@dataclass(frozen=True)
class UnlockPickupWitness:
    """Evaluator-only verified events; not derived from reward or policy logits.

    Event indices refer to the *native executed-action* trace, not planned
    or teacher-labeled actions. The evaluator must establish the event flags
    using true native state changes, without exposing them to the learner.
    """

    initially_locked: bool
    box_initially_behind_locked_door: bool
    initial_goal_already_satisfied: bool
    actual_forward_movements: int
    key_pickup_action_index: int | None
    door_unlock_action_index: int | None
    key_drop_action_index: int | None
    correct_box_pickup_action_index: int | None
    reward_at_terminal: float

    def physical_success(self) -> bool:
        events = (
            self.key_pickup_action_index,
            self.door_unlock_action_index,
            self.key_drop_action_index,
            self.correct_box_pickup_action_index,
        )
        return (
            self.initially_locked
            and self.box_initially_behind_locked_door
            and not self.initial_goal_already_satisfied
            and self.actual_forward_movements > 0
            and all(index is not None and index >= 0 for index in events)
            and events[0] < events[1] < events[2] < events[3]
            and self.reward_at_terminal > 0
        )


def primary_zero_shot_gate(
    model_based: tuple[int, ...],
    source_bc: tuple[int, ...],
    replay_bc: tuple[int, ...],
    layouts_per_seed: int = 32,
) -> bool:
    """Preregistered gate; takes independent physical-success counts only.

    This is a necessary numerical condition, not a confidence interval.
    Statistical uncertainty is reported separately at the seed/layout level.
    """
    if any(len(x) != len(MODEL_SEEDS) for x in (model_based, source_bc, replay_bc)):
        raise ValueError("M23 primary gate requires all eight model seeds")
    if layouts_per_seed != len(FINAL_LAYOUTS["unlockpickup"]):
        raise ValueError("M23 final layout contract changed")
    if any(not 0 <= x <= layouts_per_seed for x in model_based + source_bc + replay_bc):
        raise ValueError("M23 success counts exceed evaluated native episodes")
    for control in (source_bc, replay_bc):
        advantages = [a - b for a, b in zip(model_based, control, strict=True)]
        if sum(advantages) < 16 or sum(x > 0 for x in advantages) < 6:
            return False
    return True

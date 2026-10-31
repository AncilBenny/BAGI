"""Synthetic contract-only validation, no XLand experiment implied."""
import json

import numpy as np
import pytest

from bagi.experiments.xland_gate_b_exact_collection import (
    BALANCED_LAYOUT_QUOTAS,
    FRESH_EVAL_LAYOUTS,
    QUERY_BUDGET,
    SOURCE_LAYOUTS,
)
from bagi.experiments.xland_gate_b_matched_learning import (
    BASE_ROWS,
    INPUT_WIDTH,
    diagnostic_tile_positions,
    train_matched_bc,
    validate_collection_header,
)


def metadata():
    return {"total_expert_queries": 582, "heldout_teacher_labels": 0,
            "model_results": [{"model_seed":17}, {"model_seed":29}]}


def summary():
    return {"source_run":39637067506,"source_artifact":12362840074,
            "model_seeds":[17,29],"source_layouts":list(SOURCE_LAYOUTS),
            "fixed_source_layout_quotas": {
                str(seed): count for seed, count in BALANCED_LAYOUT_QUOTAS
            },
            "future_evaluation_layouts_not_queried":list(FRESH_EVAL_LAYOUTS),
            "new_heldout_teacher_queries":0,
            "new_postintervention_policy_training_steps":0,
            "real_native_collection_steps":4*QUERY_BUDGET,
            "real_teacher_query_attempts":4*QUERY_BUDGET,
            "comparison_complete":True,
            "results":[{"model_seed":17,"matched":True},
                       {"model_seed":29,"matched":True}]}


def test_valid_header_requires_both_policy_seeds(tmp_path):
    (tmp_path/'summary.json').write_text(json.dumps(summary()))
    report, records=validate_collection_header(tmp_path, metadata())
    assert len(records)==2 and records[0]['model_seed']==17
    assert report['comparison_complete']


@pytest.mark.parametrize('key,bad',[
    ('comparison_complete',False),('new_heldout_teacher_queries',1),
    ('real_teacher_query_attempts',79),('real_native_collection_steps',79),
    ('model_seeds',[29,17]),('source_artifact',0),
    ('future_evaluation_layouts_not_queried',[9810]),
    ('fixed_source_layout_quotas',{'9701': 11, '9702': 9})])
def test_bad_summary_rejected_before_model_training(tmp_path,key,bad):
    d=summary();d[key]=bad
    (tmp_path/'summary.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='gate not passed'):
        validate_collection_header(tmp_path, metadata())


def test_missing_or_incomplete_model_seed_rejected(tmp_path):
    d=summary();d['results']=[d['results'][0]]
    (tmp_path/'summary.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Missing or reordered'):
        validate_collection_header(tmp_path, metadata())
    d=summary();d['results'][1]['matched']=False
    (tmp_path/'summary.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Incomplete model'):
        validate_collection_header(tmp_path, metadata())


def test_bc_rejects_missing_labels_without_native_libraries():
    x=np.zeros((BASE_ROWS,INPUT_WIDTH),dtype=np.float32)
    y=np.zeros(BASE_ROWS,dtype=np.int64)
    done=np.zeros(BASE_ROWS,dtype=bool)
    new={'obs':np.zeros((19,INPUT_WIDTH),dtype=np.float32),
         'expert_label':np.zeros(19,dtype=np.int64),'done':np.zeros(19,dtype=bool)}
    with pytest.raises(ValueError,match='feature contract'):
        train_matched_bc(x,y,done,new,17)


def test_eval_tile_position_diagnostic_tracks_grid_without_privileged_policy_input():
    grid = np.zeros((5, 5, 2), dtype=np.int32)
    blue = np.array([5, 2], dtype=np.int32)
    red = np.array([4, 0], dtype=np.int32)
    grid[1, 1] = blue
    grid[3, 3] = red
    assert diagnostic_tile_positions(grid, blue) == [[1, 1]]
    assert diagnostic_tile_positions(grid, red) == [[3, 3]]
    grid[1, 1] = [0, 0]
    grid[3, 2] = blue
    assert diagnostic_tile_positions(grid, blue) == [[3, 2]]
    assert diagnostic_tile_positions(grid, red) == [[3, 3]]
    assert diagnostic_tile_positions(grid, [8, 8]) == []

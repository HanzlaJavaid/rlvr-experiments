"""Independent reward normalization with whole-native-trace process credit."""
import pytest
from science_gspo_g.advantages import build_advantages


def meta(n=4, length=8, span=(1, 5)):
    # token 0 is an optional native opening; token 5 closes thinking;
    # token 6 is the final answer and token 7 is EOS.
    return [dict(example_id='q', length=length, status='complete',
                 reasoning_token_span=list(span), visible_start_token=span[1]+1,
                 process_token_spans=[[list(span)] for _ in range(3)]) for _ in range(n)]


def test_three_independent_scores_credit_same_trace_not_answer_or_delimiters():
    scores=[[.5,.9,.3,.4],[.5,.1,.7,.6]]
    out=build_advantages(scores,meta(2),group_size=2)
    assert out['process_z'][0] == pytest.approx([1,-1,-1])
    assert out['advantages'][0] == pytest.approx([0]+[-1/12]*4+[0]*3)
    assert out['advantages'][1] == pytest.approx([0]+[1/12]*4+[0]*3)


def test_accuracy_is_the_only_outcome_and_population_normalized():
    out=build_advantages([[1,.5,.5,.5],[0,.5,.5,.5]],meta(2),group_size=2)
    assert out['outcome_rewards']==[1,0]
    assert out['outcome_advantages']==[1,-1]
    assert out['advantages']==[[1]*8,[-1]*8]


def test_noise_floor_and_ties():
    out=build_advantages([[.5]*4,[.5,.501,.5,.5]],meta(2),group_size=2)
    assert abs(out['process_z'][0][0]) < .006
    assert out['process_z'][0][1:]==[0,0]
    assert build_advantages([[.5]*4]*4,meta())['advantages']==[[0]*8]*4


def test_censoring_and_single_valid_group_masked():
    out=build_advantages([[.5]*4,[None]*4,[float('nan')]*4,[None]*4],meta())
    assert out['skipped_groups']==1
    assert out['active_rows']==[False]*4
    assert out['advantages']==[[0]*8]*4


@pytest.mark.parametrize('bad_value',[float('inf'),True,-1,1.1])
def test_invalid_rewards_rejected(bad_value):
    with pytest.raises(ValueError):
        build_advantages([[.5]*3+[bad_value],[.5]*4],meta(2),group_size=2)


def test_legacy_six_columns_rejected():
    with pytest.raises(ValueError,match='exactly four'):
        build_advantages([[.5]*6]*2,meta(2),group_size=2)


def test_distributed_order_group_validation():
    m=meta(4);m[3]['example_id']='other'
    with pytest.raises(ValueError,match='crosses'):
        build_advantages([[.5]*4]*4,m)
    m=meta(8)
    for row in m[4:]:row['example_id']='q2'
    out=build_advantages([[.2]*4]*4+[[.9]*4]*4,m)
    assert out['advantages']==[[0]*8]*8


def test_missing_process_component_preserves_accuracy_and_other_stages():
    out=build_advantages([[1,1,None,1],[0,0,0,0]],meta(2),group_size=2)
    assert out['active_rows']==[True,True]
    assert out['process_z'][0]==[1,0,1]
    assert out['advantages'][0][1:5]==pytest.approx([1+1/6]*4)
    assert out['advantages'][0][-1]==1
    assert out['advantages'][1][-1]==-1


def test_accuracy_cohort_uses_only_available_accuracy_without_fabricated_zero():
    rows=[[1,None,None,None],[0,None,None,None],
          [.5,None,None,None],[None,None,None,None]]
    out=build_advantages(rows,meta())
    assert out['outcome_cohorts'][0]['columns']==[0]
    assert out['outcome_cohorts'][0]['members']==[0,1,2]
    assert out['outcome_rewards']==[1,0,.5,None]
    assert out['active_rows']==[True,True,True,False]


def test_each_process_dimension_uses_its_own_available_cohort():
    rows=[[None,1,None,1],[None,0,1,None],[None,None,0,0],[None]*4]
    out=build_advantages(rows,meta())
    assert out['process_z']==[[1,0,1],[-1,1,0],[0,-1,-1],[0,0,0]]
    assert out['advantages'][0][1:5]==pytest.approx([1/6]*4)
    assert out['advantages'][1]==[0]*8
    assert out['advantages'][2][1:5]==pytest.approx([-1/6]*4)
    assert out['active_rows']==[True,True,True,False]


@pytest.mark.parametrize('failed_column',range(4))
def test_each_failed_reward_keeps_every_other_dimension_usable(failed_column):
    rows=[[1]*4,[0]*4]
    rows[0][failed_column]=None
    out=build_advantages(rows,meta(2),group_size=2)
    assert out['active_rows']==[True,True]
    assert any(x != 0 for x in out['advantages'][0])
    if failed_column==0:
        assert out['outcome_rewards']==[None,None]
        assert out['process_z'][0]==[1,1,1]
        assert out['advantages'][0][1:5]==pytest.approx([.25]*4)
        assert out['advantages'][0][-1]==0
    else:
        assert out['outcome_rewards']==[1,0]
        assert out['outcome_advantages']==[1,-1]
        assert out['process_z'][0][failed_column-1]==0


def test_padding_is_zero_and_process_bound():
    out=build_advantages([[1]*4,[0]*4],meta(2),group_size=2,width=11)
    assert out['advantages'][0][-3:]==[0,0,0]
    for a,o in zip(out['advantages'],out['outcome_advantages']):
        assert all(abs(v-o)<=.5 for v in a[:8])


def test_process_clip_limits_large_z_score_before_combining():
    rows=[[.5,1,1,1]]+[[.5,0,0,0] for _ in range(15)]
    out=build_advantages(rows,meta(16),group_size=16)
    assert out['process_z'][0]==[2,2,2]
    assert out['advantages'][0][1:5]==pytest.approx([.5]*4)


def test_accuracy_only_weight_zero_removes_all_process_bonus():
    out=build_advantages([[1]*4,[0]*4],meta(2),group_size=2,process_weight=0)
    assert out['advantages']==[[1]*8,[-1]*8]


def test_partial_closed_thinking_retains_feedback_without_final_answer():
    m=meta(2,length=6)
    for row in m:row.update(status='truncated')
    out=build_advantages([[None,1,None,1],[None,0,None,0]],m,group_size=2)
    assert out['advantages'][0]==pytest.approx([0]+[1/6]*4+[0])
    assert out['advantages'][1]==pytest.approx([0]+[-1/6]*4+[0])
    assert out['outcome_rewards']==[None,None]


def test_outcome_only_rows_need_no_thinking_span():
    m=meta(2)
    for row in m:
        row.update(reasoning_token_span=None,process_token_spans=[[],[],[]],status='empty_reasoning')
    out=build_advantages([[1,None,None,None],[0,None,None,None]],m,group_size=2)
    assert out['advantages']==[[1]*8,[-1]*8]


def test_unavailable_output_has_no_fabricated_negative_scores():
    m=meta(2);m[1].update(status='malformed',reasoning_token_span=None,visible_start_token=None)
    out=build_advantages([[1]*4,[None]*4],m,group_size=2)
    assert out['advantages'][1] == [0]*8
    assert out['active_rows']==[False,False]


@pytest.mark.parametrize('span',[None,[],[1],[1,1],[-1,3],[False,3],[1,8],[1,9]])
@pytest.mark.parametrize('scores',[[[1]*4,[0]*4],[[.5]*4,[.5]*4],[[1]*4,[None]*4]])
def test_invalid_trace_span_rejected_even_when_tied_or_singleton(span,scores):
    m=meta(2);m[0]['reasoning_token_span']=span
    with pytest.raises(ValueError,match='reasoning token span'):
        build_advantages(scores,m,group_size=2)


def test_rejects_old_prefix_metadata_without_explicit_trace_span():
    m=meta(2)
    for row in m:
        del row['reasoning_token_span']
        row.update(process_start_token=1,section_end_tokens=[2,3,4])
    with pytest.raises(ValueError,match='reasoning token span'):
        build_advantages([[1]*4,[0]*4],m,group_size=2)


def test_rejects_span_that_would_cross_native_closing_boundary():
    m=meta(2);m[0]['visible_start_token']=5
    with pytest.raises(ValueError,match='native closing boundary'):
        build_advantages([[1]*4,[0]*4],m,group_size=2)


def test_rejects_different_prefixes_for_available_process_scores():
    m=meta(2);m[0]['process_token_spans']=[[[1,3]],[[1,4]],[[1,5]]]
    with pytest.raises(ValueError,match='same whole reasoning span'):
        build_advantages([[1]*4,[0]*4],m,group_size=2)

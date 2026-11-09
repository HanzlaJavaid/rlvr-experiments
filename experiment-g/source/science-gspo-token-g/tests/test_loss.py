import pytest
import torch
from science_gspo_g.loss import gspo_token_loss


def tensors(adv=(1,-1), current=(-1.,-1.), old=None):
    p=torch.tensor([current],requires_grad=True)
    return p, torch.tensor([old or current]),torch.tensor([adv],dtype=torch.float32),torch.ones_like(p)


def test_opposite_token_advantages_do_not_cancel():
    p,old,a,m=tensors()
    loss,_=gspo_token_loss(p,old,a,m)
    loss.backward()
    assert p.grad.tolist()[0]==pytest.approx([-.5,.5])
    q=p.detach().clone().requires_grad_()
    naive=-(torch.exp((q-old).mean(-1,keepdim=True))*a).mean()
    naive.backward()
    assert q.grad.tolist()[0]==pytest.approx([0,0])


def test_constant_advantages_match_original_gspo_value_and_gradient():
    p,old,a,m=tensors(adv=(.7,.7),current=(-.9998,-1),old=(-1,-1))
    loss,_=gspo_token_loss(p,old,a,m)
    loss.backward();gradient=p.grad.clone()
    q=p.detach().clone().requires_grad_()
    s=torch.exp((q-old).mean(-1))
    baseline=-torch.minimum(s*.7,s.clamp(.9997,1.0004)*.7).mean()
    baseline.backward()
    assert loss.item()==pytest.approx(baseline.item())
    assert torch.allclose(gradient,q.grad,atol=1e-7)


def test_shared_ratio_clipping_respects_advantage_sign():
    p,old,a,m=tensors(current=(-.9,-.9),old=(-1,-1))
    loss,d=gspo_token_loss(p,old,a,m,epsilon_high=.01)
    loss.backward()
    assert p.grad[0,0].item()==0
    assert p.grad[0,1].item()>0
    assert d['high_clipped'].tolist()==[[True,False]]


def test_padding_nan_is_ignored_and_accumulation_correct():
    p,old,a,m=tensors(current=(-1,float('nan')),old=(-1,float('nan')))
    a[0,1]=float('nan');m[0,1]=0
    loss,_=gspo_token_loss(p,old,a,m,accumulation_steps=4)
    loss.backward()
    assert loss.item()==pytest.approx(-.25)
    assert p.grad[0,0].item()==pytest.approx(-.25)
    assert p.grad[0,1].item()==0


def test_reference_kl_and_detached_inference_correction():
    p,old,a,m=tensors(adv=(0,0),current=(-.8,-.8),old=(-1,-1))
    ref=torch.full_like(p,-1)
    loss,d=gspo_token_loss(p,old,a,m,ref_logps=ref,beta=.01)
    loss.backward()
    assert loss.item()>0 and (p.grad>0).all()
    p,old,a,m=tensors(adv=(1,1))
    correction=torch.full_like(p,2,requires_grad=True)
    loss,_=gspo_token_loss(p,old,a,m,inference_correction=correction)
    loss.backward()
    assert correction.grad is None
    assert p.grad.tolist()[0]==pytest.approx([-1,-1])


def test_nonfinite_active_and_missing_reference_fail():
    p,old,a,m=tensors(current=(float('nan'),-1))
    with pytest.raises(ValueError,match='Nonfinite'):gspo_token_loss(p,old,a,m)
    p,old,a,m=tensors()
    with pytest.raises(ValueError,match='reference'):gspo_token_loss(p,old,a,m,beta=.01)


def test_process_only_rewards_update_thinking_without_final_answer_policy_gradient():
    from test_parsing import ByteTokenizer,TEXT
    from science_gspo_g.parsing import parse_completion
    from science_gspo_g.advantages import build_advantages
    tok=ByteTokenizer();ids=tok.encode(TEXT)+[tok.eos_token_id]
    parsed=parse_completion(ids,tok,len(ids))
    meta=[dict(example_id='q',length=len(ids),**{k:parsed[k] for k in
          ('status','reasoning_token_span','process_token_spans','visible_start_token')}) for _ in range(2)]
    result=build_advantages([[.5,.9,.9,.9],[.5,.1,.1,.1]],meta,group_size=2)
    logps=torch.full((2,len(ids)),-1.,requires_grad=True)
    advantages=torch.tensor(result['advantages'])
    loss,_=gspo_token_loss(logps,logps.detach(),advantages,torch.ones_like(logps))
    loss.backward()
    start,stop=parsed['reasoning_token_span']
    assert torch.count_nonzero(logps.grad[:,:start]) == 0
    assert (logps.grad[0,start:stop] < 0).all()
    assert (logps.grad[1,start:stop] > 0).all()
    assert torch.count_nonzero(logps.grad[:,stop:]) == 0

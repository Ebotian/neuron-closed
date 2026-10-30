import sys, json, math, torch
sys.path.insert(0,'/root/autodl-tmp/scripts/j2space')
import engine as E
dev='cuda:0'
tok, model, lm = E.load_model('/root/autodl-tmp/Qwen3.5-4B', dev)
p50 = E.estimate_p50(tok, model, lm, list(range(32)), dev)
norms = E.column_norms(lm, list(range(32)))
blob = torch.load('/root/autodl-tmp/runs/p_a3.pt')
g = blob['g_teach']
res = {}
for l in (0, 13, 31):
    W = lm.layers[l].mlp.down_proj.weight.detach().float().to(dev)      # [2560, 9216]
    coln = W.norm(dim=0, keepdim=True).clamp(min=1e-12)
    Wn = W / coln
    y = g[l].to(dev).float() * coln.squeeze(0) / p50[l]                  # = W^T gradPhi  (9216)
    A = W @ W.T                                                          # 2560x2560
    gphi = torch.linalg.solve(A.double(), (W.double() @ y.double()).unsqueeze(1)).squeeze(1).float()  # gradPhi
    # min-norm lift of the dose gradient:  u = M^+ g  via thin SVD of Wn
    U_, S_, Vh = torch.linalg.svd(Wn, full_matrices=False)
    proj = Vh @ g[l].to(dev).float()
    uM = Vh.T @ torch.where(S_ > 1e-6*S_.max(), proj/S_.pow(2), torch.zeros_like(proj))
    # naive lift: u = g / ||g||_M
    gl = g[l].to(dev).float()
    uG = gl / (Wn @ gl).norm()
    c = lambda a,b: float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
    res[l] = {
      'cos(Wn u_M , gradPhi)': c(Wn @ uM, gphi),
      'cos(Wn u_g , gradPhi)': c(Wn @ uG, gphi),
      'cos(Wn u_M , Wn u_g)': c(Wn @ uM, Wn @ uG),
      'norm_ratio |Wn u_M|/|Wn u_g|': float((Wn@uM).norm()/(Wn@uG).norm()),
    }
    print(l, {k: round(v,4) if isinstance(v,float) else v for k,v in res[l].items()}, flush=True)
json.dump(res, open('/root/autodl-tmp/runs/gauge.json','w'), indent=1)

import torch, matplotlib.pyplot as plt
from cnn import BigCNN
from tv_split import train_raw, val_raw, train_deep, val_deep, device

def run_training(train_set, val_set, label):
    loader_tr = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    loader_va = torch.utils.data.DataLoader(val_set,   batch_size=64)
    model = BigCNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = torch.nn.MSELoss()
    rmse_hist = []
    for epoch in range(30):
        # ---- train ----
        model.train(); tr_loss, n=0,0
        for batch in loader_tr:
            x = batch['features'].to(device); y = batch['labels'].to(device).squeeze(-1)
            optim.zero_grad()
            loss = crit(model(x).squeeze(-1), y); loss.backward(); optim.step()
        # ---- val ----
        model.eval(); se, n = 0.0, 0
        with torch.no_grad():
            for batch in loader_va:
                x = batch['features'].to(device); y = batch['labels'].to(device).squeeze(-1)
                p = model(x).squeeze(-1)
                se += torch.sum((p - y)**2).item(); n += y.numel()
        rmse = (se/n)**0.5; rmse_hist.append(rmse)
        print(f"{label} | epoch {epoch:02d}  val RMSE {rmse:6.3f}")
    return rmse_hist

rmse_raw  = run_training(train_raw,  val_raw,  "RAW ")
rmse_deep = run_training(train_deep, val_deep, "DEEP")

# ------------ 그래프 비교 -------------
plt.plot(rmse_raw,  label='Raw input')
plt.plot(rmse_deep, label='Deep-tone BP input')
plt.xlabel('Epoch'); plt.ylabel('Val RMSE [km]')
plt.legend(); plt.tight_layout()
plt.savefig("rmse_compare.png", dpi=300)
print("Saved → rmse_compare.png")

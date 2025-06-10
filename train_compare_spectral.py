import torch, matplotlib.pyplot as plt
from cnn import *
from tv_split import train_raw, val_raw, train_deep, val_deep, train_fft, val_fft, device
import importlib

#%
def run_training(train_set, val_set, model, label):
    loader_tr = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    loader_va = torch.utils.data.DataLoader(val_set,   batch_size=64)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=8e-4)
    crit  = torch.nn.MSELoss()
    rmse_hist = []
    for epoch in range(500):
        # ---- train ----
        model.train(); tr_loss, n=0,0
        for batch in loader_tr:
            x = batch['features'].to(device); y = batch['labels'].to(device).squeeze(-1)
            optim.zero_grad()
            out = model(x).squeeze(-1)
            assert out.shape == y.shape, f"{out.shape} != {y.shape}"
            loss = crit(out, y)
            loss.backward()
            optim.step()
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

#%%
rmse_fft_mlp = run_training(train_fft, val_fft, SpectralMLP(), "FFT")
rmse_fft_mlp_sig = run_training(train_fft, val_fft, SpectralMLP(activation=nn.Sigmoid()), "FFT Sigmoid")
rmse_raw  = run_training(train_raw,  val_raw, DeepCNN(), "RAW")
rmse_deep  = run_training(train_deep,  val_deep, DeepCNN(), "DEEP")
rmse_deep_sig  = run_training(train_deep,  val_deep, DeepCNN(activation=nn.Sigmoid()), "DEEP Sigmoid")
rmse_fft = run_training(train_fft, val_fft, SpectralCNN(), "FFT")
rmse_fft_sig = run_training(train_fft, val_fft, SpectralCNN(activation=nn.Sigmoid()), "FFT Sigmoid")


# rmse_fft_sig = run_training(train_fft, val_fft, SpectralMLP(activation=nn.LeakyReLU()), "FFT MLP")
rmse_fourier = run_training(train_deep, val_deep, FourierNet(activation_func=nn.LeakyReLU()), "Fourier")


rmse_beamcnn = run_training(train_fft, val_fft, BeamCNN(activation=nn.ReLU()), "BeamCNN")



#%%
# ------------ 그래프 비교 Graph Comparison -------------
plt.plot(rmse_raw,  label='Raw input')
plt.plot(rmse_deep, label='Deep-tone BP input')
plt.plot(rmse_fft, label="FFT Selected Coefficients")
plt.plot(rmse_deep_sig, label='Deep-tone Sigmoid')
plt.plot(rmse_fft_sig, label="FFT Sigmoid")
plt.plot(rmse_fourier, label='FNO-Like')
plt.plot(rmse_beamcnn, label='BeamCNN')
plt.xlabel('Epoch'); plt.ylabel('Val RMSE [km]')

plt.title("Comparison of Methods")
plt.legend(); plt.tight_layout()
plt.savefig("rmse_all_fm_compare_fft_rerun.png", dpi=300)
plt.show()
# print("Saved → rmse_compare.png")

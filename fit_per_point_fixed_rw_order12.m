% fit_roi_fixed_rw_order12.m
% 固定 r,w；只擬合以中心(541,541)為中心的 713x713 ROI，用 (Order1 + Order2) 聯合曲線。
clc; clear; close all;
tic;

% ====== 可調參數 ======
USE_PARFOR = true;            % parfor 在 (a2pi,alpha) 維度平行；若不穩先設 false
outdir = 'fitting_roi_outputs';
if ~exist(outdir,'dir'), mkdir(outdir); end

% 固定 (r, w)
r_fixed     = 0.0675;
w_fixed     = 0.75 * 8e-3;    % 與 dx=8e-3 一致

% 搜索範圍（總 K = numel(a2pi_range)*numel(alpha_range)）
a2pi_range  = 232:2:242;                 % 6
alpha_range = (-0.5:0.1:0.4) * pi;       % 10
% → K = 60

% ====== 讀資料 ======
L = load('exp_order12_aligned.mat', 'exp1','exp2','a_values','Na','center_rc','n1','n2');
exp1      = L.exp1;        % [1080 x 1080 x Na] (single)
exp2      = L.exp2;
a_values  = L.a_values(:);
Na        = L.Na;
center_rc = L.center_rc;   % 一般是 [541 541]
n1        = L.n1;          % 實驗端中心正規化常數（order1）
n2        = L.n2;          % 實驗端中心正規化常數（order2）
[Ny, Nx, Na_chk] = size(exp1);
assert(Ny==1080 && Nx==1080 && Na_chk==Na, '資料尺寸不是 1080x1080xNa。');

% ====== 定義 713x713 ROI ======
roi_half = (713-1)/2;         % = 356
cy = 541; cx = 541;
r1 = cy - roi_half; r2 = cy + roi_half;   % 185:897
c1 = cx - roi_half; c2 = cx + roi_half;   % 185:897
assert(r1>=1 && c1>=1 && r2<=Ny && c2<=Nx, 'ROI 超出邊界。');

% 先把實驗堆疊裁成 ROI，省記憶體與傳輸
exp1R = exp1(r1:r2, c1:c2, :);   % [713 x 713 x Na]
exp2R = exp2(r1:r2, c1:c2, :);
[NyR, NxR, ~] = size(exp1R);     % 713, 713

% ====== 模擬常數 / 場域準備（仍在 1080 網格做模擬，再取 ROI） ======
lambda = 447e-6; f = 300; dx = 8e-3;
Beam_size = 2.634705;
l = 600; z = 600;
pixel_grating = 1080;        % Nx=Ny=1080

% 實長度座標（供 PSF）
x = -Nx/2*dx : dx : (Nx/2-1)*dx;
y = -Ny/2*dx : dx : (Ny/2-1)*dx;
[X, Y] = meshgrid(x, -y);

theta_deg = -90;
theta_blazed = deg2rad(theta_deg);
level = 12; repeat = 1; min_phase = 0;

% 入射光場
U0 = call_Gaussian_beam(Beam_size, pixel_grating, dx);

% 頻域遮罩（Order1 / Order2）— 仍然在 1080 網格建立
h = 541;
v1 = 631; shift_y1 = 427;   % order1
v2 = 721; shift_y2 = 857;   % order2
r_filter = 50;
[xi, yi] = meshgrid(1:Nx, 1:Ny);
mask1 = (xi - h).^2 + (yi - v1).^2 <= r_filter^2;
mask2 = (xi - h).^2 + (yi - v2).^2 <= r_filter^2;

% 串擾 PSF（固定 w）
Gaussian_PSF = exp(-(X.^2 + Y.^2)/(2*w_fixed^2));
Gaussian_PSF = Gaussian_PSF / sum(Gaussian_PSF,'all');

Lbox = Nx*dx;

% ====== (a2pi, alpha) 列表 ======
[A2, AL] = ndgrid(a2pi_range, alpha_range);
combos = [A2(:), AL(:)];
K = size(combos,1);

% ====== ROI 結果圖初始化 ======
best_chi2_roi  = inf(NyR, NxR, 'single');
best_a2pi_roi  = zeros(NyR, NxR, 'single');
best_alpha_roi = zeros(NyR, NxR, 'single');

% ====== 平行化設定 ======
if USE_PARFOR
    p = gcp('nocreate');
    if isempty(p)
        try, parpool('Processes'); catch, parpool('local'); end
    end
end

% ====== 主迴圈（在 (a2pi,alpha) 維度平行） ======
chi2_cells  = cell(K,1);
a2pi_cells  = cell(K,1);
alpha_cells = cell(K,1);

if USE_PARFOR
    parfor k = 1:K
        a2pi  = combos(k,1);
        alpha = combos(k,2);
        [chi2_cells{k}, a2pi_cells{k}, alpha_cells{k}] = compute_chi2_roi( ...
            a2pi, alpha, a_values, Na, ...
            U0, Gaussian_PSF, mask1, mask2, ...
            r_fixed, lambda, Lbox, l, z, f, ...
            level, repeat, min_phase, theta_blazed, ...
            n1, n2, exp1R, exp2R, r1, r2, c1, c2, shift_y1, shift_y2);
    end
else
    for k = 1:K
        a2pi  = combos(k,1);
        alpha = combos(k,2);
        [chi2_cells{k}, a2pi_cells{k}, alpha_cells{k}] = compute_chi2_roi( ...
            a2pi, alpha, a_values, Na, ...
            U0, Gaussian_PSF, mask1, mask2, ...
            r_fixed, lambda, Lbox, l, z, f, ...
            level, repeat, min_phase, theta_blazed, ...
            n1, n2, exp1R, exp2R, r1, r2, c1, c2, shift_y1, shift_y2);
    end
end

% 合併取得最小 chi^2 與參數
for k = 1:K
    chi2_k = chi2_cells{k};
    a2_k   = a2pi_cells{k};
    al_k   = alpha_cells{k};

    better = chi2_k < best_chi2_roi;
    best_chi2_roi(better)  = chi2_k(better);
    best_a2pi_roi(better)  = a2_k;
    best_alpha_roi(better) = al_k;
end

% ====== 存檔與 quicklook ======
save(fullfile(outdir,'fit_roi_713x713_fixed_rw.mat'), ...
    'best_a2pi_roi','best_alpha_roi','best_chi2_roi', ...
    'r_fixed','w_fixed','a2pi_range','alpha_range', ...
    'r1','r2','c1','c2','-v7.3');

figure; imagesc(best_a2pi_roi);  axis image ; colorbar; title('ROI best a2pi');
figure; imagesc(best_alpha_roi); axis image ; colorbar; title('ROI best \alpha');
q99 = quantile(double(best_chi2_roi(:)), 0.99);
if isempty(q99) || ~isfinite(q99), q99 = max(best_chi2_roi(:)); end
figure; imagesc(best_chi2_roi, [0 q99]); axis image off; colorbar; title('ROI min \chi^2 (99% clip)');

fprintf('Done. ROI mean(min chi2) = %.6f\n', mean(best_chi2_roi(:),'omitnan'));
toc;

%% ===================== 內部函式 =====================

function [chi2_roi, a2pi_s, alpha_s] = compute_chi2_roi( ...
    a2pi, alpha, a_values, Na, ...
    U0, Gaussian_PSF, mask1, mask2, ...
    r_fixed, lambda, Lbox, l, z, f, ...
    level, repeat, min_phase, theta_blazed, ...
    n1, n2, exp1R, exp2R, r1, r2, c1, c2, shift_y1, shift_y2)

    % 在整張 1080×1080 上做模擬，再切 ROI（避免邊界效應）
    NyR = size(exp1R,1);  NxR = size(exp1R,2);
    chi2_roi = zeros(NyR, NxR, 'single');

    Nx = 1080; % 模擬網格大小（與資料一致）

    for kk = 1:Na
        a = a_values(kk);

        % 相位 + 串擾
        Blazed_theta = call_Grating_phase(Nx, a, min_phase, level, repeat, theta_blazed, a2pi);
        phi = conv2(Blazed_theta, Gaussian_PSF, 'same') + alpha;

        den = 1 + r_fixed * exp(1j*phi);
        den(abs(den) < 1e-12) = 1e-12;
        E = -(r_fixed + exp(1j*phi)) ./ den;
        U_slm = U0 .* E;

        % 傳播到 Fourier 面
        U_len = call_propTF(U_slm, Lbox, lambda, l - f);
        U_fourier = call_DFT(U_len);

        % Order1
        U_f1 = zeros(size(U_fourier)); U_f1(mask1) = U_fourier(mask1);
        U_img1 = call_propTF(U_f1, Lbox, lambda, -(z - f));
        U_img1 = circshift(U_img1, [-shift_y1, 1]);
        im1 = abs(U_img1).^2 / max(n1,1);

        % Order2
        U_f2 = zeros(size(U_fourier)); U_f2(mask2) = U_fourier(mask2);
        U_img2 = call_propTF(U_f2, Lbox, lambda, -(z - f));
        U_img2 = circshift(U_img2, [-shift_y2, 1]);
        im2 = abs(U_img2).^2 / max(n2,1);

        % 只取 ROI 比對
        im1R = im1(r1:r2, c1:c2);
        im2R = im2(r1:r2, c1:c2);

        exp1_k = exp1R(:,:,kk);
        exp2_k = exp2R(:,:,kk);

        chi2_roi = chi2_roi + ((exp1_k - im1R).^2) ./ (im1R + 1e-6) ...
                            + ((exp2_k - im2R).^2) ./ (im2R + 1e-6);
    end

    a2pi_s  = single(a2pi);
    alpha_s = single(alpha);
end

% === 包裝：相容不同簽名的自訂函式 ===
function U0 = call_Gaussian_beam(Beam_size, N, dx)
    nin = nargin('Gaussian_beam');
    if nin < 0, error('找不到 Gaussian_beam()'); end
    if nin == 3
        U0 = Gaussian_beam(Beam_size, N, dx);
    elseif nin == 2
        U0 = Gaussian_beam(Beam_size, N);
    else
        error('Gaussian_beam() 參數數量不支援：%d', nin);
    end
end

function Blazed_theta = call_Grating_phase(N, a, min_phase, level, repeat, theta_blazed, a2pi)
    nin = nargin('Grating_phase');
    if nin < 0, error('找不到 Grating_phase()'); end
    if nin == 7
        Blazed_theta = Grating_phase(N, a, min_phase, level, repeat, theta_blazed, a2pi);
    elseif nin == 4
        Blazed_theta = Grating_phase(N, a, theta_blazed, a2pi);
    else
        error('Grating_phase() 參數數量不支援：%d', nin);
    end
end

function Uprop = call_propTF(U, L, lambda, dz)
    nin = nargin('propTF');
    if nin < 0, error('找不到 propTF()'); end
    if nin == 4
        Uprop = propTF(U, L, lambda, dz);
    elseif nin == 3
        Uprop = propTF(U, dz, lambda);  % 若你的 3 參數版本不同，改這裡
    else
        error('propTF() 參數數量不支援：%d', nin);
    end
end

function U_fourier = call_DFT(U)
    nin = nargin('DFT');
    if nin < 0
        U_fourier = fftshift(fft2(U));
    elseif nin == 1
        U_fourier = DFT(U);
    else
        error('DFT() 參數數量不支援：%d', nin);
    end
end

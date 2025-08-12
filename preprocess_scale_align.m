% preprocess_scale_align.m  (v2 one-pass affine)
% 讀 ccd_stack_order1_3.mat / ccd_stack_order2_3.mat
% 將 CCD 2.2um 像素重取樣到理論 8um 網格：縮放 s=2.2/8；把 CCD 中心(1095,1415)對到(541,541)；
% 裁成 1080×1080；各自以中心像素曲線最大值做正規化；存成 exp_order12_aligned.mat

clc; clear; close all;

in1 = 'ccd_stack_order1_3.mat';
in2 = 'ccd_stack_order1_7.mat';   % ← 你原本這裡寫成 order1_7，看起來是手誤

% 讀堆疊（自動抓最大3D陣列）
read_stack = @(fname) local_read_stack(fname);
stk1 = read_stack(in1);
stk2 = read_stack(in2);
[H1,W1,N1] = size(stk1);
[H2,W2,N2] = size(stk2);
if N1~=N2, error('order1 與 order2 幅數不一致：%d vs %d', N1, N2); end
Na = N1;

% 取得 a_values（若沒有就用預設）
a_values = local_get_a_values(in1, Na);

% 幾何參數：縮放 + 置中
s = 2.2/8;                 % CCD→理論 像素比例（縮小）
targetN = 1080;            % 理論平面大小
tCenter = [541,541];       % 理論影像中心 (row,col)
ccdCenter = [1095,1415];   % CCD 中心 (row,col)
ctr_scaled = ccdCenter * s;                         % 縮放後的中心位置
rowShift   = tCenter(1) - ctr_scaled(1);
colShift   = tCenter(2) - ctr_scaled(2);

% 合併為單一仿射（一次插值）：先縮放，再平移
T = [ s  0  0;
      0  s  0;
      0  0  1 ];
Tx = [ 1  0  0;
       0  1  0;
       colShift rowShift 1 ];
tform = affine2d(T * Tx);   % 注意座標順序：imwarp 用 [x y 1] 行向量右乘

% 目標座標系：1080×1080，以 (541,541) 為中心
RA = imref2d([targetN targetN]);  % 預設 world limits 與像素對齊即可

% 一次仿射變換 + 直接得到 1080×1080（不足自動補零）
exp1 = zeros(targetN, targetN, Na, 'single');
exp2 = zeros(targetN, targetN, Na, 'single');

fprintf('One-pass affine warp: scale s=%.6f, shift (row=%.3f, col=%.3f)\n', s, rowShift, colShift);
for k = 1:Na
    exp1(:,:,k) = imwarp(stk1(:,:,k), tform, 'OutputView', RA, ...
                         'InterpolationMethod','linear', 'FillValues', 0);
    exp2(:,:,k) = imwarp(stk2(:,:,k), tform, 'OutputView', RA, ...
                         'InterpolationMethod','linear', 'FillValues', 0);
end

clear stk1 stk2

% 正規化（各 order 用中心像素曲線最大值）
center_rc = [541,541];
center_curve1 = squeeze(exp1(center_rc(1), center_rc(2), :));
center_curve2 = squeeze(exp2(center_rc(1), center_rc(2), :));
n1 = max(center_curve1); if n1<=0, n1 = 1; end
n2 = max(center_curve2); if n2<=0, n2 = 1; end
exp1 = exp1 / n1;
exp2 = exp2 / n2;

% 基本檢查
assert(isequal(size(exp1), [1080 1080 Na]), 'exp1 尺寸不是 1080×1080×Na');
assert(isequal(size(exp2), [1080 1080 Na]), 'exp2 尺寸不是 1080×1080×Na');
% 存 v7.3（大陣列）
save('exp_order12_aligned.mat', 'exp1','exp2','a_values','Na','center_rc','n1','n2','-v7.3');

% 簡易檢視
figure; 
subplot(1,3,1); imagesc(mean(exp1,3)); axis image ; title('Order1 mean'); colormap turbo; 
subplot(1,3,2); imagesc(mean(exp2,3)); axis image ; title('Order2 mean'); colormap turbo; 
subplot(1,3,3); plot(a_values, center_curve1/n1, 'rx'); hold on; plot(a_values, center_curve2/n2, 'g+');
legend('Order1 center','Order2 center'); xlabel('Amplitude'); ylabel('Norm. Intensity');
title('Center curves (normalized)');

% ============ Local helpers ============
function stack = local_read_stack(fname)
    S = load(fname);
    vars = fieldnames(S);
    stack = [];
    for i = 1:numel(vars)
        v = S.(vars{i});
        if isnumeric(v) && ndims(v)==3
            if isempty(stack) || numel(v) > numel(stack)
                stack = v;
            end
        end
    end
    if isempty(stack)
        error('在 %s 找不到 3D 影像堆疊變數。', fname);
    end
    stack = single(stack);
end

function avals = local_get_a_values(fname, Na)
    S = load(fname);
    if isfield(S,'a_values')
        avals = S.a_values(:);
    elseif isfield(S,'GratingPhaseAmplitude')
        avals = S.GratingPhaseAmplitude(:);
    else
        warning('%s 找不到 a_values；預設 0:2:248', fname);
        avals = (0:2:248).';
        if numel(avals) ~= Na
            warning('a_values 長度(%d)與影格數(%d)不符，改用 1:Na。', numel(avals), Na);
            avals = (1:Na).';
        end
    end
end

% 旋轉
% ang = deg2rad(theta_small);
% R = [cos(ang) sin(ang) 0; -sin(ang) cos(ang) 0; 0 0 1];
% tform = affine2d(R * T * Tx);

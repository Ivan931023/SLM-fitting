% Main SLM Fitting Script (Vectorized + Best Params Local & Global)
clc; clear; close all;
tic;

output_folder = 'fitting_outputs';
if ~exist(output_folder, 'dir'); mkdir(output_folder); end

% ----------------------- Load Experimental Data ----------------------- %
order1_table = readtable('intensity_vs_grating_order1_total.csv', 'VariableNamingRule', 'preserve');
order2_table = readtable('intensity_vs_grating_order2_total.csv', 'VariableNamingRule', 'preserve');

cols1 = 2:width(order1_table);
cols2 = 2:width(order2_table);

a_min = 0; a_max = 248;
a_values_full = order1_table.GratingPhaseAmplitude;
a_idx_range = find(a_values_full >= a_min & a_values_full <= a_max);
a_values = a_values_full(a_idx_range);

I_exp1 = order1_table{a_idx_range, cols1};
I_exp2 = order2_table{a_idx_range, cols2};

% --------- Parse positions from headers ---------
headers1 = order1_table.Properties.VariableNames(cols1);
P1 = numel(headers1);
positions1 = zeros(P1, 2);
for i = 1:P1
    str_clean = erase(headers1{i}, {'-', ','});
    nums = sscanf(str_clean, '%d');
    num_str = num2str(nums);
    col = str2double(num_str(5:end));
    col = col - 1385;
    positions1(i,:) = [0, col];
end

headers2 = order2_table.Properties.VariableNames(cols2);
P2 = numel(headers2);
positions2 = zeros(P2, 2);
for i = 1:P2
    str_clean = erase(headers2{i}, {'-', ','});
    nums = sscanf(str_clean, '%d');
    num_str = num2str(nums);
    col = str2double(num_str(5:end));
    col = col - 1400;
    positions2(i,:) = [0, col];
end

if P1 ~= P2
    error('Order1 與 Order2 點數不同');
end
P = P1;

positions1 = positions1./(8/2.2);
positions2 = positions2./(8/2.2);

% normalization
ref_center_index = 7;
[I_exp1_ref, ~] = max(order1_table{:, ref_center_index+1});
[I_exp2_ref, ~] = max(order2_table{:, ref_center_index+1});
I_exp1 = I_exp1 ./ I_exp1_ref;
I_exp2 = I_exp2 ./ I_exp2_ref;

% ----------------------- Grid Search Parameters ----------------------- %
r_range    = 0.0675:0.0025:0.075;
w_range    = (0.70:0.025:0.775) * 8e-3;
a2pi_range = 234:2:242;
alpha_range= (0:0.25:2) * pi;

% r_range    = 0.07;
% w_range    = 0.7 * 8e-3;
% a2pi_range = 234:2:236;
% alpha_range= 0:0.2:0.2;

% ----------------------- Pre-Allocate Output -------------------------- %
chi2_r_w_min_p = inf(length(r_range), length(w_range), P);
best_a2pi_per_point = zeros(length(r_range), length(w_range), P);
best_alpha_per_point = zeros(length(r_range), length(w_range), P);


chi2_r_w_ave   = zeros(length(r_range), length(w_range));
csv_data1 = [];  
csv_data2 = [];  
csv_data3 = [];  


% ----------------------- Fixed constants ------------------------------ %
lambda = 447e-6; f = 300; dx = 8e-3;
Beam_size = 2.634705;
l = 600; z = 600;
pixel_grating = 1080;

Nx = pixel_grating; Ny = pixel_grating;
x = -Nx/2*dx : dx : (Nx/2-1)*dx;
y = -Ny/2*dx : dx : (Ny/2-1)*dx;
[X, Y] = meshgrid(x, -y);

theta_deg = -90;
theta_blazed = deg2rad(theta_deg);
level = 12; repeat = 1;
min_phase = 0;
U0 = Gaussian_beam(Beam_size, pixel_grating, dx);

h = 541;
v1 = 631; shift_y1 = 427;
v2 = 721; shift_y2 = 857;
r_filter = 50;
[x1, y1] = meshgrid(1:Nx, 1:Ny);
mask1 = (x1 - h).^2 + (y1 - v1).^2 <= r_filter^2;
mask2 = (x1 - h).^2 + (y1 - v2).^2 <= r_filter^2;

row_base = 541; col_base = 541;
rows1 = row_base + round(positions1(:,1));
cols1 = col_base + round(positions1(:,2));
rows2 = row_base + round(positions2(:,1));
cols2 = col_base + round(positions2(:,2));
rows1 = max(1, min(Ny, rows1)); cols1 = max(1, min(Nx, cols1));
rows2 = max(1, min(Ny, rows2)); cols2 = max(1, min(Nx, cols2));
lin1 = sub2ind([Ny, Nx], rows1, cols1);
lin2 = sub2ind([Ny, Nx], rows2, cols2);
lin_center = sub2ind([Ny,Nx], row_base, col_base);

fprintf('Fitting ALL points...\n');

for i_r = 1:length(r_range)
    r = r_range(i_r);
    fprintf('  Trying r = %.3f (%d/%d)\n', r, i_r, length(r_range));
    for i_w = 1:length(w_range)
        w = w_range(i_w);
        fprintf('  Trying w = %.3f (%d/%d)\n', w, i_w, length(w_range));

        Gaussian_PSF = exp(-(X.^2+Y.^2)/(2*w^2));
        Gaussian_PSF = Gaussian_PSF / sum(Gaussian_PSF,'all');

        for a2pi_idx = 1:length(a2pi_range)
            a2pi = a2pi_range(a2pi_idx);
            fprintf('      Trying a2pi = %d (%d/%d)\n', a2pi, a2pi_idx, length(a2pi_range));
            for alpha_idx = 1:length(alpha_range)
                alpha = alpha_range(alpha_idx);
                fprintf('        Trying alpha = %.2f (%d/%d)\n', alpha, alpha_idx, length(alpha_range));

                I1_sim = zeros(length(a_values), P);
                I2_sim = zeros(length(a_values), P);
                I1_center_over_a = zeros(length(a_values),1);
                I2_center_over_a = zeros(length(a_values),1);

                for kk = 1:length(a_values)
                    a = a_values(kk);
                    fprintf('          Simulating amplitude %d/%d (a = %g)\n', kk, length(a_values), a);
                    Blazed_theta = Grating_phase(pixel_grating, a, min_phase, level, repeat, theta_blazed, a2pi);
                    phi = conv2(Blazed_theta, Gaussian_PSF, 'same') + alpha;
                    den = 1 + r * exp(1j*phi);
                    den(abs(den) < 1e-12) = 1e-12;
                    E = -(r + exp(1j*phi)) ./ den;
                    U_slm = U0 .* E;

                    U_len = propTF(U_slm, Nx*dx, lambda, l - f);
                    U_fourier = DFT(U_len);

                    U_f1 = zeros(size(U_fourier)); U_f1(mask1) = U_fourier(mask1);
                    U_img1 = propTF(U_f1, Nx*dx, lambda, -(z - f));
                    U_img1 = circshift(U_img1, [-shift_y1, 1]);
                    im1 = abs(U_img1).^2;

                    U_f2 = zeros(size(U_fourier)); U_f2(mask2) = U_fourier(mask2);
                    U_img2 = propTF(U_f2, Nx*dx, lambda, -(z - f));
                    U_img2 = circshift(U_img2, [-shift_y2, 1]);
                    im2 = abs(U_img2).^2;

                    I1_sim(kk, :) = im1(lin1);
                    I2_sim(kk, :) = im2(lin2);
                    I1_center_over_a(kk) = im1(lin_center);
                    I2_center_over_a(kk) = im2(lin_center);
                end

                n1 = max(I1_center_over_a); if n1 <= 0, n1 = 1; end
                n2 = max(I2_center_over_a); if n2 <= 0, n2 = 1; end
                I1_sim = I1_sim / n1;
                I2_sim = I2_sim / n2;

                chi2_vec = sum( ((I_exp1 - I1_sim).^2) ./ (I1_sim + 1e-6) + ...
                                ((I_exp2 - I2_sim).^2) ./ (I2_sim + 1e-6), 1 );
                % chi2_vec: 1×P，這一組 (r,w,a2pi,alpha) 對每個點 p 的 χ²
                better = chi2_vec < squeeze(chi2_r_w_min_p(i_r, i_w, :)).';  % 1×P 邏輯陣列
                chi2_r_w_min_p(i_r, i_w, better)      = chi2_vec(better);
                best_a2pi_per_point(i_r, i_w, better)  = a2pi;
                best_alpha_per_point(i_r, i_w, better) = alpha;


                csv_data1 = [csv_data1; [repmat([r, w, a2pi, alpha], P, 1), (1:P).', chi2_vec(:)]];
            
            end
        end

        chi2_r_w_ave(i_r, i_w) = mean(chi2_r_w_min_p(i_r, i_w, :), 'all', 'omitnan');
        csv_data2 = [csv_data2; [r, w, chi2_r_w_ave(i_r, i_w)]];
    end
end

% === 輸出 ===
writematrix(csv_data1, fullfile(output_folder, 'chi2_total_all_points.csv'));
writematrix(csv_data2, fullfile(output_folder, 'chi2_min_all_points.csv'));

for i_r = 1:length(r_range)
    for i_w = 1:length(w_range)
        for p = 1:P
            a2pi_val = best_a2pi_per_point(i_r, i_w, p);
            alpha_val = best_alpha_per_point(i_r, i_w, p);
            csv_data3 = [csv_data3; r_range(i_r), w_range(i_w), p, a2pi_val, alpha_val];
        end
    end
end

writematrix(csv_data3, fullfile(output_folder, 'best_params_per_point.csv'));

% 存成 table
T = array2table(csv_data3, ...
    'VariableNames', {'r', 'w', 'point_index', 'best_a2pi', 'best_alpha'});


writetable(T, fullfile(output_folder, 'best_params_per_point_table.csv'));

toc;






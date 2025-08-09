% function phase = Grating_phase(pixel, max_phase, min_phase, level, repeat, theta)
%     [X, Y] = meshgrid(1:pixel, 1:pixel);
%     phase = mod(2*pi*repeat * (cos(theta)*X + sin(theta)*Y) / pixel, 2*pi);
%     phase = min_phase + (max_phase - min_phase) * phase / (2*pi);
%     phase = 2*pi * (phase - min_phase) / (max_phase - min_phase);
% end
function grat = Grating_phase(pixel, max_phase, min_phase, level, repeat, theta_blazed, a2pi)
    grat = Blazed_grating_rotate(pixel, max_phase, min_phase, level, repeat, theta_blazed);
    grat = 2*pi/a2pi * grat;
end

function grating = Blazed_grating_rotate(pixel, max, min, levels, repeat, theta)
    x = 1:pixel;
    y = x;
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    [X_rot, Y_rot] = meshgrid(x, y);
    coords = R * [X_rot(:)'; Y_rot(:)'];
    X_rot = reshape(coords(1, :), size(X_rot));
    Y_rot = reshape(coords(2, :), size(Y_rot));
    
    blazed_grating_function = @(x) mod(floor(x / repeat), levels) * (max - min) / (levels - 1) + min;
    grating = blazed_grating_function(X_rot);
end
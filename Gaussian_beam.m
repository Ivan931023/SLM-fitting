function U = Gaussian_beam(Beam_size, pixel, dx)
    w0 = (Beam_size/2)/(sqrt(2));
    x = linspace(-pixel/2*dx, pixel/2*dx, pixel);
    [X, Y] = meshgrid(x, -x);
    r2 = X.^2 + Y.^2;
    U = exp(-r2 / (w0^2));
    U = U ./ max(U(:));
end

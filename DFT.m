function result = DFT(u)
    result = fftshift(fft2(ifftshift(u)));
end
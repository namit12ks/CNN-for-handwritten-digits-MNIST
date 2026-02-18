function y = Conv(x,w)


[wrow, wcol, numFilters] = size (w) ; % kernel
[xrow, xcol,  ~        ] = size (x) ; % input/image 

 
yrow = xrow - wrow + 1;
ycol = xcol - wcol + 1;

y = zeros (yrow, ycol, numFilters);

    for k = 1:numFilters
    
         filter = w(:, :, k);
         filter = rot90 (squeeze (filter), 2);
    
        y(:, :, k) = conv2(x, filter,'valid');
    
    end
 
end 
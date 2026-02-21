





load ('MnistConv.mat')

% Forward pass for one image 

    k = 11;           % which  image no
    x = X(:, :, k);
    y1 = Conv(x, W1);      % convoution 
    y2 = ReLU(y1);          % relu  
    y3 = Pool(y2);         % pool 
    y4 = reshape(y3, [], 1);          
    v5 = W5*y4;                 
    y5 = ReLU (v5);             
    v = Wo*y5;                  
    y = Softmax(v) ;            % softmax,    


figure;
display_network (x (:)) ;
title ('Input Image')

convFilters = zeros(5*5, 32);

    for i = 1:32

    filter = W1(:, :, i) ;
    convFilters(:, i) = filter (:);

    end

figure
display_network (convFilters);
title ('Convolution Filters')




fList = zeros(24*24, 32) ;

for i = 1:32
    feature = y1 (:, :, i) ;
    fList(:,i)= feature(:);
end


figure
display_network(fList);
title('Features [Convolution]')

fList = zeros(24*24, 32);

for i = 1:32
    feature = y2(:, :, i);
    fList(:, i) = feature (:);
end


figure
display_network(fList);
title('Features [Convolution + ReLU]')

fList = zeros(12*12, 32);

for i = 1:32
    feature = y3 (:, :, i);
    fList (:, i) = feature (:);
end

figure
display_network (fList);
title('Features [Convolution + ReLU + MeanPool] ')





